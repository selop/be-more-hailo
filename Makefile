# Developer tasks. Run `make` or `make help` for the list.
#
# The first group runs on this machine; the second talks to the Pi over SSH.
# Choose another device with e.g. `make deploy BMO_HOST=selop@bmo`.

BMO_HOST ?= selop@raspi
BMO_DIR  ?= Git/be-more-hailo
PY       ?= python3
VENV     ?= .venv-dev

export BMO_HOST BMO_DIR

# Hardware-free tests (tests/test_wakeword.py needs the model and a Pi)
TESTS := tests/test_character.py tests/test_turn.py tests/test_brain.py \
         tests/test_timers.py tests/test_display.py tests/test_monitor.py

# What a deploy copies. rsync never deletes on the Pi; remove files there by hand.
DEPLOY := core/ face/ characters/ scripts/ tests/ docs/ agent_hailo.py cli_chat.py \
          requirements.txt setup.sh setup_services.sh start_agent.sh CLAUDE.md README.md Makefile

SSH    := ssh -o BatchMode=yes -o ConnectTimeout=8 $(BMO_HOST)
REMOTE := cd $(BMO_DIR) &&

.DEFAULT_GOAL := help
.PHONY: help setup test check clean face face-demo \
        deploy pi-test restart ship status temps monitor monitor-60 logs turns stats events say screenshot ssh

help: ## Show this list
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Device: $(BMO_HOST):$(BMO_DIR)   (override with BMO_HOST=... BMO_DIR=...)"

##@ On this machine

setup: ## Create .venv-dev with what the tests and tools need
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --quiet --upgrade pip
	$(VENV)/bin/pip install --quiet -r requirements-dev.txt
	@echo "Done. Use it with: make test PY=$(VENV)/bin/python"

test: ## Run the hardware-free tests
	@fail=0; for t in $(TESTS); do \
	  out=$$($(PY) $$t 2>&1); status=$$?; \
	  printf '%-28s %s passed' "$$t" "$$(printf '%s\n' "$$out" | grep -c '^PASS')"; \
	  if [ $$status -ne 0 ]; then printf '  FAILED\n'; printf '%s\n' "$$out" | grep '^FAIL'; fail=1; else printf '\n'; fi; \
	done; exit $$fail

check: ## Compile every Python file, then run the tests
	@$(PY) -m compileall -q core scripts tests agent_hailo.py cli_chat.py
	@$(MAKE) --no-print-directory test

clean: ## Remove Python caches
	@$(PY) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if not {'venv', '$(VENV)'} & set(p.parts)]"
	@echo "Caches removed."

face: ## Open the face page as a file (arrow keys step through the faces)
	open face/index.html

face-demo: ## Serve the face locally and feed it made-up engine events
	$(PY) scripts/face_demo.py

##@ On the Pi

deploy: ## Copy the code to the Pi (does not restart)
	rsync -az --relative --exclude '__pycache__' $(DEPLOY) $(BMO_HOST):$(BMO_DIR)/
	@echo "Deployed to $(BMO_HOST). Run 'make restart' to load it."

pi-test: ## Run the hardware-free tests on the Pi
	@$(SSH) '$(REMOTE) fail=0; for t in $(TESTS); do out=$$(venv/bin/python $$t 2>&1) || fail=1; printf "%-28s %s passed\n" "$$t" "$$(printf "%s\n" "$$out" | grep -c "^PASS")"; printf "%s\n" "$$out" | grep "^FAIL"; done; exit $$fail'

restart: ## Restart the agent and wait until it is ready
	@$(SSH) 'sudo systemctl restart bmo-gui'
	@echo "Restarted; waiting for boot..."
	@n=0; while [ $$n -lt 30 ]; do \
	  sleep 4; n=$$((n + 1)); \
	  if $(SSH) '$(REMOTE) venv/bin/python scripts/bmoctl.py status 2>/dev/null' | grep -q '"llm_ready": true'; then echo "Ready."; exit 0; fi; \
	done; echo "Not ready after 2 minutes; see: make logs"; exit 1

ship: check deploy pi-test restart ## Test here, deploy, test on the Pi, restart

status: ## Phase, models, microphone and board readings
	@scripts/bmo-remote.sh status

temps: ## One reading of temperature, fan, memory and microphone
	@$(PY) scripts/bmo_monitor.py --once

monitor: ## Live monitoring dashboard (Ctrl+C for a summary)
	@$(PY) scripts/bmo_monitor.py

monitor-60: ## Watch for a minute, then print min/median/max
	@$(PY) scripts/bmo_monitor.py --duration 60

logs: ## Recent service log: make logs N=200
	@scripts/bmo-remote.sh logs -n $(or $(N),80)

turns: ## Recent turns: make turns N=20
	@scripts/bmo-remote.sh turns -n $(or $(N),10)

stats: ## Timing summary over recent turns
	@scripts/bmo-remote.sh stats

events: ## Boot, timer, music and error events
	@scripts/bmo-remote.sh events

say: ## Run a silent test turn: make say TEXT="what time is it" (LOUD=1 to hear it)
	@test -n "$(TEXT)" || { echo 'Usage: make say TEXT="what time is it"'; exit 2; }
	@scripts/bmo-remote.sh say "$(TEXT)" $(if $(LOUD),,--quiet)

screenshot: ## Capture the Pi's display and open it
	@f=$$(scripts/bmo-remote.sh screenshot) && echo "$$f" && open "$$f"

ssh: ## Open a shell on the Pi in the project folder
	@ssh -t $(BMO_HOST) 'cd $(BMO_DIR) && exec $$SHELL -l'
