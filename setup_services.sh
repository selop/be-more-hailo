#!/bin/bash

BMO_USER="$USER"
BMO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Setting up BMO background service..."
echo "  User: $BMO_USER"
echo "  Directory: $BMO_DIR"

# 1. Create the GUI Service (agent and live face)
#
# Machine-specific settings (ALSA_DEVICE, MIC_DEVICE_INDEX, BMO_LANGUAGE, ...)
# live in Environment= lines of the installed unit. If a unit already exists,
# its Environment= lines are kept so re-running this script does not reset them.
UNIT=/etc/systemd/system/bmo-gui.service
BMO_UID="$(id -u "$BMO_USER")"
if [ -f "$UNIT" ] && grep -q '^Environment=' "$UNIT"; then
    echo "Keeping Environment= settings from the existing bmo-gui.service"
    ENV_LINES="$(grep '^Environment=' "$UNIT")"
else
    ENV_LINES="Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/$BMO_USER/.Xauthority
Environment=XDG_RUNTIME_DIR=/run/user/$BMO_UID"
fi

echo "Creating bmo-gui.service..."
cat << EOF | sudo tee "$UNIT"
[Unit]
Description=BMO GUI Agent (Hailo NPU)
After=display-manager.service sound.target bluetooth.target

[Service]
Type=simple
User=$BMO_USER
Group=$BMO_USER
WorkingDirectory=$BMO_DIR
$ENV_LINES
ExecStart=$BMO_DIR/start_agent.sh
# The agent exits with a failure status when boot fails (for example a
# model load that timed out), so on-failure restarts it.
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
# graphical.target, not graphical-session.target: the latter is a user-session
# target and never starts a system service at boot.
WantedBy=graphical.target
EOF

# 2. Fix Permissions
echo "Fixing permissions..."
sudo chown -R "$BMO_USER:$BMO_USER" "$BMO_DIR"
sudo chmod +x "$BMO_DIR"/*.sh

# 3. Remove the retired web UI service if it exists
if [ -f /etc/systemd/system/bmo-web.service ]; then
    echo "Removing retired bmo-web.service (web interface was dropped)..."
    sudo systemctl stop bmo-web.service 2>/dev/null
    sudo systemctl disable bmo-web.service 2>/dev/null
    sudo rm -f /etc/systemd/system/bmo-web.service
fi

# 4. Remove legacy hailo-ollama service if it exists
if [ -f /etc/systemd/system/bmo-ollama.service ]; then
    echo "Removing legacy bmo-ollama.service (hailo-ollama no longer needed)..."
    sudo systemctl stop bmo-ollama.service 2>/dev/null
    sudo systemctl disable bmo-ollama.service 2>/dev/null
    sudo rm -f /etc/systemd/system/bmo-ollama.service
fi

# 5. Reload systemd and enable the service
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling service to start on boot..."
sudo systemctl enable bmo-gui.service

echo "Starting service now..."
sudo systemctl start bmo-gui.service

echo ""
echo "========================================================"
echo "Setup Complete!"
echo "BMO GUI is now running in the background."
echo "It will automatically start whenever the Pi reboots."
echo "========================================================"
echo "To check status:  sudo systemctl status bmo-gui"
echo "To view logs:      journalctl -u bmo-gui -f"
echo "========================================================"
