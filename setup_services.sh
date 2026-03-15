#!/bin/bash

BMO_USER="$USER"
BMO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Setting up BMO background services..."
echo "  User: $BMO_USER"
echo "  Directory: $BMO_DIR"

# 1. Create the GUI Service (on-device Tkinter display)
echo "Creating bmo-gui.service..."
cat << EOF | sudo tee /etc/systemd/system/bmo-gui.service
[Unit]
Description=BMO GUI Agent (Tkinter fullscreen)
After=network.target

[Service]
Type=simple
User=$BMO_USER
WorkingDirectory=$BMO_DIR
Environment="DISPLAY=:0"
ExecStart=/bin/bash $BMO_DIR/start_agent.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 2. Create the Web UI Service
echo "Creating bmo-web.service..."
cat << EOF | sudo tee /etc/systemd/system/bmo-web.service
[Unit]
Description=BMO Web UI Service
After=network.target

[Service]
Type=simple
User=$BMO_USER
WorkingDirectory=$BMO_DIR
ExecStart=/bin/bash $BMO_DIR/start_web.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 3. Fix Permissions
echo "Fixing permissions for the web app..."
sudo chown -R "$BMO_USER:$BMO_USER" "$BMO_DIR"
sudo chmod +x "$BMO_DIR"/*.sh

# 4. Remove legacy hailo-ollama service if it exists
if [ -f /etc/systemd/system/bmo-ollama.service ]; then
    echo "Removing legacy bmo-ollama.service (hailo-ollama no longer needed)..."
    sudo systemctl stop bmo-ollama.service 2>/dev/null
    sudo systemctl disable bmo-ollama.service 2>/dev/null
    sudo rm -f /etc/systemd/system/bmo-ollama.service
fi

# 5. Reload systemd and enable services
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling services to start on boot..."
sudo systemctl enable bmo-gui.service
sudo systemctl enable bmo-web.service

echo "Starting services now..."
sudo systemctl start bmo-gui.service
sudo systemctl start bmo-web.service

echo ""
echo "========================================================"
echo "Setup Complete!"
echo "BMO GUI and Web UI are now running in the background."
echo "They will automatically start whenever the Pi reboots."
echo "========================================================"
echo "To check status:  sudo systemctl status bmo-gui"
echo "                   sudo systemctl status bmo-web"
echo "To view logs:      journalctl -u bmo-gui -f"
echo "========================================================"
