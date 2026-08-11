#!/usr/bin/env sh
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v systemctl >/dev/null 2>&1; then
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/sayit.service" <<EOF
[Unit]
Description=sayIt local speech service
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory="$APP_DIR"
ExecStart="$APP_DIR/sayit-service"
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF
  cat > "$UNIT_DIR/sayit-desktop.service" <<EOF
[Unit]
Description=sayIt shortcuts and tray
After=graphical-session.target sayit.service
Wants=sayit.service

[Service]
Type=simple
WorkingDirectory="$APP_DIR"
ExecStart="$APP_DIR/sayit-desktop"
Environment=LD_LIBRARY_PATH=$APP_DIR/runtime/lib
Restart=on-failure
RestartSec=2

[Install]
WantedBy=graphical-session.target
EOF
  systemctl --user daemon-reload
  systemctl --user disable --now say-the-rest.service say-the-rest-desktop.service >/dev/null 2>&1 || true
  systemctl --user enable sayit.service sayit-desktop.service
  systemctl --user stop sayit-desktop.service >/dev/null 2>&1 || true
  pkill -TERM -u "$(id -u)" -f "$APP_DIR/sayit-desktop" >/dev/null 2>&1 || true
  systemctl --user restart sayit.service sayit-desktop.service
else
  "$APP_DIR/sayit-service" --config "$APP_DIR/sayit.json" >/dev/null 2>&1 &
  AUTOSTART_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
  mkdir -p "$AUTOSTART_DIR"
  cat > "$AUTOSTART_DIR/sayit.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=sayIt
Comment=Global offline text-to-speech shortcuts
Exec="$APP_DIR/sayit-desktop"
Terminal=false
X-GNOME-Autostart-enabled=true
EOF
  "$APP_DIR/sayit-desktop" >/dev/null 2>&1 &
fi
echo "Setup complete. sayIt is running in your system tray. Choose a model in onboarding to download it with integrity verification."
