#!/usr/bin/env sh
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_NAME=vits-piper-en_US-lessac-medium
MODEL_ARCHIVE="$MODEL_NAME.tar.bz2"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$MODEL_ARCHIVE"

mkdir -p "$APP_DIR/models"
if [ ! -f "$APP_DIR/models/$MODEL_NAME/en_US-lessac-medium.onnx" ]; then
  echo "Downloading the default English voice (about 60 MB)..."
  curl --fail --location --progress-bar "$MODEL_URL" --output "$APP_DIR/models/$MODEL_ARCHIVE"
  tar -xjf "$APP_DIR/models/$MODEL_ARCHIVE" -C "$APP_DIR/models"
  rm "$APP_DIR/models/$MODEL_ARCHIVE"
fi

cat > "$APP_DIR/say-the-rest.json" <<EOF
{
  "engine": "sherpa-onnx-vits",
  "executable": "$APP_DIR/runtime/bin/sherpa-onnx-offline-tts",
  "model": "$APP_DIR/models/$MODEL_NAME/en_US-lessac-medium.onnx",
  "tokens": "$APP_DIR/models/$MODEL_NAME/tokens.txt",
  "data_dir": "$APP_DIR/models/$MODEL_NAME/espeak-ng-data",
  "provider": "cpu",
  "num_threads": 4,
  "speaker_id": 0
}
EOF

if command -v systemctl >/dev/null 2>&1; then
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/say-the-rest.service" <<EOF
[Unit]
Description=Say the Rest local speech service
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory="$APP_DIR"
ExecStart="$APP_DIR/say-the-rest-service" --config "$APP_DIR/say-the-rest.json"
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now say-the-rest.service
else
  "$APP_DIR/say-the-rest-service" --config "$APP_DIR/say-the-rest.json" >/dev/null 2>&1 &
fi

AUTOSTART_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
mkdir -p "$AUTOSTART_DIR"
cat > "$AUTOSTART_DIR/say-the-rest.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Say the Rest
Comment=Global offline text-to-speech shortcuts
Exec="$APP_DIR/say-the-rest-desktop"
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

"$APP_DIR/say-the-rest-desktop" >/dev/null 2>&1 &
echo "Setup complete. Say the Rest is running in your system tray."
