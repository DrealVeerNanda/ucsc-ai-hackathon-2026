#!/usr/bin/env bash
# Put the Proprio icon on the GX10's desktop (and in the applications list). Double-clicking it
# opens the start page: see launcher.py. No sudo needed. Safe to run again.
#
#   bash tools/install_launcher.sh                 # for the checkout this script is in
#   bash tools/install_launcher.sh --remove        # take the icon away again
#   bash tools/install_launcher.sh --boot          # also bring the demo back by itself when the GX10 starts
#   bash tools/install_launcher.sh --no-boot       # ...and not any more
#   NAME="Proprio test" bash tools/install_launcher.sh   # under another name (two checkouts side by side)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${NAME:-Proprio}"
FILE="$(echo "$NAME" | tr 'A-Z ' 'a-z-').desktop"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
APPS="$HOME/.local/share/applications"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"   # gio needs it over SSH

# The same without anyone at the keyboard, when the GX10 starts: see tools/boot_start.py.
UNIT="$HOME/.config/systemd/user/proprio-boot.service"
if [ "${1:-}" = "--boot" ]; then
    mkdir -p "$(dirname "$UNIT")"
    cat > "$UNIT" <<EOF
[Unit]
Description=Proprio: bring the demo back when the GX10 starts (tools/boot_start.py)
After=pipewire.service wireplumber.service

[Service]
Type=oneshot
WorkingDirectory=$REPO
ExecStart="$REPO/.venv/bin/python" "$REPO/tools/boot_start.py"
KillMode=process
TimeoutStartSec=2100

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable proprio-boot.service
    loginctl show-user "$(id -un)" -p Linger | grep -q yes || echo "note: without linger (sudo loginctl enable-linger $(id -un)) it only runs once someone logs in"
    echo "installed: the demo comes back by itself when this computer starts (undo: $0 --no-boot)"
    exit 0
fi
if [ "${1:-}" = "--no-boot" ]; then
    systemctl --user disable proprio-boot.service 2>/dev/null || true
    rm -f "$UNIT"
    systemctl --user daemon-reload
    echo "removed: nothing starts by itself any more"
    exit 0
fi

if [ "${1:-}" = "--remove" ]; then
    rm -f "$DESKTOP/$FILE" "$APPS/$FILE"
    echo "removed $FILE from the desktop and the applications list"
    exit 0
fi
[ -x "$REPO/.venv/bin/python" ] || { echo "no Python environment at $REPO/.venv" >&2; exit 1; }

mkdir -p "$DESKTOP" "$APPS"
for target in "$DESKTOP/$FILE" "$APPS/$FILE"; do
    cat > "$target" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$NAME
Comment=Check everything is ready, then start the demo
Exec="$REPO/.venv/bin/python" "$REPO/launcher.py"
Path=$REPO
Icon=$REPO/dashboard/icon.svg
Terminal=false
StartupNotify=false
Categories=Utility;
EOF
    chmod +x "$target"
done
# GNOME only starts a desktop file by double-click once it is marked trusted (the "Allow Launching"
# in its right-click menu), and the icon redraws when the file's time changes.
gio set "$DESKTOP/$FILE" metadata::trusted true
touch "$DESKTOP/$FILE"
update-desktop-database "$APPS" 2>/dev/null || true
echo "installed: double-click \"$NAME\" on the desktop (or run: gtk-launch ${FILE%.desktop})"
