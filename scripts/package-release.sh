#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:?usage: package-release.sh <rust-target> <sherpa-asset>}
SHERPA_ASSET=${2:?usage: package-release.sh <rust-target> <sherpa-asset>}
SHERPA_VERSION=1.13.4
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PACKAGE_NAME="sayit-${TARGET}"
STAGE="$ROOT_DIR/dist/$PACKAGE_NAME"
ARCHIVE="$ROOT_DIR/dist/$SHERPA_ASSET"

case "$SHERPA_ASSET" in
  sherpa-onnx-v1.13.4-linux-x64-shared.tar.bz2)
    SHERPA_SHA256=18887dc13c7d313d0e0f6c164ed31715c27c1c2c4f71acd7c0147dc84cf02514
    ;;
  sherpa-onnx-v1.13.4-win-x64-shared-MT-Release.tar.bz2)
    SHERPA_SHA256=c312b30e55258d291067537ce4a6f90e155f1c16b6a10381a5729924e98b7879
    ;;
  *)
    echo "Unsupported or unpinned sherpa-onnx release asset: $SHERPA_ASSET" >&2
    exit 1
    ;;
esac

verify_sha256() {
  expected=$1
  file=$2
  actual=$(sha256sum "$file" | cut -d ' ' -f 1)
  if [ "$actual" != "$expected" ]; then
    echo "SHA-256 mismatch for $file: expected $expected, received $actual" >&2
    exit 1
  fi
}

rm -rf "$STAGE"
mkdir -p "$STAGE/runtime" "$STAGE/icons" "$ROOT_DIR/dist"
cargo build --release --locked --target "$TARGET" --manifest-path "$ROOT_DIR/Cargo.toml"

case "$TARGET" in
  *windows*) APP_BINARY=sayit.exe; SERVICE_BINARY=sayit-service.exe; DESKTOP_BINARY=sayit-desktop.exe ;;
  *) APP_BINARY=sayit; SERVICE_BINARY=sayit-service; DESKTOP_BINARY=sayit-desktop ;;
esac
cp "$ROOT_DIR/target/$TARGET/release/$APP_BINARY" "$STAGE/"
cp "$ROOT_DIR/target/$TARGET/release/$SERVICE_BINARY" "$STAGE/"
cp "$ROOT_DIR/target/$TARGET/release/$DESKTOP_BINARY" "$STAGE/"
cp "$ROOT_DIR/LICENSE" "$ROOT_DIR/README.md" "$STAGE/"
cp "$ROOT_DIR/apps/desktop/src-tauri/icons/icon.ico" "$STAGE/icons/"

curl --fail --location --retry 3 \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/v${SHERPA_VERSION}/${SHERPA_ASSET}" \
  --output "$ARCHIVE"
verify_sha256 "$SHERPA_SHA256" "$ARCHIVE"
tar -xjf "$ARCHIVE" -C "$STAGE/runtime" --strip-components=1
mkdir -p "$STAGE/runtime/generative"
cp "$ROOT_DIR/runtime/qwen_worker.py" "$STAGE/runtime/generative/"
cp "$ROOT_DIR/runtime/qwen-runtime/pyproject.toml" \
  "$ROOT_DIR/runtime/qwen-runtime/uv.lock" "$STAGE/runtime/generative/"
rm "$ARCHIVE"

if [[ "$TARGET" == *windows* ]]; then
  cp "$ROOT_DIR/packaging/setup.ps1" "$STAGE/"
  (cd "$ROOT_DIR/dist" && 7z a -tzip "$PACKAGE_NAME.zip" "$PACKAGE_NAME" >/dev/null)
  WINDOWS_STAGE=$(cygpath -w "$STAGE")
  WINDOWS_OUTPUT=$(cygpath -w "$ROOT_DIR/dist")
  WINDOWS_ISS=$(cygpath -w "$ROOT_DIR/packaging/windows-installer.iss")
  STR_STAGE="$WINDOWS_STAGE" STR_OUTPUT="$WINDOWS_OUTPUT" STR_ISS="$WINDOWS_ISS" powershell -NoProfile -Command \
    '$compiler = (Get-Command ISCC.exe -ErrorAction Stop).Source; & $compiler "/DStage=$env:STR_STAGE" "/DOutputDir=$env:STR_OUTPUT" $env:STR_ISS; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }'
else
  cp "$ROOT_DIR/packaging/setup.sh" "$STAGE/"
  chmod +x "$STAGE/setup.sh" "$STAGE/sayit" "$STAGE/sayit-service" "$STAGE/sayit-desktop"
  tar -czf "$ROOT_DIR/dist/$PACKAGE_NAME.tar.gz" -C "$ROOT_DIR/dist" "$PACKAGE_NAME"

  VERSION=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$ROOT_DIR/Cargo.toml" | head -n 1)
  DEB_ROOT="$ROOT_DIR/dist/deb-root"
  rm -rf "$DEB_ROOT"
  mkdir -p \
    "$DEB_ROOT/DEBIAN" \
    "$DEB_ROOT/opt/sayit" \
    "$DEB_ROOT/usr/bin" \
    "$DEB_ROOT/usr/lib/systemd/user" \
    "$DEB_ROOT/usr/share/applications" \
    "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps"
  cp -a "$STAGE/runtime" "$DEB_ROOT/opt/sayit/"
  cp "$STAGE/sayit" "$STAGE/sayit-service" "$STAGE/sayit-desktop" \
    "$STAGE/LICENSE" "$STAGE/README.md" "$DEB_ROOT/opt/sayit/"
  cp "$ROOT_DIR/packaging/linux/sayit.service" \
    "$ROOT_DIR/packaging/linux/sayit-desktop.service" \
    "$DEB_ROOT/usr/lib/systemd/user/"
  cp "$ROOT_DIR/packaging/linux/sayit-launcher" "$DEB_ROOT/usr/bin/sayit-desktop"
  cp "$ROOT_DIR/packaging/linux/sayit.desktop" "$DEB_ROOT/usr/share/applications/"
  cp "$ROOT_DIR/apps/desktop/src-tauri/icons/icon.png" \
    "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps/sayit.png"
  ln -s /opt/sayit/sayit "$DEB_ROOT/usr/bin/sayit"
  chmod 755 "$DEB_ROOT/opt/sayit/sayit" \
    "$DEB_ROOT/opt/sayit/sayit-service" \
    "$DEB_ROOT/opt/sayit/sayit-desktop" \
    "$DEB_ROOT/usr/bin/sayit-desktop"
  chmod 644 "$DEB_ROOT/opt/sayit/LICENSE" "$DEB_ROOT/opt/sayit/README.md" \
    "$DEB_ROOT/usr/lib/systemd/user/sayit.service" \
    "$DEB_ROOT/usr/lib/systemd/user/sayit-desktop.service" \
    "$DEB_ROOT/usr/share/applications/sayit.desktop" \
    "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps/sayit.png"
  cat > "$DEB_ROOT/DEBIAN/control" <<EOF
Package: sayit
Version: $VERSION
Section: sound
Priority: optional
Architecture: amd64
Maintainer: a1denvalu3
Provides: say-the-rest
Conflicts: say-the-rest
Replaces: say-the-rest
Depends: libgtk-3-0t64 | libgtk-3-0, libwebkit2gtk-4.1-0, libayatana-appindicator3-1, libxdo3
Description: Private local text-to-speech for Linux
 Read selected text aloud with offline models and local voice cloning.
EOF
  dpkg-deb --root-owner-group --build "$DEB_ROOT" \
    "$ROOT_DIR/dist/sayit_${VERSION}_amd64.deb"

  APPDIR="$ROOT_DIR/dist/sayIt.AppDir"
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps" "$APPDIR/usr/share/metainfo" "$APPDIR/usr/share/sayit"
  cp -a "$STAGE/runtime" "$APPDIR/usr/bin/"
  cp "$STAGE/sayit" "$STAGE/sayit-service" "$STAGE/sayit-desktop" "$APPDIR/usr/bin/"
  cp "$ROOT_DIR/packaging/linux-appimage/AppRun" "$APPDIR/AppRun"
  cp "$ROOT_DIR/packaging/linux/sayit.desktop" "$APPDIR/usr/share/applications/sayit.desktop"
  cp "$ROOT_DIR/apps/desktop/src-tauri/icons/icon.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/sayit.png"
  cp "$ROOT_DIR/packaging/linux/sayit.metainfo.xml" \
    "$APPDIR/usr/share/metainfo/sh.sayit.desktop.metainfo.xml"
  cp "$ROOT_DIR/packaging/linux-appimage/sayit-appimage.service" "$APPDIR/usr/share/sayit/"
  cp "$ROOT_DIR/packaging/linux-appimage/sayit-appimage-desktop.service" "$APPDIR/usr/share/sayit/"
  chmod 755 "$APPDIR/AppRun" "$APPDIR/usr/bin/"sayit*
  LINUXDEPLOY="$ROOT_DIR/dist/linuxdeploy-x86_64.AppImage"
  LINUXDEPLOY_SHA256=421ca71d5c69ea97c6309276232990d43df1dcece0edfaa26bbf926ff96ed12e
  curl --fail --location --retry 3 \
    "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage" \
    --output "$LINUXDEPLOY"
  verify_sha256 "$LINUXDEPLOY_SHA256" "$LINUXDEPLOY"
  chmod +x "$LINUXDEPLOY"
  APPIMAGETOOL="$ROOT_DIR/dist/appimagetool-x86_64.AppImage"
  APPIMAGETOOL_SHA256=a6d71e2b6cd66f8e8d16c37ad164658985e0cf5fcaa950c90a482890cb9d13e0
  curl --fail --location --retry 3 \
    "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" \
    --output "$APPIMAGETOOL"
  verify_sha256 "$APPIMAGETOOL_SHA256" "$APPIMAGETOOL"
  chmod +x "$APPIMAGETOOL"
  APPIMAGE_OUTPUT="$ROOT_DIR/dist/sayIt-${VERSION}-x86_64.AppImage"
  APPIMAGE_TEMP="$ROOT_DIR/dist/sayIt-${VERSION}-x86_64.new.AppImage"
  rm -f "$APPIMAGE_TEMP"
  APPIMAGE_EXTRACT_AND_RUN=1 \
    "$LINUXDEPLOY" --appdir "$APPDIR" \
      --executable "$APPDIR/usr/bin/sayit-desktop" \
      --executable "$APPDIR/usr/bin/sayit-service" \
      --desktop-file "$APPDIR/usr/share/applications/sayit.desktop" \
      --icon-file "$APPDIR/usr/share/icons/hicolor/256x256/apps/sayit.png"
  APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 \
    "$APPIMAGETOOL" "$APPDIR" "$APPIMAGE_TEMP"
  mv -f "$APPIMAGE_TEMP" "$APPIMAGE_OUTPUT"
fi
