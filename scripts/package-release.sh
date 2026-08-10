#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:?usage: package-release.sh <rust-target> <sherpa-asset>}
SHERPA_ASSET=${2:?usage: package-release.sh <rust-target> <sherpa-asset>}
SHERPA_VERSION=1.13.4
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PACKAGE_NAME="say-the-rest-${TARGET}"
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
  *windows*) APP_BINARY=say-the-rest.exe; SERVICE_BINARY=say-the-rest-service.exe; DESKTOP_BINARY=say-the-rest-desktop.exe ;;
  *) APP_BINARY=say-the-rest; SERVICE_BINARY=say-the-rest-service; DESKTOP_BINARY=say-the-rest-desktop ;;
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
  chmod +x "$STAGE/setup.sh" "$STAGE/say-the-rest" "$STAGE/say-the-rest-service" "$STAGE/say-the-rest-desktop"
  tar -czf "$ROOT_DIR/dist/$PACKAGE_NAME.tar.gz" -C "$ROOT_DIR/dist" "$PACKAGE_NAME"

  VERSION=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$ROOT_DIR/Cargo.toml" | head -n 1)
  DEB_ROOT="$ROOT_DIR/dist/deb-root"
  rm -rf "$DEB_ROOT"
  mkdir -p \
    "$DEB_ROOT/DEBIAN" \
    "$DEB_ROOT/opt/say-the-rest" \
    "$DEB_ROOT/usr/bin" \
    "$DEB_ROOT/usr/lib/systemd/user" \
    "$DEB_ROOT/usr/share/applications" \
    "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps"
  cp -a "$STAGE/runtime" "$DEB_ROOT/opt/say-the-rest/"
  cp "$STAGE/say-the-rest" "$STAGE/say-the-rest-service" "$STAGE/say-the-rest-desktop" \
    "$STAGE/LICENSE" "$STAGE/README.md" "$DEB_ROOT/opt/say-the-rest/"
  cp "$ROOT_DIR/packaging/linux/say-the-rest.service" \
    "$ROOT_DIR/packaging/linux/say-the-rest-desktop.service" \
    "$DEB_ROOT/usr/lib/systemd/user/"
  cp "$ROOT_DIR/packaging/linux/say-the-rest-launcher" "$DEB_ROOT/usr/bin/say-the-rest-desktop"
  cp "$ROOT_DIR/packaging/linux/say-the-rest.desktop" "$DEB_ROOT/usr/share/applications/"
  cp "$ROOT_DIR/apps/desktop/src-tauri/icons/icon.png" \
    "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps/say-the-rest.png"
  ln -s /opt/say-the-rest/say-the-rest "$DEB_ROOT/usr/bin/say-the-rest"
  chmod 755 "$DEB_ROOT/opt/say-the-rest/say-the-rest" \
    "$DEB_ROOT/opt/say-the-rest/say-the-rest-service" \
    "$DEB_ROOT/opt/say-the-rest/say-the-rest-desktop" \
    "$DEB_ROOT/usr/bin/say-the-rest-desktop"
  chmod 644 "$DEB_ROOT/opt/say-the-rest/LICENSE" "$DEB_ROOT/opt/say-the-rest/README.md" \
    "$DEB_ROOT/usr/lib/systemd/user/say-the-rest.service" \
    "$DEB_ROOT/usr/lib/systemd/user/say-the-rest-desktop.service" \
    "$DEB_ROOT/usr/share/applications/say-the-rest.desktop" \
    "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps/say-the-rest.png"
  cat > "$DEB_ROOT/DEBIAN/control" <<EOF
Package: say-the-rest
Version: $VERSION
Section: sound
Priority: optional
Architecture: amd64
Maintainer: a1denvalu3
Depends: libgtk-3-0t64 | libgtk-3-0, libwebkit2gtk-4.1-0, libayatana-appindicator3-1, libxdo3
Description: Private local text-to-speech for Linux
 Read selected text aloud with offline models and local voice cloning.
EOF
  dpkg-deb --root-owner-group --build "$DEB_ROOT" \
    "$ROOT_DIR/dist/say-the-rest_${VERSION}_amd64.deb"

  APPDIR="$ROOT_DIR/dist/SayTheRest.AppDir"
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps" "$APPDIR/usr/share/metainfo" "$APPDIR/usr/share/say-the-rest"
  cp -a "$STAGE/runtime" "$APPDIR/usr/bin/"
  cp "$STAGE/say-the-rest" "$STAGE/say-the-rest-service" "$STAGE/say-the-rest-desktop" "$APPDIR/usr/bin/"
  cp "$ROOT_DIR/packaging/linux-appimage/AppRun" "$APPDIR/AppRun"
  cp "$ROOT_DIR/packaging/linux/say-the-rest.desktop" "$APPDIR/usr/share/applications/say-the-rest.desktop"
  cp "$ROOT_DIR/apps/desktop/src-tauri/icons/icon.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/say-the-rest.png"
  cp "$ROOT_DIR/packaging/linux/say-the-rest.metainfo.xml" \
    "$APPDIR/usr/share/metainfo/sh.saytherest.desktop.metainfo.xml"
  cp "$ROOT_DIR/packaging/linux-appimage/say-the-rest-appimage.service" "$APPDIR/usr/share/say-the-rest/"
  cp "$ROOT_DIR/packaging/linux-appimage/say-the-rest-appimage-desktop.service" "$APPDIR/usr/share/say-the-rest/"
  chmod 755 "$APPDIR/AppRun" "$APPDIR/usr/bin/"say-the-rest*
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
  APPIMAGE_OUTPUT="$ROOT_DIR/dist/SayTheRest-${VERSION}-x86_64.AppImage"
  APPIMAGE_TEMP="$ROOT_DIR/dist/SayTheRest-${VERSION}-x86_64.new.AppImage"
  rm -f "$APPIMAGE_TEMP"
  APPIMAGE_EXTRACT_AND_RUN=1 \
    "$LINUXDEPLOY" --appdir "$APPDIR" \
      --executable "$APPDIR/usr/bin/say-the-rest-desktop" \
      --executable "$APPDIR/usr/bin/say-the-rest-service" \
      --desktop-file "$APPDIR/usr/share/applications/say-the-rest.desktop" \
      --icon-file "$APPDIR/usr/share/icons/hicolor/256x256/apps/say-the-rest.png"
  APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 \
    "$APPIMAGETOOL" "$APPDIR" "$APPIMAGE_TEMP"
  mv -f "$APPIMAGE_TEMP" "$APPIMAGE_OUTPUT"
fi
