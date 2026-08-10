use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{str::FromStr, sync::RwLock};
use tauri::{
    Emitter, Manager,
    image::Image,
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

mod recorder;
mod selection;
use selection::CaptureKind;

const SERVICE: &str = "http://127.0.0.1:55391/v1";
const RELEASES_API: &str = "https://api.github.com/repos/a1denvalu3/SayTheRest/releases?per_page=1";
const RELEASES_PAGE: &str = "https://github.com/a1denvalu3/SayTheRest/releases";

#[derive(Clone, Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    html_url: String,
    assets: Vec<GitHubReleaseAsset>,
}

#[derive(Clone, Debug, Deserialize)]
struct GitHubReleaseAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Clone, Debug, Serialize)]
struct UpdateInfo {
    current_version: String,
    latest_version: String,
    update_available: bool,
    package_name: Option<String>,
    download_url: String,
    release_url: String,
}

fn version_numbers(version: &str) -> Option<Vec<u64>> {
    let version = version.trim().trim_start_matches(['v', 'V']);
    let core = version.split(['-', '+']).next()?;
    let numbers = core
        .split('.')
        .map(str::parse::<u64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (numbers.len() >= 2 && numbers.len() <= 4).then_some(numbers)
}

fn newer_version(latest: &str, current: &str) -> Result<bool, String> {
    let mut latest = version_numbers(latest).ok_or("release has an invalid version tag")?;
    let mut current = version_numbers(current).ok_or("application has an invalid version")?;
    let width = latest.len().max(current.len());
    latest.resize(width, 0);
    current.resize(width, 0);
    Ok(latest > current)
}

fn preferred_release_asset<'a>(assets: &'a [GitHubReleaseAsset]) -> Option<&'a GitHubReleaseAsset> {
    let suffixes: &[&str] = if cfg!(windows) {
        &["Setup-x64.exe", "windows-msvc.zip"]
    } else if std::env::var_os("APPIMAGE").is_some() {
        &["x86_64.AppImage", "amd64.deb", "linux-gnu.tar.gz"]
    } else {
        &["amd64.deb", "x86_64.AppImage", "linux-gnu.tar.gz"]
    };
    suffixes
        .iter()
        .find_map(|suffix| assets.iter().find(|asset| asset.name.ends_with(suffix)))
}

fn validate_release_url(candidate: &str) -> Result<(), String> {
    let url = url::Url::parse(candidate).map_err(|_| "release URL is invalid")?;
    if url.scheme() != "https"
        || !matches!(
            url.host_str(),
            Some("github.com" | "objects.githubusercontent.com")
        )
    {
        return Err("release URL is not hosted by GitHub over HTTPS".into());
    }
    Ok(())
}

#[tauri::command]
async fn check_for_updates() -> Result<UpdateInfo, String> {
    let releases = reqwest::Client::new()
        .get(RELEASES_API)
        .header(reqwest::header::USER_AGENT, "SayTheRest desktop updater")
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .map_err(|error| format!("cannot check GitHub releases: {error}"))?
        .error_for_status()
        .map_err(|error| format!("GitHub release check failed: {error}"))?
        .json::<Vec<GitHubRelease>>()
        .await
        .map_err(|error| format!("invalid GitHub release response: {error}"))?;
    let current = env!("CARGO_PKG_VERSION");
    let Some(release) = releases.into_iter().next() else {
        return Ok(UpdateInfo {
            current_version: current.into(),
            latest_version: current.into(),
            update_available: false,
            package_name: None,
            download_url: RELEASES_PAGE.into(),
            release_url: RELEASES_PAGE.into(),
        });
    };
    validate_release_url(&release.html_url)?;
    let asset = preferred_release_asset(&release.assets);
    let download_url = asset
        .map(|asset| asset.browser_download_url.as_str())
        .unwrap_or(&release.html_url);
    validate_release_url(download_url)?;
    Ok(UpdateInfo {
        current_version: current.into(),
        latest_version: release.tag_name.clone(),
        update_available: newer_version(&release.tag_name, current)?,
        package_name: asset.map(|asset| asset.name.clone()),
        download_url: download_url.into(),
        release_url: release.html_url,
    })
}

#[tauri::command]
fn open_release_download(url: String) -> Result<(), String> {
    validate_release_url(&url)?;
    #[cfg(target_os = "linux")]
    let mut command = std::process::Command::new("xdg-open");
    #[cfg(target_os = "linux")]
    command.arg(&url);
    #[cfg(windows)]
    let mut command = {
        let mut command = std::process::Command::new("rundll32.exe");
        command.args(["url.dll,FileProtocolHandler", &url]);
        command
    };
    command
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("cannot open the release download: {error}"))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DesktopSettings {
    selection_shortcut: String,
    clipboard_shortcut: String,
    #[serde(default = "default_true")]
    launch_at_login: bool,
}

fn default_true() -> bool {
    true
}

impl Default for DesktopSettings {
    fn default() -> Self {
        Self {
            selection_shortcut: "ctrl+alt+s".into(),
            clipboard_shortcut: "ctrl+alt+v".into(),
            launch_at_login: true,
        }
    }
}

struct DesktopState(RwLock<DesktopSettings>);

fn desktop_settings_path() -> std::path::PathBuf {
    default_data_dir().join("desktop-settings.json")
}

fn load_desktop_settings() -> DesktopSettings {
    std::fs::read(desktop_settings_path())
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn parsed_shortcuts(settings: &DesktopSettings) -> Result<(Shortcut, Shortcut), String> {
    let selection = Shortcut::from_str(&settings.selection_shortcut)
        .map_err(|error| format!("invalid selection shortcut: {error}"))?;
    let clipboard = Shortcut::from_str(&settings.clipboard_shortcut)
        .map_err(|error| format!("invalid clipboard shortcut: {error}"))?;
    if selection == clipboard {
        return Err("selection and clipboard shortcuts must be different".into());
    }
    Ok((selection, clipboard))
}

fn register_shortcuts(app: &tauri::AppHandle, settings: &DesktopSettings) -> Result<(), String> {
    let (selection, clipboard) = parsed_shortcuts(settings)?;
    app.global_shortcut()
        .unregister_all()
        .map_err(|error| error.to_string())?;
    if let Err(error) = app
        .global_shortcut()
        .register_multiple([selection, clipboard])
    {
        return Err(format!(
            "shortcut is unavailable or conflicts with another application: {error}"
        ));
    }
    Ok(())
}

#[tauri::command]
fn desktop_settings(state: tauri::State<'_, DesktopState>) -> Result<DesktopSettings, String> {
    state
        .0
        .read()
        .map(|value| value.clone())
        .map_err(|_| "desktop settings lock is unavailable".into())
}

#[tauri::command]
fn update_desktop_settings(
    app: tauri::AppHandle,
    state: tauri::State<'_, DesktopState>,
    settings: DesktopSettings,
) -> Result<DesktopSettings, String> {
    let previous = state
        .0
        .read()
        .map_err(|_| "desktop settings lock is unavailable")?
        .clone();
    if let Err(error) = register_shortcuts(&app, &settings) {
        let _ = register_shortcuts(&app, &previous);
        return Err(error);
    }
    if settings.launch_at_login != previous.launch_at_login
        && let Err(error) = configure_launch_at_login(settings.launch_at_login)
    {
        let _ = register_shortcuts(&app, &previous);
        return Err(error);
    }
    let path = desktop_settings_path();
    std::fs::create_dir_all(path.parent().ok_or("invalid desktop settings path")?)
        .map_err(|error| format!("cannot create settings directory: {error}"))?;
    let bytes = serde_json::to_vec_pretty(&settings).map_err(|error| error.to_string())?;
    if let Err(error) = std::fs::write(&path, bytes) {
        let _ = register_shortcuts(&app, &previous);
        return Err(format!("cannot save desktop settings: {error}"));
    }
    *state
        .0
        .write()
        .map_err(|_| "desktop settings lock is unavailable")? = settings.clone();
    Ok(settings)
}

#[cfg(target_os = "linux")]
fn configure_launch_at_login(enabled: bool) -> Result<(), String> {
    let systemctl_action = if enabled { "enable" } else { "disable" };
    let status = std::process::Command::new("systemctl")
        .args([
            "--user",
            systemctl_action,
            "say-the-rest.service",
            "say-the-rest-desktop.service",
        ])
        .status()
        .map_err(|error| format!("cannot run systemctl: {error}"))?;
    if !status.success() {
        return Err(format!(
            "cannot {systemctl_action} the per-user Say the Rest service"
        ));
    }
    let autostart = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(std::env::temp_dir)
                .join(".config")
        })
        .join("autostart/say-the-rest.desktop");
    std::fs::create_dir_all(autostart.parent().ok_or("invalid autostart path")?)
        .map_err(|error| format!("cannot create the autostart directory: {error}"))?;
    let contents = linux_autostart_entry(enabled)?;
    std::fs::write(&autostart, contents)
        .map_err(|error| format!("cannot update desktop autostart: {error}"))
}

#[cfg(target_os = "linux")]
fn linux_autostart_entry(enabled: bool) -> Result<String, String> {
    if !enabled {
        return Ok("[Desktop Entry]\nType=Application\nName=Say the Rest\nHidden=true\n".into());
    }
    let executable = std::env::var_os("APPIMAGE")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::current_exe().ok())
        .ok_or("cannot determine the desktop executable")?;
    let executable = executable
        .to_str()
        .ok_or("desktop executable path is not valid UTF-8")?;
    let quoted = executable.replace('\\', "\\\\").replace('"', "\\\"");
    Ok(format!(
        "[Desktop Entry]\nType=Application\nName=Say the Rest\nComment=Global offline text-to-speech shortcuts\nExec=\"{quoted}\"\nTerminal=false\nX-GNOME-Autostart-enabled=true\n"
    ))
}

#[cfg(windows)]
fn configure_launch_at_login(enabled: bool) -> Result<(), String> {
    let action = if enabled { "/ENABLE" } else { "/DISABLE" };
    for task in ["Say the Rest", "Say the Rest Desktop"] {
        let status = std::process::Command::new("schtasks.exe")
            .args(["/Change", "/TN", task, action])
            .status()
            .map_err(|error| format!("cannot update Windows startup task {task}: {error}"))?;
        if !status.success() {
            return Err(format!("cannot update Windows startup task {task}"));
        }
    }
    Ok(())
}

fn api_token() -> Result<String, String> {
    if let Ok(token) = std::env::var("SAY_THE_REST_TOKEN") {
        return Ok(token);
    }
    std::fs::read_to_string(default_data_dir().join("api-token"))
        .map(|token| token.trim().to_owned())
        .map_err(|error| format!("cannot read the local service token: {error}"))
}

fn default_data_dir() -> std::path::PathBuf {
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("SayTheRest")
    } else {
        std::env::var_os("XDG_STATE_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir)
                    .join(".local/state")
            })
            .join("say-the-rest")
    }
}

#[tauri::command]
fn export_history(source: String, destination: String) -> Result<(), String> {
    let source = std::path::PathBuf::from(source)
        .canonicalize()
        .map_err(|error| format!("archived audio is unavailable: {error}"))?;
    let history = default_data_dir()
        .join("history")
        .canonicalize()
        .map_err(|error| format!("history directory is unavailable: {error}"))?;
    if !source.starts_with(history)
        || source.extension().and_then(|extension| extension.to_str()) != Some("wav")
    {
        return Err("only archived history WAV files can be exported".into());
    }
    let destination = std::path::PathBuf::from(destination);
    if destination.as_os_str().is_empty() || destination.is_dir() {
        return Err("choose a destination WAV filename".into());
    }
    std::fs::copy(source, destination)
        .map(|_| ())
        .map_err(|error| format!("cannot export audio: {error}"))
}

#[tauri::command]
fn start_voice_recording(
    state: tauri::State<'_, recorder::RecorderState>,
) -> Result<recorder::RecordingStatus, String> {
    recorder::start(&state)
}

#[tauri::command]
fn voice_recording_status(
    state: tauri::State<'_, recorder::RecorderState>,
) -> Result<recorder::RecordingStatus, String> {
    recorder::status(&state)
}

#[tauri::command]
fn stop_voice_recording(
    state: tauri::State<'_, recorder::RecorderState>,
) -> Result<recorder::RecordingResult, String> {
    recorder::stop(&state, &default_data_dir().join("recordings"))
}

#[tauri::command]
fn cancel_voice_recording(state: tauri::State<'_, recorder::RecorderState>) -> Result<(), String> {
    recorder::cancel(&state)
}

#[tauri::command]
fn discard_voice_recording(path: String) -> Result<(), String> {
    let recordings = default_data_dir().join("recordings");
    let candidate = std::path::PathBuf::from(path);
    if candidate.parent() != Some(recordings.as_path())
        || candidate.extension().and_then(|value| value.to_str()) != Some("wav")
    {
        return Err("only temporary Say the Rest recordings can be discarded".into());
    }
    match std::fs::remove_file(candidate) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("cannot discard temporary recording: {error}")),
    }
}

#[tauri::command]
fn desktop_diagnostics() -> Value {
    let session = if cfg!(windows) {
        "windows"
    } else if std::env::var_os("WAYLAND_DISPLAY").is_some() && std::env::var_os("DISPLAY").is_none()
    {
        "wayland-native"
    } else if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        "wayland-with-xwayland"
    } else {
        "x11"
    };
    let selection_support = match session {
        "wayland-native" => {
            "AT-SPI selection; explicit clipboard access uses compositor data-control or a user-approved desktop portal"
        }
        "windows" => "UI Automation selection with copy-based fallback available",
        _ => "AT-SPI selection with copy-based fallback available",
    };
    let startup_requested = load_desktop_settings().launch_at_login;
    #[cfg(target_os = "linux")]
    let startup_active = startup_requested
        && std::process::Command::new("systemctl")
            .args(["--user", "is-enabled", "--quiet", "say-the-rest.service"])
            .status()
            .is_ok_and(|status| status.success());
    #[cfg(not(target_os = "linux"))]
    let startup_active = startup_requested;
    serde_json::json!({
        "platform": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "session": session,
        "selection_support": selection_support,
        "launch_at_login_requested": startup_requested,
        "launch_at_login_active": startup_active,
        "token_available": api_token().is_ok()
    })
}

fn authorized(request: reqwest::RequestBuilder) -> Result<reqwest::RequestBuilder, String> {
    Ok(request.bearer_auth(api_token()?))
}

#[tauri::command]
async fn service_get(resource: String) -> Result<Value, String> {
    let resource = safe_resource(&resource)?;
    authorized(reqwest::Client::new().get(format!("{SERVICE}/{resource}")))?
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?
        .json()
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn service_post(resource: String, body: Option<Value>) -> Result<Option<Value>, String> {
    let resource = safe_resource(&resource)?;
    let request = authorized(reqwest::Client::new().post(format!("{SERVICE}/{resource}")))?;
    let response = match body {
        Some(body) => request.json(&body),
        None => request,
    }
    .send()
    .await
    .map_err(|error| error.to_string())?
    .error_for_status()
    .map_err(|error| error.to_string())?;
    let bytes = response.bytes().await.map_err(|error| error.to_string())?;
    if bytes.is_empty() {
        return Ok(None);
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn service_delete(resource: String) -> Result<(), String> {
    let resource = safe_resource(&resource)?;
    authorized(reqwest::Client::new().delete(format!("{SERVICE}/{resource}")))?
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn safe_resource(resource: &str) -> Result<&str, String> {
    if resource.is_empty()
        || resource.starts_with('/')
        || resource.contains("..")
        || resource.contains('?')
    {
        return Err("invalid service resource".into());
    }
    Ok(resource)
}

fn tray_image() -> Image<'static> {
    let size = 32usize;
    let mut rgba = vec![0u8; size * size * 4];
    for y in 4..28 {
        let half = 3 + ((y as isize - 16).unsigned_abs() / 3);
        for x in (16 - half)..=(16 + half) {
            let offset = (y * size + x) * 4;
            rgba[offset..offset + 4].copy_from_slice(&[238, 83, 47, 255]);
        }
    }
    Image::new_owned(rgba, size as u32, size as u32)
}

fn handle_capture(app: tauri::AppHandle, kind: CaptureKind) {
    tauri::async_runtime::spawn(async move {
        let result = tauri::async_runtime::spawn_blocking(move || selection::capture(kind)).await;
        let mut result = match result {
            Ok(result) => result,
            Err(error) => Err(format!("selection worker failed: {error}")),
        };
        #[cfg(target_os = "linux")]
        if kind == CaptureKind::Clipboard
            && result.is_err()
            && selection::portal_clipboard_required()
        {
            result = selection::capture_portal_clipboard().await;
        }
        let status = match result {
            Ok(text) => {
                let client = reqwest::Client::new();
                let threshold = match authorized(client.get(format!("{SERVICE}/settings"))) {
                    Ok(request) => match request.send().await {
                        Ok(response) => response
                            .json::<Value>()
                            .await
                            .ok()
                            .and_then(|settings| {
                                settings
                                    .get("long_text_confirmation_characters")
                                    .and_then(Value::as_u64)
                            })
                            .unwrap_or(20_000),
                        Err(_) => 20_000,
                    },
                    Err(_) => 20_000,
                };
                let character_count = text.chars().count();
                if character_count as u64 > threshold {
                    let _ = app.emit(
                        "long-text-request",
                        serde_json::json!({
                            "text": text,
                            "kind": kind.source(),
                            "characters": character_count,
                            "threshold": threshold
                        }),
                    );
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                    return;
                }
                let response = match authorized(client.post(format!("{SERVICE}/jobs"))) {
                    Ok(request) => request,
                    Err(error) => {
                        return app
                            .emit(
                                "capture-status",
                                serde_json::json!({
                                    "ok": false,
                                    "kind": kind.source(),
                                    "error": error
                                }),
                            )
                            .map_err(|_| ())
                            .unwrap_or(());
                    }
                }
                .json(&serde_json::json!({
                    "text": text,
                    "source": kind.source(),
                    "queue_policy": "replace",
                    "confirmed_long_text": false
                }))
                .send()
                .await;
                match response {
                    Ok(response) if response.status().is_success() => {
                        serde_json::json!({"ok": true, "kind": kind.source()})
                    }
                    Ok(response) => serde_json::json!({
                        "ok": false,
                        "kind": kind.source(),
                        "error": format!("speech service returned {}", response.status())
                    }),
                    Err(error) => serde_json::json!({
                        "ok": false,
                        "kind": kind.source(),
                        "error": format!("speech service is unavailable: {error}")
                    }),
                }
            }
            Err(error) => serde_json::json!({
                "ok": false,
                "kind": kind.source(),
                "error": error
            }),
        };
        let failed = status.get("ok").and_then(Value::as_bool) == Some(false);
        let _ = app.emit("capture-status", status);
        if failed && let Some(window) = app.get_webview_window("main") {
            let _ = window.show();
            let _ = window.set_focus();
        }
    });
}

fn handle_tray_playback(app: tauri::AppHandle, toggle: bool) {
    tauri::async_runtime::spawn(async move {
        let result = async {
            let resource = if toggle {
                let state = authorized(reqwest::Client::new().get(format!("{SERVICE}/state")))?
                    .send()
                    .await
                    .map_err(|error| error.to_string())?
                    .error_for_status()
                    .map_err(|error| error.to_string())?
                    .json::<Value>()
                    .await
                    .map_err(|error| error.to_string())?;
                match state.pointer("/playback/state").and_then(Value::as_str) {
                    Some("paused") => "playback/resume",
                    Some("playing") => "playback/pause",
                    _ => return Err("nothing is currently playing".into()),
                }
            } else {
                "playback/stop"
            };
            authorized(reqwest::Client::new().post(format!("{SERVICE}/{resource}")))?
                .send()
                .await
                .map_err(|error| error.to_string())?
                .error_for_status()
                .map_err(|error| error.to_string())?;
            Ok::<_, String>(if toggle {
                "Playback updated"
            } else {
                "Playback stopped"
            })
        }
        .await;
        let failed = result.is_err();
        let status = match result {
            Ok(message) => serde_json::json!({"ok": true, "message": message}),
            Err(error) => serde_json::json!({"ok": false, "message": error}),
        };
        let _ = app.emit("desktop-action-status", status);
        if failed && let Some(window) = app.get_webview_window("main") {
            let _ = window.show();
            let _ = window.set_focus();
        }
    });
}

fn tray_status_text(state: &Value) -> (String, String, bool) {
    let playback = &state["playback"];
    let status = playback["state"].as_str();
    let position = playback["position_seconds"].as_f64().unwrap_or(0.0);
    let duration = playback["duration_seconds"].as_f64().unwrap_or(0.0);
    let queue = state["queue_depth"].as_u64().unwrap_or(0);
    let clock = |seconds: f64| {
        let seconds = seconds.max(0.0).floor() as u64;
        format!("{}:{:02}", seconds / 60, seconds % 60)
    };
    match status {
        Some("playing") => (
            format!("Speaking · {} / {}", clock(position), clock(duration)),
            "Pause".into(),
            true,
        ),
        Some("paused") => (
            format!("Paused · {} / {}", clock(position), clock(duration)),
            "Resume".into(),
            true,
        ),
        Some("synthesizing") => (
            if queue > 0 {
                format!("Generating locally · {queue} queued")
            } else {
                "Generating speech locally".into()
            },
            "Pause / resume".into(),
            false,
        ),
        _ if queue > 0 => (
            format!("Ready · {queue} queued"),
            "Pause / resume".into(),
            false,
        ),
        _ => (
            "Ready · local speech".into(),
            "Pause / resume".into(),
            false,
        ),
    }
}

fn start_tray_status_poll(
    status_item: MenuItem<tauri::Wry>,
    pause_item: MenuItem<tauri::Wry>,
    stop_item: MenuItem<tauri::Wry>,
) {
    tauri::async_runtime::spawn(async move {
        loop {
            let state = match authorized(reqwest::Client::new().get(format!("{SERVICE}/state"))) {
                Ok(request) => match request.send().await {
                    Ok(response) if response.status().is_success() => {
                        response.json::<Value>().await.ok()
                    }
                    _ => None,
                },
                Err(_) => None,
            };
            let (status, pause, controls_enabled) = state
                .as_ref()
                .map(tray_status_text)
                .unwrap_or_else(|| ("Service unavailable".into(), "Pause / resume".into(), false));
            let _ = status_item.set_text(status);
            let _ = pause_item.set_text(pause);
            let _ = pause_item.set_enabled(controls_enabled);
            let _ = stop_item.set_enabled(controls_enabled);
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    });
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .invoke_handler(tauri::generate_handler![
            service_get,
            service_post,
            service_delete,
            export_history,
            desktop_diagnostics,
            desktop_settings,
            update_desktop_settings,
            check_for_updates,
            open_release_download,
            start_voice_recording,
            voice_recording_status,
            stop_voice_recording,
            cancel_voice_recording,
            discard_voice_recording
        ])
        .manage(DesktopState(RwLock::new(load_desktop_settings())))
        .manage(recorder::RecorderState(std::sync::Mutex::new(None)))
        .setup(|app| {
            app.handle().plugin(
                tauri_plugin_global_shortcut::Builder::new()
                    .with_handler(|app, shortcut, event| {
                        if event.state != ShortcutState::Pressed {
                            return;
                        }
                        let settings = app.state::<DesktopState>();
                        let Ok(settings) = settings.0.read() else {
                            return;
                        };
                        let Ok((selection, clipboard)) = parsed_shortcuts(&settings) else {
                            return;
                        };
                        if shortcut == &selection {
                            handle_capture(app.clone(), CaptureKind::Selection);
                        } else if shortcut == &clipboard {
                            handle_capture(app.clone(), CaptureKind::Clipboard);
                        }
                    })
                    .build(),
            )?;
            let settings = app
                .state::<DesktopState>()
                .0
                .read()
                .map_err(|_| "desktop settings lock is unavailable")?
                .clone();
            register_shortcuts(app.handle(), &settings)?;
            let status = MenuItem::with_id(
                app,
                "status",
                "Connecting to local service…",
                false,
                None::<&str>,
            )?;
            let open = MenuItem::with_id(app, "open", "Open Say the Rest", true, None::<&str>)?;
            let read_selection =
                MenuItem::with_id(app, "read-selection", "Read selection", true, None::<&str>)?;
            let read_clipboard =
                MenuItem::with_id(app, "read-clipboard", "Read clipboard", true, None::<&str>)?;
            let pause_resume =
                MenuItem::with_id(app, "pause-resume", "Pause / resume", false, None::<&str>)?;
            let stop = MenuItem::with_id(app, "stop", "Stop", false, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(
                app,
                &[
                    &status,
                    &open,
                    &read_selection,
                    &read_clipboard,
                    &pause_resume,
                    &stop,
                    &quit,
                ],
            )?;
            TrayIconBuilder::new()
                .icon(tray_image())
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "read-selection" => handle_capture(app.clone(), CaptureKind::Selection),
                    "read-clipboard" => handle_capture(app.clone(), CaptureKind::Clipboard),
                    "pause-resume" => handle_tray_playback(app.clone(), true),
                    "stop" => handle_tray_playback(app.clone(), false),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;
            start_tray_status_poll(status, pause_resume, stop);
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run Say the Rest desktop application");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn service_proxy_rejects_path_traversal() {
        assert!(safe_resource("../secrets").is_err());
        assert!(safe_resource("models/piper/install").is_ok());
    }

    #[test]
    fn linux_packages_supervise_the_shortcut_host_separately() {
        let appimage_unit = include_str!(
            "../../../../packaging/linux-appimage/say-the-rest-appimage-desktop.service"
        );
        let installed_unit =
            include_str!("../../../../packaging/linux/say-the-rest-desktop.service");
        let app_run = include_str!("../../../../packaging/linux-appimage/AppRun");
        for unit in [appimage_unit, installed_unit] {
            assert!(unit.contains("Restart=on-failure"));
            assert!(unit.contains("Wants=say-the-rest.service"));
        }
        assert!(appimage_unit.contains("--desktop"));
        assert!(app_run.contains("restart say-the-rest-desktop.service"));
        assert!(app_run.contains("import-environment DISPLAY WAYLAND_DISPLAY"));
    }

    #[test]
    fn packaged_installers_are_offline_and_require_no_developer_toolchain() {
        let linux = include_str!("../../../../packaging/setup.sh");
        let windows = include_str!("../../../../packaging/setup.ps1");
        for setup in [linux, windows] {
            let lower = setup.to_ascii_lowercase();
            for forbidden in [
                "cargo",
                "python",
                "curl ",
                "wget ",
                "invoke-webrequest",
                "http://",
                "https://",
            ] {
                assert!(
                    !lower.contains(forbidden),
                    "installer contains forbidden dependency or network operation: {forbidden}"
                );
            }
            assert!(setup.contains("say-the-rest-service"));
            assert!(setup.contains("say-the-rest-desktop"));
        }
        assert!(windows.contains("-RestartCount 999"));
        assert!(windows.contains("-DontStopIfGoingOnBatteries"));
        let packaging = include_str!("../../../../scripts/package-release.sh");
        assert!(packaging.contains("SHERPA_SHA256="));
        assert!(packaging.contains("LINUXDEPLOY_SHA256="));
        assert!(packaging.contains("APPIMAGETOOL_SHA256="));
        assert!(packaging.contains("verify_sha256"));
    }

    #[test]
    fn legacy_desktop_settings_keep_launch_at_login_enabled() {
        let settings: DesktopSettings = serde_json::from_str(
            r#"{"selection_shortcut":"ctrl+alt+s","clipboard_shortcut":"ctrl+alt+v"}"#,
        )
        .unwrap();
        assert!(settings.launch_at_login);
    }

    #[test]
    fn tray_status_reflects_service_and_player_state() {
        let playing = serde_json::json!({
            "queue_depth": 2,
            "playback": {
                "state": "playing",
                "position_seconds": 65.9,
                "duration_seconds": 130.2
            }
        });
        assert_eq!(
            tray_status_text(&playing),
            ("Speaking · 1:05 / 2:10".into(), "Pause".into(), true)
        );

        let paused = serde_json::json!({
            "queue_depth": 0,
            "playback": {"state": "paused", "position_seconds": 2, "duration_seconds": 9}
        });
        assert_eq!(tray_status_text(&paused).1, "Resume");
        assert!(tray_status_text(&paused).2);

        let generating = serde_json::json!({
            "queue_depth": 3,
            "playback": {"state": "synthesizing"}
        });
        assert_eq!(
            tray_status_text(&generating).0,
            "Generating locally · 3 queued"
        );
        assert!(!tray_status_text(&generating).2);
    }

    #[test]
    fn updater_compares_versions_and_accepts_only_github_https_urls() {
        assert!(newer_version("v0.2.0", "0.1.9").unwrap());
        assert!(newer_version("1.0.0", "0.99.99").unwrap());
        assert!(!newer_version("v0.1.0", "0.1.0").unwrap());
        assert!(!newer_version("0.1.0", "0.2.0").unwrap());
        assert!(newer_version("latest", "0.1.0").is_err());
        assert!(validate_release_url("https://github.com/a1denvalu3/SayTheRest/releases").is_ok());
        assert!(validate_release_url("http://github.com/a1denvalu3/SayTheRest").is_err());
        assert!(validate_release_url("https://example.com/SayTheRest.exe").is_err());
    }

    #[test]
    fn updater_selects_a_native_release_package() {
        let assets = vec![
            GitHubReleaseAsset {
                name: "say-the-rest-x86_64-pc-windows-msvc.zip".into(),
                browser_download_url: "https://github.com/example/windows.zip".into(),
            },
            GitHubReleaseAsset {
                name: "SayTheRest-Setup-x64.exe".into(),
                browser_download_url: "https://github.com/example/setup.exe".into(),
            },
            GitHubReleaseAsset {
                name: "SayTheRest-0.2.0-x86_64.AppImage".into(),
                browser_download_url: "https://github.com/example/appimage".into(),
            },
            GitHubReleaseAsset {
                name: "say-the-rest_0.2.0_amd64.deb".into(),
                browser_download_url: "https://github.com/example/deb".into(),
            },
        ];
        let selected = preferred_release_asset(&assets).unwrap();
        if cfg!(windows) {
            assert!(selected.name.ends_with("Setup-x64.exe"));
        } else {
            assert!(selected.name.ends_with("amd64.deb"));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_autostart_entries_enable_or_mask_the_desktop_process() {
        let enabled = linux_autostart_entry(true).unwrap();
        assert!(enabled.contains("Exec=\""));
        assert!(enabled.contains("X-GNOME-Autostart-enabled=true"));
        let disabled = linux_autostart_entry(false).unwrap();
        assert!(disabled.contains("Hidden=true"));
        assert!(!disabled.contains("Exec="));
    }
}
