use std::{thread, time::Duration};

use clipboard_rs::{Clipboard, ClipboardContent, ClipboardContext};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

const COPY_SETTLE_TIME: Duration = Duration::from_millis(140);
const MAX_CLIPBOARD_SNAPSHOT_BYTES: usize = 256 * 1024 * 1024;
const MAX_CLIPBOARD_FORMATS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureKind {
    Selection,
    Clipboard,
}

impl CaptureKind {
    pub fn source(self) -> &'static str {
        match self {
            Self::Selection => "selection",
            Self::Clipboard => "clipboard",
        }
    }
}

pub fn capture(kind: CaptureKind) -> Result<String, String> {
    match kind {
        CaptureKind::Selection => capture_selection_with_copy(),
        CaptureKind::Clipboard => clipboard_text(),
    }
}

fn clipboard_text() -> Result<String, String> {
    #[cfg(target_os = "linux")]
    if wayland_data_control_unavailable() {
        if std::env::var_os("DISPLAY").is_some()
            && let Ok(mut clipboard) = arboard::Clipboard::new()
            && let Ok(text) = clipboard.get_text()
        {
            return cleaned(text);
        }
        return Err("the compositor blocks background clipboard access; approve the desktop portal when prompted".into());
    }
    let text = ClipboardContext::new()
        .and_then(|clipboard| clipboard.get_text())
        .map_err(|error| format!("clipboard is unavailable: {error}"))?;
    cleaned(text)
}

struct ClipboardSnapshot(Vec<ClipboardContent>);

fn snapshot_clipboard() -> Result<ClipboardSnapshot, String> {
    #[cfg(target_os = "linux")]
    if wayland_data_control_unavailable() {
        return Err("the focused application did not expose selected text through AT-SPI, and this Wayland compositor does not permit a lossless copy-and-restore fallback".into());
    }
    let clipboard = ClipboardContext::new().map_err(|error| error.to_string())?;
    let formats = clipboard
        .available_formats()
        .map_err(|error| format!("cannot enumerate clipboard formats: {error}"))?;
    if formats.len() > MAX_CLIPBOARD_FORMATS {
        return Err(format!(
            "clipboard exposes {} formats; refusing copy fallback because it cannot be preserved safely",
            formats.len()
        ));
    }
    let mut total = 0usize;
    let mut contents = Vec::with_capacity(formats.len());
    for format in formats {
        let bytes = clipboard
            .get_buffer(&format)
            .map_err(|error| format!("cannot preserve clipboard format {format:?}: {error}"))?;
        total = total
            .checked_add(bytes.len())
            .ok_or("clipboard snapshot size overflow")?;
        if total > MAX_CLIPBOARD_SNAPSHOT_BYTES {
            return Err("clipboard exceeds the 256 MB safe preservation limit; use the clipboard shortcut instead".into());
        }
        contents.push(ClipboardContent::Other(format, bytes));
    }
    Ok(ClipboardSnapshot(contents))
}

#[cfg(target_os = "linux")]
pub fn portal_clipboard_required() -> bool {
    wayland_data_control_unavailable()
}

#[cfg(not(target_os = "linux"))]
pub fn portal_clipboard_required() -> bool {
    false
}

#[cfg(target_os = "linux")]
fn wayland_data_control_unavailable() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some()
        && wl_clipboard_rs::utils::is_primary_selection_supported().is_err()
}

#[cfg(target_os = "linux")]
pub async fn capture_portal_clipboard() -> Result<String, String> {
    use ashpd::desktop::{
        Session,
        clipboard::{Clipboard as PortalClipboardProxy, RequestClipboardOptions},
        remote_desktop::{RemoteDesktop, SelectDevicesOptions},
    };
    use std::{io::Read, sync::OnceLock};
    use tokio::sync::Mutex;

    struct PortalSession {
        clipboard: PortalClipboardProxy,
        session: Session<RemoteDesktop>,
    }

    static PORTAL: OnceLock<Mutex<Option<PortalSession>>> = OnceLock::new();
    let portal = PORTAL.get_or_init(|| Mutex::new(None));
    let mut guard = portal.lock().await;
    if guard.is_none() {
        let remote = RemoteDesktop::new()
            .await
            .map_err(|error| format!("desktop portal is unavailable: {error}"))?;
        if remote.version() < 2 {
            return Err("the desktop portal is too old for clipboard access (RemoteDesktop version 2 is required)".into());
        }
        let clipboard = PortalClipboardProxy::new()
            .await
            .map_err(|error| format!("clipboard portal is unavailable: {error}"))?;
        let session = remote
            .create_session(Default::default())
            .await
            .map_err(|error| format!("cannot create a clipboard portal session: {error}"))?;
        clipboard
            .request(&session, RequestClipboardOptions::default())
            .await
            .map_err(|error| format!("cannot request portal clipboard access: {error}"))?;
        remote
            .select_devices(&session, SelectDevicesOptions::default())
            .await
            .and_then(|request| request.response())
            .map_err(|error| format!("clipboard portal request was rejected: {error}"))?;
        let response = remote
            .start(&session, None, Default::default())
            .await
            .and_then(|request| request.response())
            .map_err(|error| format!("clipboard portal session was not approved: {error}"))?;
        if !response.is_clipboard_enabled() {
            return Err("clipboard access was not enabled in the desktop portal prompt".into());
        }
        *guard = Some(PortalSession { clipboard, session });
    }
    let portal = guard.as_ref().expect("portal session initialized");
    let mut last_error = None;
    for mime_type in ["text/plain;charset=utf-8", "text/plain", "UTF8_STRING"] {
        match portal
            .clipboard
            .selection_read(&portal.session, mime_type)
            .await
        {
            Ok(fd) => {
                let owned: std::os::fd::OwnedFd = fd.into();
                let mut file = std::fs::File::from(owned);
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)
                    .map_err(|error| format!("cannot read portal clipboard data: {error}"))?;
                let text = String::from_utf8(bytes)
                    .map_err(|_| "portal clipboard text is not valid UTF-8".to_string())?;
                return cleaned(text);
            }
            Err(error) => last_error = Some(error.to_string()),
        }
    }
    Err(format!(
        "the portal clipboard has no readable plain-text format{}",
        last_error
            .map(|error| format!(": {error}"))
            .unwrap_or_default()
    ))
}

fn restore_clipboard(snapshot: ClipboardSnapshot) -> Result<(), String> {
    let clipboard = ClipboardContext::new().map_err(|error| error.to_string())?;
    if snapshot.0.is_empty() {
        clipboard.clear().map_err(|error| error.to_string())
    } else {
        clipboard
            .set(snapshot.0)
            .map_err(|error| format!("cannot restore clipboard formats: {error}"))
    }
}

fn capture_selection_with_copy() -> Result<String, String> {
    #[cfg(windows)]
    if let Ok(selection) = capture_windows_accessible_selection() {
        return cleaned(selection);
    }

    #[cfg(target_os = "linux")]
    if let Ok(selection) = capture_linux_accessible_selection() {
        return cleaned(selection);
    }

    capture_selection_via_copy(|| {
        let mut keyboard = Enigo::new(&Settings::default())
            .map_err(|error| format!("cannot access global keyboard input: {error}"))?;
        keyboard
            .key(Key::Control, Direction::Press)
            .map_err(|error| format!("cannot press the copy modifier: {error}"))?;
        let copy_result = keyboard
            .key(Key::Unicode('c'), Direction::Click)
            .map_err(|error| format!("cannot request the selected text: {error}"));
        let release_result = keyboard
            .key(Key::Control, Direction::Release)
            .map_err(|error| format!("cannot release the copy modifier: {error}"));
        copy_result.and(release_result)
    })
}

fn capture_selection_via_copy(copy: impl FnOnce() -> Result<(), String>) -> Result<String, String> {
    let previous = snapshot_clipboard()?;
    let copy_result = copy();
    thread::sleep(COPY_SETTLE_TIME);

    let selected = if let Err(error) = copy_result {
        Err(error)
    } else {
        ClipboardContext::new()
            .map_err(|error| error.to_string())
            .and_then(|clipboard| {
                clipboard.get_text().map_err(|error| {
                    format!("the focused application did not expose selected text: {error}")
                })
            })
    };
    let restored = restore_clipboard(previous);
    match (selected, restored) {
        (_, Err(error)) => Err(format!(
            "selection capture was attempted but the clipboard could not be restored: {error}"
        )),
        (Err(error), Ok(())) => Err(error),
        (Ok(text), Ok(())) => cleaned(text),
    }
}

#[cfg(target_os = "linux")]
fn capture_linux_accessible_selection() -> Result<String, String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?
        .block_on(capture_linux_accessible_selection_async())
}

#[cfg(target_os = "linux")]
async fn capture_linux_accessible_selection_async() -> Result<String, String> {
    use atspi::{
        AccessibilityConnection, State,
        proxy::{accessible::ObjectRefExt, text::TextProxy},
    };
    use std::collections::VecDeque;

    let accessibility = AccessibilityConnection::new()
        .await
        .map_err(|error| format!("AT-SPI is unavailable: {error}"))?;
    let root = accessibility
        .root_accessible_on_registry()
        .await
        .map_err(|error| error.to_string())?;
    let mut pending = VecDeque::from(
        root.get_children()
            .await
            .map_err(|error| error.to_string())?,
    );
    let mut visited = 0usize;
    while let Some(reference) = pending.pop_front() {
        visited += 1;
        if visited > 4_000 {
            break;
        }
        let accessible = match reference
            .as_accessible_proxy(accessibility.connection())
            .await
        {
            Ok(accessible) => accessible,
            Err(_) => continue,
        };
        let focused = accessible
            .get_state()
            .await
            .is_ok_and(|states| states.contains(State::Focused));
        if focused {
            let text = TextProxy::builder(accessibility.connection())
                .destination(accessible.inner().destination().clone())
                .map_err(|error| error.to_string())?
                .path(accessible.inner().path().clone())
                .map_err(|error| error.to_string())?
                .build()
                .await
                .map_err(|error| error.to_string())?;
            let selections = text
                .get_n_selections()
                .await
                .map_err(|error| error.to_string())?;
            let mut selected = Vec::new();
            for index in 0..selections {
                let (start, end) = text
                    .get_selection(index)
                    .await
                    .map_err(|error| error.to_string())?;
                if end > start {
                    selected.push(
                        text.get_text(start, end)
                            .await
                            .map_err(|error| error.to_string())?,
                    );
                }
            }
            if !selected.is_empty() {
                return Ok(selected.join("\n"));
            }
        }
        if let Ok(children) = accessible.get_children().await {
            pending.extend(children);
        }
    }
    Err("the focused Linux application did not expose selected text through AT-SPI".into())
}

#[cfg(windows)]
fn capture_windows_accessible_selection() -> Result<String, String> {
    use uiautomation::{UIAutomation, patterns::UITextPattern};

    let automation = UIAutomation::new().map_err(|error| error.to_string())?;
    let walker = automation
        .get_control_view_walker()
        .map_err(|error| error.to_string())?;
    let mut element = automation
        .get_focused_element()
        .map_err(|error| error.to_string())?;
    for _ in 0..8 {
        if let Ok(pattern) = element.get_pattern::<UITextPattern>() {
            let text = pattern
                .get_selection()
                .map_err(|error| error.to_string())?
                .into_iter()
                .filter_map(|range| range.get_text(-1).ok())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if !text.trim().is_empty() {
                return Ok(text);
            }
        }
        element = walker
            .get_parent(&element)
            .map_err(|error| error.to_string())?;
    }
    Err("the focused Windows control did not expose a UI Automation text selection".into())
}

fn cleaned(text: String) -> Result<String, String> {
    let text = text.trim().to_owned();
    if text.is_empty() {
        Err("no text was available".into())
    } else {
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_blank_capture() {
        assert!(cleaned(" \n\t".into()).is_err());
    }

    #[test]
    fn reports_protocol_source() {
        assert_eq!(CaptureKind::Selection.source(), "selection");
        assert_eq!(CaptureKind::Clipboard.source(), "clipboard");
    }

    #[test]
    #[ignore = "requires a desktop clipboard; run under Xvfb or a Windows desktop"]
    fn rich_clipboard_snapshot_restores_multiple_formats() {
        let clipboard = ClipboardContext::new().unwrap();
        clipboard
            .set(vec![
                ClipboardContent::Text("original plain text".into()),
                ClipboardContent::Html("<b>original rich text</b>".into()),
                ClipboardContent::Other(
                    "application/x-say-the-rest-test".into(),
                    vec![0, 1, 2, 255],
                ),
            ])
            .unwrap();
        let selected = capture_selection_via_copy(|| {
            ClipboardContext::new()
                .map_err(|error| error.to_string())?
                .set_text("temporary selection".into())
                .map_err(|error| error.to_string())
        })
        .unwrap();
        assert_eq!(selected, "temporary selection");
        let restored = ClipboardContext::new().unwrap();
        assert_eq!(restored.get_text().unwrap(), "original plain text");
        assert_eq!(restored.get_html().unwrap(), "<b>original rich text</b>");
        assert_eq!(
            restored
                .get_buffer("application/x-say-the-rest-test")
                .unwrap(),
            vec![0, 1, 2, 255]
        );
    }
}
