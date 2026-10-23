use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{Emitter, Manager};

#[derive(Default)]
pub struct PresentationState(pub AtomicBool);

pub fn show_management(app: &tauri::AppHandle, section: &str) -> Result<(), String> {
    if !["player", "models", "voices", "history", "settings"].contains(&section) {
        return Err("unknown application section".into());
    }
    let window = app
        .get_webview_window("main")
        .ok_or("management window is unavailable")?;
    hide(app, "player")?;
    window.unminimize().map_err(|e| e.to_string())?;
    window.show().map_err(|e| e.to_string())?;
    let _ = window.set_focus();
    let _ = app.emit(
        "window-visibility",
        serde_json::json!({"label":"main", "visible":true}),
    );
    window
        .emit("show-section", section)
        .map_err(|e| e.to_string())
}

pub fn hide(app: &tauri::AppHandle, label: &str) -> Result<(), String> {
    if let Some(window) = app.get_webview_window(label) {
        window.hide().map_err(|e| e.to_string())?;
        let _ = app.emit(
            "window-visibility",
            serde_json::json!({"label":label, "visible":false}),
        );
    }
    Ok(())
}

/// Physical coordinates, including negative monitor origins and high-DPI sizes.
fn popup_position(
    anchor: (i32, i32, i32, i32),
    work: (i32, i32, i32, i32),
    size: (i32, i32),
) -> (i32, i32) {
    let (ax, ay, aw, ah) = anchor;
    let (wx, wy, ww, wh) = work;
    let x = (ax + aw / 2 - size.0 / 2).clamp(wx, (wx + ww - size.0).max(wx));
    let y = if ay + ah / 2 < wy + wh / 2 {
        ay + ah + 8
    } else {
        ay - size.1 - 8
    };
    (x, y.clamp(wy, (wy + wh - size.1).max(wy)))
}

pub fn show_player(app: &tauri::AppHandle, anchor: Option<tauri::Rect>) -> Result<(), String> {
    let window = app
        .get_webview_window("player")
        .ok_or("player window is unavailable")?;
    app.state::<PresentationState>()
        .0
        .store(false, Ordering::Relaxed);
    // AppIndicator does not supply tray click coordinates on Linux. The menu
    // entry remains available; use the cursor if positioning is permitted.
    let location = anchor
        .map(|rect| {
            let p = rect
                .position
                .to_physical::<i32>(window.scale_factor().unwrap_or(1.0));
            let s = rect
                .size
                .to_physical::<u32>(window.scale_factor().unwrap_or(1.0));
            (p.x, p.y, s.width as i32, s.height as i32)
        })
        .or_else(|| {
            window
                .cursor_position()
                .ok()
                .map(|p| (p.x as i32, p.y as i32, 1, 1))
        });
    if let Some(anchor) = location
        && let Ok(Some(monitor)) = window.monitor_from_point(anchor.0 as f64, anchor.1 as f64)
        && let Ok(size) = window.outer_size()
    {
        let work = monitor.work_area();
        let (x, y) = popup_position(
            anchor,
            (
                work.position.x,
                work.position.y,
                work.size.width as i32,
                work.size.height as i32,
            ),
            (size.width as i32, size.height as i32),
        );
        let _ = window.set_position(tauri::PhysicalPosition::new(x, y));
    }
    window.show().map_err(|e| e.to_string())?;
    let _ = window.set_focus();
    app.emit(
        "window-visibility",
        serde_json::json!({"label":"player", "visible":true}),
    )
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn open_management(app: tauri::AppHandle, section: String) -> Result<(), String> {
    show_management(&app, &section)
}
#[tauri::command]
pub fn open_player(app: tauri::AppHandle) -> Result<(), String> {
    show_player(&app, None)
}
#[tauri::command]
pub fn close_player(app: tauri::AppHandle) -> Result<(), String> {
    hide(&app, "player")
}
#[tauri::command]
pub fn window_visible(window: tauri::WebviewWindow) -> Result<bool, String> {
    window.is_visible().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn desktop_context(window: tauri::WebviewWindow) -> Result<serde_json::Value, String> {
    Ok(
        serde_json::json!({"visible":window.is_visible().map_err(|e|e.to_string())?, "testing":crate::desktop_test_mode(),
            "snapshots":crate::desktop_test_mode() && cfg!(target_os="linux") && std::env::var_os("SAYIT_UI_TEST_SNAPSHOTS").is_some()}),
    )
}
#[tauri::command]
pub fn read_clipboard(app: tauri::AppHandle) {
    crate::handle_capture(app, crate::selection::CaptureKind::Clipboard);
}
#[tauri::command]
pub fn quit_desktop(app: tauri::AppHandle) {
    app.exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn popup_stays_on_screen_at_taskbar_edges_and_negative_dpi_origins() {
        assert_eq!(
            popup_position((1880, 1040, 30, 30), (0, 0, 1920, 1040), (360, 460)),
            (1560, 572)
        );
        assert_eq!(
            popup_position((0, 0, 24, 24), (0, 24, 1920, 1056), (360, 460)),
            (0, 32)
        );
        let p = popup_position((-1920, 20, 40, 40), (-1920, 0, 1920, 1080), (720, 920));
        assert_eq!(p, (-1920, 68));
        assert_eq!(
            popup_position((3, 3, 1, 1), (0, 0, 200, 200), (360, 460)),
            (0, 0)
        );
    }
}
