//! Debug-only native WebView acceptance driver. Release builds have no script
//! injection or result listener. The runner supplies an isolated service and
//! XDG data directories, and disables tray/global-shortcut registration.
use tauri::Listener;
#[cfg(target_os = "linux")]
use tauri::{Emitter, Manager};

pub fn install(app: &tauri::AppHandle) {
    if !crate::desktop_test_mode() {
        return;
    }
    let Some(destination) = std::env::var_os("SAYIT_UI_TEST_RESULT") else {
        return;
    };
    let destination = std::path::PathBuf::from(destination);
    let handle = app.clone();
    app.listen("ui-test-result", move |event| {
        let result = serde_json::from_str::<serde_json::Value>(event.payload());
        let passed = result.as_ref().is_ok_and(|value| value["passed"] == true);
        if let Err(error) = std::fs::write(&destination, event.payload()) {
            eprintln!("cannot write UI test result: {error}");
        }
        handle.exit(if passed { 0 } else { 1 });
    });
    #[cfg(target_os = "linux")]
    if let Some(directory) = std::env::var_os("SAYIT_UI_TEST_SNAPSHOTS") {
        let directory = std::path::PathBuf::from(directory);
        let handle = app.clone();
        app.listen("ui-test-snapshot", move |event| {
            let Ok(theme) = serde_json::from_str::<String>(event.payload()) else { return; };
            if !["light","dark","idle"].contains(&theme.as_str()) { return; }
            let Some(window) = handle.get_webview_window("player") else { return; };
            let destination = directory.join(format!("player-{theme}.png"));
            let sender = handle.clone();
            let _ = window.with_webview(move |platform| {
                use webkit2gtk::WebViewExt;
                platform.inner().snapshot(webkit2gtk::SnapshotRegion::Visible, webkit2gtk::SnapshotOptions::NONE,
                    None::<&webkit2gtk::gio::Cancellable>, move |result| {
                        let result = (|| -> Result<(),String> {
                            let surface = result.map_err(|e|e.to_string())?;
                            let mut file = std::fs::File::create(destination).map_err(|e|e.to_string())?;
                            surface.write_to_png(&mut file).map_err(|e|e.to_string())
                        })();
                        let _ = sender.emit("ui-test-snapshot-done", serde_json::json!({"theme":theme,"ok":result.is_ok(),"error":result.err()}));
                    });
            });
        });
    }
}

pub fn page_loaded(webview: &tauri::Webview, payload: &tauri::webview::PageLoadPayload<'_>) {
    if !crate::desktop_test_mode()
        || webview.label() != "player"
        || payload.event() != tauri::webview::PageLoadEvent::Finished
    {
        return;
    }
    if let Some(path) = std::env::var_os("SAYIT_UI_TEST_SCRIPT") {
        match std::fs::read_to_string(path) {
            Ok(script) => {
                if let Err(error) = webview.eval(&script) {
                    eprintln!("cannot start UI test: {error}");
                }
            }
            Err(error) => eprintln!("cannot read UI test: {error}"),
        }
    }
}
