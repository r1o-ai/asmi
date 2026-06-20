//! SCDynamicStore link watcher — subscribes to macOS Thunderbolt interface
//! link state changes and emits ConnectionPoolEvent notifications.
//!
//! On link_down, immediately triggers HeartbeatAll (instead of waiting for
//! the 120s safety-net timer). On critical events, POSTs a webhook alert
//! to the web app for push/iMessage relay via Hermes.
//!
//! The SCDynamicStore callback runs on a dedicated OS thread (CFRunLoop
//! requirement — cannot use tokio). The webhook sender also runs on its
//! own bounded thread to prevent unbounded thread spawn on cable flap.

use core_foundation::{
    array::CFArray,
    base::{CFType, TCFType, ToVoid},
    dictionary::CFDictionary,
    propertylist::CFPropertyList,
    runloop::CFRunLoop,
    string::CFString,
};
use system_configuration::dynamic_store::{
    SCDynamicStore, SCDynamicStoreBuilder, SCDynamicStoreCallBackContext,
};
use std::sync::Arc;
use tokio::sync::broadcast;

struct LinkWatcherCtx {
    events_tx: broadcast::Sender<String>,
    jaccl_worker: Arc<crate::transfer::JacclWorker>,
    /// Bounded sender for webhook alerts (prevents thread-per-event leak)
    alert_tx: Option<std::sync::mpsc::SyncSender<String>>,
}

/// Callback invoked by SCDynamicStore when watched keys change.
fn link_state_callback(
    store: SCDynamicStore,
    changed_keys: CFArray<CFString>,
    ctx: &mut LinkWatcherCtx,
) {
    for key in changed_keys.iter() {
        let key_str = key.to_string();
        // Key format: State:/Network/Interface/en3/Link
        let iface = key_str
            .strip_prefix("State:/Network/Interface/")
            .and_then(|s| s.strip_suffix("/Link"))
            .unwrap_or("unknown")
            .to_string();

        // Read current link status from the store.
        // The value is a CFDictionary with an "Active" key (CFBoolean).
        let active = store
            .get(key_str.as_str())
            .and_then(CFPropertyList::downcast_into::<CFDictionary>)
            .and_then(|dict| {
                let active_key = CFString::new("Active");
                dict.find(active_key.to_void())
                    .map(|ptr| {
                        let val = unsafe { CFType::wrap_under_get_rule(*ptr) };
                        // CFBoolean: compare against kCFBooleanTrue
                        val.downcast_into::<core_foundation::boolean::CFBoolean>()
                            .map(|b| bool::from(b))
                            .unwrap_or(false)
                    })
            })
            .unwrap_or(false);

        if active {
            let ev = crate::transfer::ConnectionPoolEvent::LinkUp {
                iface: iface.clone(),
            };
            let _ = ctx.events_tx.send(serde_json::to_string(&ev).unwrap_or_default());
            tracing::info!(iface = %iface, "TB link up");
        } else {
            let ev = crate::transfer::ConnectionPoolEvent::LinkDown {
                iface: iface.clone(),
            };
            let _ = ctx.events_tx.send(serde_json::to_string(&ev).unwrap_or_default());
            tracing::warn!(iface = %iface, "TB link down — triggering immediate heartbeat");

            // Immediate heartbeat to probe affected connections
            let _ = ctx.jaccl_worker.send(crate::transfer::JacclCmd::HeartbeatAll {
                events_tx: ctx.events_tx.clone(),
            });

            // Alert via bounded channel (never spawns unbounded threads)
            if let Some(ref tx) = ctx.alert_tx {
                let body = serde_json::json!({
                    "type": "rdma_alert",
                    "severity": "warning",
                    "title": format!("RDMA link down: {iface}"),
                    "body": format!("Thunderbolt interface {iface} lost link. Heartbeat probe triggered."),
                });
                let _ = tx.try_send(body.to_string()); // drop if full (bounded=8)
            }
        }
    }
}

/// Spawn the link watcher on a dedicated OS thread (CFRunLoop requirement).
pub fn spawn_link_watcher(
    tb_interfaces: Vec<String>,
    events_tx: broadcast::Sender<String>,
    jaccl_worker: Arc<crate::transfer::JacclWorker>,
    webhook_url: Option<String>,
) {
    // Bounded alert channel: a single sender thread POSTs webhooks.
    // Prevents unbounded thread spawn on cable flap.
    let alert_tx = webhook_url.map(|url| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<String>(8);
        std::thread::Builder::new()
            .name("alert-sender".into())
            .spawn(move || {
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(5))
                    .build()
                    .unwrap();
                for body in rx {
                    let _ = client
                        .post(&url)
                        .header("Content-Type", "application/json")
                        .body(body)
                        .send();
                }
            })
            .expect("spawn alert-sender");
        tx
    });

    std::thread::Builder::new()
        .name("link-watcher".into())
        .spawn(move || {
            let ctx = SCDynamicStoreCallBackContext {
                callout: link_state_callback,
                info: LinkWatcherCtx {
                    events_tx,
                    jaccl_worker,
                    alert_tx,
                },
            };

            // build() returns Option<SCDynamicStore> — unwrap here (fatal if unavailable)
            let store = SCDynamicStoreBuilder::new("asmi-link-watcher")
                .callback_context(ctx)
                .build()
                .expect("SCDynamicStore creation failed");

            // Watch for link state changes on Thunderbolt interfaces.
            // Keys: State:/Network/Interface/<iface>/Link
            let keys: Vec<CFString> = tb_interfaces
                .iter()
                .map(|iface| CFString::new(&format!("State:/Network/Interface/{iface}/Link")))
                .collect();
            let cf_keys = CFArray::from_CFTypes(&keys);
            let empty_patterns: CFArray<CFString> = CFArray::from_CFTypes(&[]);

            store.set_notification_keys(&cf_keys, &empty_patterns);

            // create_run_loop_source returns Option<CFRunLoopSource>
            let run_loop_source = store
                .create_run_loop_source()
                .expect("SCDynamicStore run loop source creation failed");
            let run_loop = CFRunLoop::get_current();
            run_loop.add_source(
                &run_loop_source,
                unsafe { core_foundation::runloop::kCFRunLoopCommonModes },
            );

            // Blocks forever — dedicated thread. Killed on process exit.
            CFRunLoop::run_current();
        })
        .expect("spawn link-watcher thread");
}
