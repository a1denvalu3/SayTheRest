use serde_json::Value;
use std::{
    future::Future,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

/// Both desktop windows share one authoritative snapshot per service event.
/// Mutations invalidate immediately; the age limit also handles reconnects.
#[derive(Default)]
pub struct SnapshotCache {
    revision: AtomicU64,
    cached: Mutex<Option<(u64, Instant, Value)>>,
}
impl SnapshotCache {
    pub fn invalidate(&self) {
        self.revision.fetch_add(1, Ordering::Relaxed);
    }
    pub async fn get<F, Fut>(&self, fetch: F) -> Result<Value, String>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Value, String>>,
    {
        let mut cached = self.cached.lock().await;
        let revision = self.revision.load(Ordering::Relaxed);
        if let Some((version, created, value)) = cached.as_ref()
            && *version == revision
            && created.elapsed() < Duration::from_secs(2)
        {
            return Ok(value.clone());
        }
        let value = fetch().await?;
        *cached = Some((revision, Instant::now(), value.clone()));
        Ok(value)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn windows_share_fetches_and_mutations_expiration_and_errors_stay_fresh() {
        let cache = SnapshotCache::default();
        let requests = AtomicU64::new(0);
        let fetch = || async {
            let count = requests.fetch_add(1, Ordering::Relaxed) + 1;
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(serde_json::json!({"sequence":count}))
        };
        let (a, b) = tokio::join!(cache.get(fetch), cache.get(fetch));
        assert_eq!(a.unwrap(), b.unwrap());
        assert_eq!(requests.load(Ordering::Relaxed), 1);
        cache.invalidate();
        assert_eq!(cache.get(fetch).await.unwrap()["sequence"], 2);
        cache.cached.lock().await.as_mut().unwrap().1 = Instant::now() - Duration::from_secs(3);
        assert_eq!(cache.get(fetch).await.unwrap()["sequence"], 3);
        cache.invalidate();
        assert!(
            cache
                .get(|| async { Err("service unavailable".into()) })
                .await
                .is_err()
        );
        assert_eq!(cache.get(fetch).await.unwrap()["sequence"], 4);
    }
}
