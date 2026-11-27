//! Linux-aware CPU topology discovery, thread-count recommendations, and thread
//! affinity helpers. On non-Linux targets, falls back cleanly to portable
//! logical-core heuristics without platform-specific syscalls.

use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuTopology {
    /// One primary logical CPU per physical core, ordered fastest-first.
    pub physical_cpus: Vec<usize>,
    /// All logical CPU siblings belonging to each physical core (same order as `physical_cpus`).
    pub physical_core_siblings: Vec<Vec<usize>>,
    /// Performance-tier physical CPUs (P-cores on hybrid chips, or all physical
    /// CPUs on uniform CPUs).
    pub performance_cpus: Vec<usize>,
    /// Secondary efficiency-tier physical CPUs (e.g. Intel E-cores, excluding
    /// ultra-low-power LP E-cores when separable).
    pub efficiency_cpus: Vec<usize>,
    /// Number of online/allowed logical CPUs.
    pub logical_cpus: usize,
}

impl CpuTopology {
    pub fn detect() -> &'static Self {
        static TOPOLOGY: OnceLock<CpuTopology> = OnceLock::new();
        TOPOLOGY.get_or_init(detect_uncached)
    }

    /// Return all logical CPUs (SMT siblings) for the physical core owning `cpu`.
    pub fn siblings_of(&self, cpu: usize) -> Vec<usize> {
        if let Some(pos) = self.physical_cpus.iter().position(|&c| c == cpu)
            && let Some(sibs) = self.physical_core_siblings.get(pos)
            && !sibs.is_empty()
        {
            return sibs.clone();
        }
        vec![cpu]
    }

    /// Select up to `threads` physical CPUs, prioritizing performance cores
    /// first, then efficiency cores, then remaining physical cores.
    pub fn select_worker_cpus(&self, threads: usize) -> Vec<usize> {
        if threads == 0 || self.physical_cpus.is_empty() {
            return Vec::new();
        }
        let mut selected = Vec::with_capacity(threads.min(self.physical_cpus.len()));
        for &cpu in &self.performance_cpus {
            if selected.len() == threads {
                return selected;
            }
            selected.push(cpu);
        }
        for &cpu in &self.efficiency_cpus {
            if selected.len() == threads {
                return selected;
            }
            if !selected.contains(&cpu) {
                selected.push(cpu);
            }
        }
        for &cpu in &self.physical_cpus {
            if selected.len() == threads {
                return selected;
            }
            if !selected.contains(&cpu) {
                selected.push(cpu);
            }
        }
        selected
    }

    /// CPUs dedicated to a background pipelined decoder on hybrid machines
    /// (e.g. Intel Meteor Lake / Arrow Lake E-cores running concurrently with
    /// P-core autoregressive generation). Returns empty when no separate
    /// efficiency tier of at least 4 cores exists.
    pub fn pipelined_decoder_cpus(&self, generator_cpus: &[usize]) -> Vec<usize> {
        let available: Vec<usize> = self
            .efficiency_cpus
            .iter()
            .copied()
            .filter(|cpu| !generator_cpus.contains(cpu))
            .collect();
        if available.len() >= 4 {
            available
        } else {
            Vec::new()
        }
    }

    /// Recommended thread count for native Qwen3-TTS ONNX CPU inference.
    /// Caps at the performance-core count (max 6) so autoregressive generation
    /// avoids SMT and slow LP E-core stragglers while leaving E-cores available
    /// for pipelined audio decoding and the desktop UI.
    pub fn recommended_qwen_onnx_threads(&self) -> usize {
        let perf = self.performance_cpus.len().max(1);
        perf.clamp(1, 6)
    }

    /// Recommended thread count for sherpa-onnx CPU models.
    /// Kokoro peaks at 4 performance cores (scaling past 4 threads degrades
    /// throughput on small graphs), while Piper/Kitten/Pocket benefit up to 6.
    pub fn recommended_sherpa_threads(&self, is_kokoro: bool) -> usize {
        let cap = if is_kokoro { 4 } else { 6 };
        self.performance_cpus.len().max(1).clamp(1, cap)
    }

    /// Format ONNX Runtime's 1-based intra-op worker affinity string for
    /// `threads` total threads (`threads - 1` worker threads plus caller).
    /// Returns `None` if `threads <= 1` or not enough distinct CPUs exist.
    pub fn ort_intra_affinity(cpus: &[usize], threads: usize) -> Option<String> {
        if threads <= 1 || cpus.len() < threads {
            return None;
        }
        let workers = &cpus[1..threads];
        let mut out = String::with_capacity(workers.len() * 4);
        for (idx, &cpu) in workers.iter().enumerate() {
            if idx > 0 {
                out.push(';');
            }
            // ONNX Runtime expects 1-based logical processor IDs.
            use std::fmt::Write as _;
            let _ = write!(&mut out, "{}", cpu + 1);
        }
        Some(out)
    }
}

pub fn detect() -> &'static CpuTopology {
    CpuTopology::detect()
}

pub fn recommended_qwen_onnx_threads() -> usize {
    CpuTopology::detect().recommended_qwen_onnx_threads()
}

pub fn recommended_sherpa_threads(is_kokoro: bool) -> usize {
    CpuTopology::detect().recommended_sherpa_threads(is_kokoro)
}

#[derive(Clone, Debug)]
struct RawCore {
    cpu_id: usize,
    package_id: usize,
    core_id: usize,
    primary_sibling: usize,
    score: u64,
}

fn build_topology(raw_cpus: &[RawCore]) -> CpuTopology {
    if raw_cpus.is_empty() {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let fallback: Vec<usize> = (0..logical).collect();
        let siblings = fallback.iter().map(|&c| vec![c]).collect();
        return CpuTopology {
            physical_cpus: fallback.clone(),
            physical_core_siblings: siblings,
            performance_cpus: fallback,
            efficiency_cpus: Vec::new(),
            logical_cpus: logical,
        };
    }

    let logical_cpus = raw_cpus.len();
    // Collapse SMT siblings by (package_id, core_id, primary_sibling), keeping
    // the lowest logical CPU ID (and highest reported score) per physical core
    // while remembering every sibling logical CPU on that physical core.
    let mut cores: Vec<(usize, u64, Vec<usize>)> = Vec::new();
    let mut seen: Vec<(usize, usize, usize)> = Vec::new();
    let mut sorted = raw_cpus.to_vec();
    sorted.sort_by_key(|c| c.cpu_id);
    for c in &sorted {
        let key = (c.package_id, c.core_id, c.primary_sibling);
        if let Some(pos) = seen.iter().position(|k| *k == key) {
            if c.score > cores[pos].1 {
                cores[pos].1 = c.score;
            }
            cores[pos].2.push(c.cpu_id);
        } else {
            seen.push(key);
            cores.push((c.cpu_id, c.score, vec![c.cpu_id]));
        }
    }

    // Order fastest-first, breaking ties by ascending CPU ID.
    cores.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let physical_cpus: Vec<usize> = cores.iter().map(|(id, _, _)| *id).collect();
    let physical_core_siblings: Vec<Vec<usize>> =
        cores.iter().map(|(_, _, sibs)| sibs.clone()).collect();
    let max_score = cores.first().map(|(_, s, _)| *s).unwrap_or(0);
    let min_score = cores.last().map(|(_, s, _)| *s).unwrap_or(0);

    let (performance_cpus, efficiency_cpus) =
        if max_score > 0 && min_score.saturating_mul(100) < max_score.saturating_mul(90) {
            let perf_threshold = max_score.saturating_mul(90) / 100;
            let perf: Vec<usize> = cores
                .iter()
                .filter(|(_, s, _)| *s >= perf_threshold)
                .map(|(id, _, _)| *id)
                .collect();
            let remaining: Vec<(usize, u64)> = cores
                .iter()
                .filter(|(_, s, _)| *s < perf_threshold)
                .map(|(id, s, _)| (*id, *s))
                .collect();
            let eff_max = remaining.first().map(|(_, s)| *s).unwrap_or(0);
            let eff_threshold = eff_max.saturating_mul(80) / 100;
            let eff: Vec<usize> = remaining
                .into_iter()
                .filter(|(_, s)| *s >= eff_threshold)
                .map(|(id, _)| id)
                .collect();
            (perf, eff)
        } else {
            (physical_cpus.clone(), Vec::new())
        };

    CpuTopology {
        physical_cpus,
        physical_core_siblings,
        performance_cpus,
        efficiency_cpus,
        logical_cpus,
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use std::fs;
    use std::path::Path;

    const CPU_SET_WORDS: usize = 16; // 1024 CPUs on 64-bit Linux (128 bytes)
    const BITS_PER_WORD: usize = usize::BITS as usize;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CpuSet {
        bits: [usize; CPU_SET_WORDS],
    }

    impl CpuSet {
        fn empty() -> Self {
            Self {
                bits: [0; CPU_SET_WORDS],
            }
        }

        fn set(&mut self, cpu: usize) {
            let word = cpu / BITS_PER_WORD;
            let bit = cpu % BITS_PER_WORD;
            if word < CPU_SET_WORDS {
                self.bits[word] |= 1usize << bit;
            }
        }

        fn is_set(&self, cpu: usize) -> bool {
            let word = cpu / BITS_PER_WORD;
            let bit = cpu % BITS_PER_WORD;
            word < CPU_SET_WORDS && (self.bits[word] & (1usize << bit)) != 0
        }

        fn is_empty(&self) -> bool {
            self.bits.iter().all(|&w| w == 0)
        }
    }

    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut CpuSet) -> i32;
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const CpuSet) -> i32;
    }

    pub struct AffinityGuard {
        previous: CpuSet,
    }

    impl Drop for AffinityGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &self.previous);
            }
        }
    }

    pub fn current_affinity() -> Option<CpuSet> {
        let mut mask = CpuSet::empty();
        let rc = unsafe { sched_getaffinity(0, std::mem::size_of::<CpuSet>(), &mut mask) };
        (rc == 0 && !mask.is_empty()).then_some(mask)
    }

    pub fn pin_current_thread(cpus: &[usize]) -> Option<AffinityGuard> {
        if cpus.is_empty() {
            return None;
        }
        let previous = current_affinity()?;
        let mut next = CpuSet::empty();
        for &cpu in cpus {
            if previous.is_set(cpu) {
                next.set(cpu);
            }
        }
        if next.is_empty() {
            return None;
        }
        let rc = unsafe { sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &next) };
        (rc == 0).then_some(AffinityGuard { previous })
    }

    pub fn set_current_thread_affinity(cpus: &[usize]) -> bool {
        if cpus.is_empty() {
            return false;
        }
        let allowed = current_affinity();
        let mut next = CpuSet::empty();
        for &cpu in cpus {
            if allowed.is_none_or(|mask| mask.is_set(cpu)) {
                next.set(cpu);
            }
        }
        if next.is_empty() {
            return false;
        }
        unsafe { sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &next) == 0 }
    }

    fn read_trimmed(path: &Path) -> Option<String> {
        fs::read_to_string(path).ok().map(|s| s.trim().to_owned())
    }

    fn parse_first_cpu(list: &str) -> Option<usize> {
        let token = list.split(',').next()?.trim();
        let first = token.split('-').next()?.trim();
        first.parse::<usize>().ok()
    }

    pub fn detect_uncached() -> CpuTopology {
        let allowed = current_affinity();
        let cpu_root = Path::new("/sys/devices/system/cpu");
        let Ok(entries) = fs::read_dir(cpu_root) else {
            return build_topology(&[]);
        };
        let mut raw = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            let Some(id_str) = name_str.strip_prefix("cpu") else {
                continue;
            };
            let Ok(cpu_id) = id_str.parse::<usize>() else {
                continue;
            };
            if let Some(mask) = allowed
                && !mask.is_set(cpu_id)
            {
                continue;
            }
            let dir = entry.path();
            if read_trimmed(&dir.join("online")).as_deref() == Some("0") {
                continue;
            }
            let package_id = read_trimmed(&dir.join("topology/physical_package_id"))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let core_id = read_trimmed(&dir.join("topology/core_id"))
                .and_then(|v| v.parse().ok())
                .unwrap_or(cpu_id);
            let primary_sibling = read_trimmed(&dir.join("topology/thread_siblings_list"))
                .and_then(|v| parse_first_cpu(&v))
                .unwrap_or(cpu_id);
            let score = read_trimmed(&dir.join("cpufreq/cpuinfo_max_freq"))
                .and_then(|v| v.parse::<u64>().ok())
                .or_else(|| {
                    read_trimmed(&dir.join("cpu_capacity")).and_then(|v| v.parse::<u64>().ok())
                })
                .unwrap_or(0);
            raw.push(RawCore {
                cpu_id,
                package_id,
                core_id,
                primary_sibling,
                score,
            });
        }
        build_topology(&raw)
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::*;

    pub struct AffinityGuard;

    pub fn pin_current_thread(_cpus: &[usize]) -> Option<AffinityGuard> {
        None
    }

    pub fn set_current_thread_affinity(_cpus: &[usize]) -> bool {
        false
    }

    pub fn detect_uncached() -> CpuTopology {
        build_topology(&[])
    }
}

pub use platform::{AffinityGuard, pin_current_thread, set_current_thread_affinity};

fn detect_uncached() -> CpuTopology {
    platform::detect_uncached()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_meteor_lake_topology_separates_p_e_and_lp_e_cores() {
        let mut raw = Vec::new();
        // 6 P-cores (2 with 5.1 GHz, 4 with 4.8 GHz, each with 2 SMT siblings)
        let p_specs = [
            (0, 5, 16, 4_800_000),
            (1, 2, 8, 5_100_000),
            (3, 4, 12, 5_100_000),
            (6, 7, 20, 4_800_000),
            (8, 9, 24, 4_800_000),
            (10, 11, 28, 4_800_000),
        ];
        for (a, b, core, freq) in p_specs {
            let primary = a.min(b);
            for cpu in [a, b] {
                raw.push(RawCore {
                    cpu_id: cpu,
                    package_id: 0,
                    core_id: core,
                    primary_sibling: primary,
                    score: freq,
                });
            }
        }
        // 8 E-cores @ 3.8 GHz
        for (idx, cpu) in (12..20).enumerate() {
            raw.push(RawCore {
                cpu_id: cpu,
                package_id: 0,
                core_id: idx,
                primary_sibling: cpu,
                score: 3_800_000,
            });
        }
        // 2 LP E-cores @ 2.5 GHz
        for (idx, cpu) in [20, 21].into_iter().enumerate() {
            raw.push(RawCore {
                cpu_id: cpu,
                package_id: 0,
                core_id: 32 + idx,
                primary_sibling: cpu,
                score: 2_500_000,
            });
        }

        let topo = build_topology(&raw);
        assert_eq!(topo.logical_cpus, 22);
        assert_eq!(topo.physical_cpus.len(), 16);
        assert_eq!(topo.performance_cpus, vec![1, 3, 0, 6, 8, 10]);
        assert_eq!(topo.efficiency_cpus, vec![12, 13, 14, 15, 16, 17, 18, 19]);
        assert_eq!(topo.recommended_qwen_onnx_threads(), 6);
        assert_eq!(topo.recommended_sherpa_threads(false), 6);
        assert_eq!(topo.recommended_sherpa_threads(true), 4);
        let gen_cpus = topo.select_worker_cpus(6);
        assert_eq!(gen_cpus, vec![1, 3, 0, 6, 8, 10]);
        assert_eq!(
            topo.pipelined_decoder_cpus(&gen_cpus),
            vec![12, 13, 14, 15, 16, 17, 18, 19]
        );
        assert_eq!(
            CpuTopology::ort_intra_affinity(&gen_cpus, 6).as_deref(),
            Some("4;1;7;9;11")
        );
    }

    #[test]
    fn uniform_cpu_keeps_all_physical_cores_in_performance_tier() {
        let mut raw = Vec::new();
        for core in 0..8 {
            for sibling in [core, core + 8] {
                raw.push(RawCore {
                    cpu_id: sibling,
                    package_id: 0,
                    core_id: core,
                    primary_sibling: core,
                    score: 4_500_000,
                });
            }
        }
        let topo = build_topology(&raw);
        assert_eq!(topo.logical_cpus, 16);
        assert_eq!(topo.physical_cpus, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(topo.performance_cpus, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(topo.efficiency_cpus.is_empty());
        assert_eq!(topo.recommended_qwen_onnx_threads(), 6);
        assert_eq!(topo.recommended_sherpa_threads(true), 4);
        assert!(topo.pipelined_decoder_cpus(&topo.select_worker_cpus(6)).is_empty());
    }

    #[test]
    fn live_topology_detection_and_affinity_guard_work() {
        let topo = detect();
        assert!(!topo.physical_cpus.is_empty());
        assert!(!topo.performance_cpus.is_empty());
        assert!(topo.recommended_qwen_onnx_threads() >= 1);
        let cpus = topo.select_worker_cpus(2);
        let _guard = pin_current_thread(&cpus);
    }
}
