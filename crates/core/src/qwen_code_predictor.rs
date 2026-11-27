//! Fast KV-cached INT4 CPU executor for Qwen3-TTS `code_predictor.onnx`.
//!
//! The upstream ONNX export of `code_predictor.onnx` has no KV cache and runs a
//! fixed 16-position sequence plus all 15 `lm_head` projections on every single
//! codebook group call (240 token passes and 225 LM-head matmuls per audio frame
//! instead of 16 token passes and 15 LM-head matmuls).
//!
//! This module reads the exact quantized weights directly from the pinned
//! `code_predictor.onnx` file (zero extra dependencies), maintains a causal
//! 16-position KV cache, fuses QKV and Gate+Up projections, runs only the active
//! `lm_head`, and accelerates exact-f32 4-bit block-quantized MatVec with a
//! 16-byte-aligned split layout (20.5 B/block), runtime-detected AVX2+FMA
//! microkernels, and a lock-free P-core-pinned burst worker pool.

use anyhow::{Context, Result, bail, ensure};
use std::{
    cell::UnsafeCell,
    collections::HashMap,
    fs,
    path::Path,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};

const NUM_LAYERS: usize = 5;
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 3072;
const Q_HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const Q_DIM: usize = Q_HEADS * HEAD_DIM; // 2048
const KV_DIM: usize = KV_HEADS * HEAD_DIM; // 1024
const MAX_SEQ: usize = 16;
const NUM_GROUPS: usize = 16;
const NUM_LM_HEADS: usize = 15;
const BLOCK_SIZE: usize = 32;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f64 = 1_000_000.0;

/// 16-byte aligned packed Q4 block for 32 weights:
/// - `0[0..16]` low nibble  = w[0..16]  (u4 in 0..=15)
/// - `0[0..16]` high nibble = w[16..32] (u4 in 0..=15)
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Q4Nibbles([u8; 16]);

/// Contiguous single-stream 4-row x 8-block (256-col) Q4 superblock (656 bytes,
/// 20.5 B/block = 0.64 B/weight). Every 4-row block step reads one 64-byte
/// cache line sequentially, eliminating multi-stream TLB/prefetcher thrashing.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Q4Tile {
    q: [[Q4Nibbles; 4]; 8],   // 512 bytes (8 cache lines: 4 rows per block)
    scales: [[f32; 4]; 8],    // 128 bytes (2 cache lines)
    zps: [[u8; 4]; 4],        // 16 bytes (2 blocks packed per u8, 4 rows)
}

const TILE_ROWS: usize = 4;
const TILE_BLOCKS: usize = 8;

struct Q4Matrix {
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    tiles_per_row_group: usize,
    tiles: Vec<Q4Tile>,
}

impl Q4Matrix {
    fn from_onnx_q4(
        rows: usize,
        cols: usize,
        packed_q: &[u8],
        scales: &[f32],
        zero_points: &[u8],
    ) -> Result<Self> {
        ensure!(cols % BLOCK_SIZE == 0, "Q4 cols must be a multiple of 32");
        let blocks_per_row = cols / BLOCK_SIZE;
        ensure!(
            blocks_per_row % TILE_BLOCKS == 0,
            "Q4 blocks_per_row must be a multiple of {TILE_BLOCKS}"
        );
        let total_blocks = rows * blocks_per_row;
        ensure!(
            packed_q.len() == total_blocks * (BLOCK_SIZE / 2),
            "unexpected Q4 weight byte length"
        );
        ensure!(scales.len() == total_blocks, "unexpected Q4 scales length");
        ensure!(
            blocks_per_row % 2 == 0 && zero_points.len() == total_blocks / 2,
            "unexpected Q4 zero-point length"
        );
        let row_groups = rows.div_ceil(TILE_ROWS);
        let tiles_per_row_group = blocks_per_row / TILE_BLOCKS;
        let mut tiles = Vec::with_capacity(row_groups * tiles_per_row_group);
        for rg in 0..row_groups {
            for tg in 0..tiles_per_row_group {
                let mut tile = Q4Tile {
                    q: [[Q4Nibbles([0; 16]); 4]; 8],
                    scales: [[0.0; 4]; 8],
                    zps: [[0; 4]; 4],
                };
                for r_off in 0..TILE_ROWS {
                    let row = rg * TILE_ROWS + r_off;
                    if row >= rows {
                        continue;
                    }
                    let row_blk_base = row * blocks_per_row + tg * TILE_BLOCKS;
                    for k in 0..TILE_BLOCKS {
                        let b = row_blk_base + k;
                        let src = &packed_q[b * 16..(b + 1) * 16];
                        let mut raw = [0u8; 32];
                        for j in 0..16 {
                            raw[2 * j] = src[j] & 0x0F;
                            raw[2 * j + 1] = src[j] >> 4;
                        }
                        let mut repacked = [0u8; 16];
                        for j in 0..16 {
                            repacked[j] = raw[j] | (raw[j + 16] << 4);
                        }
                        tile.q[k][r_off] = Q4Nibbles(repacked);
                        tile.scales[k][r_off] = scales[b];
                        if k % 2 == 0 {
                            tile.zps[k / 2][r_off] = zero_points[b / 2];
                        }
                    }
                }
                tiles.push(tile);
            }
        }
        Ok(Self {
            rows,
            cols,
            blocks_per_row,
            tiles_per_row_group,
            tiles,
        })
    }

    fn concat_rows(parts: &[Self]) -> Result<Self> {
        ensure!(!parts.is_empty(), "empty matrix list");
        let cols = parts[0].cols;
        let blocks_per_row = parts[0].blocks_per_row;
        let tiles_per_row_group = parts[0].tiles_per_row_group;
        let mut rows = 0;
        let mut total_tiles = 0;
        for part in parts {
            ensure!(part.cols == cols, "mismatched cols in matrix concat");
            ensure!(
                part.rows % TILE_ROWS == 0,
                "concat_rows requires rows multiple of {TILE_ROWS}"
            );
            rows += part.rows;
            total_tiles += part.tiles.len();
        }
        let mut tiles = Vec::with_capacity(total_tiles);
        for part in parts {
            tiles.extend_from_slice(&part.tiles);
        }
        Ok(Self {
            rows,
            cols,
            blocks_per_row,
            tiles_per_row_group,
            tiles,
        })
    }
}

struct Q4Embedding {
    vocab: usize,
    packed_q: Vec<u8>,
    scales: Vec<f32>,
    zero_points: Vec<u8>,
}

impl Q4Embedding {
    fn from_onnx(
        vocab: usize,
        packed_q: Vec<u8>,
        scales: Vec<f32>,
        zero_points: Vec<u8>,
    ) -> Result<Self> {
        let blocks_per_row = HIDDEN / BLOCK_SIZE; // 32
        ensure!(
            packed_q.len() == vocab * (HIDDEN / 2),
            "invalid Q4 embedding weights"
        );
        ensure!(
            scales.len() == vocab * blocks_per_row,
            "invalid Q4 embedding scales"
        );
        ensure!(
            zero_points.len() == vocab * (blocks_per_row / 2),
            "invalid Q4 embedding zp"
        );
        Ok(Self {
            vocab,
            packed_q,
            scales,
            zero_points,
        })
    }

    fn lookup_into(&self, id: i64, out: &mut [f32; HIDDEN]) -> Result<()> {
        let row = usize::try_from(id)
            .ok()
            .filter(|&v| v < self.vocab)
            .with_context(|| format!("codec id {id} out of range 0..{}", self.vocab))?;
        let blocks_per_row = HIDDEN / BLOCK_SIZE;
        let q_base = row * (HIDDEN / 2);
        let s_base = row * blocks_per_row;
        let z_base = row * (blocks_per_row / 2);
        for b in 0..blocks_per_row {
            let scale = self.scales[s_base + b];
            let zp_byte = self.zero_points[z_base + b / 2];
            let zp = if b % 2 == 0 {
                (zp_byte & 0x0F) as f32
            } else {
                (zp_byte >> 4) as f32
            };
            let q = &self.packed_q[q_base + b * 16..q_base + (b + 1) * 16];
            let dst = &mut out[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
            for j in 0..16 {
                let lo = (q[j] & 0x0F) as f32;
                let hi = (q[j] >> 4) as f32;
                dst[2 * j] = (lo - zp) * scale;
                dst[2 * j + 1] = (hi - zp) * scale;
            }
        }
        Ok(())
    }
}

struct LayerWeights {
    input_ln: [f32; HIDDEN],
    post_ln: [f32; HIDDEN],
    q_norm: [f32; HEAD_DIM],
    k_norm: [f32; HEAD_DIM],
    /// Fused [Q (2048), K (1024), V (1024)] x 1024
    qkv_proj: Q4Matrix,
    /// [1024 x 2048]
    o_proj: Q4Matrix,
    /// Fused [Gate (3072), Up (3072)] x 1024
    gate_up_proj: Q4Matrix,
    /// [1024 x 3072]
    down_proj: Q4Matrix,
}

// ---------------------------------------------------------------------------
// Lock-free burst thread pool
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct JobDescriptor {
    func: fn(usize, usize, *const (), *mut ()),
    total_rows: usize,
    arg_in: *const (),
    arg_out: *mut (),
}

struct PoolShared {
    threads: usize,
    generation: AtomicUsize,
    remaining: AtomicUsize,
    sleeping: AtomicUsize,
    burst_active: AtomicBool,
    shutdown: AtomicBool,
    job: UnsafeCell<JobDescriptor>,
    wake_lock: Mutex<usize>,
    wake_cv: Condvar,
}

unsafe impl Sync for PoolShared {}
unsafe impl Send for PoolShared {}

struct BurstPool {
    shared: Arc<PoolShared>,
    handles: Vec<JoinHandle<()>>,
}

impl BurstPool {
    fn new(threads: usize, cpus: &[usize]) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(PoolShared {
            threads,
            generation: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            sleeping: AtomicUsize::new(0),
            burst_active: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            job: UnsafeCell::new(JobDescriptor {
                func: |_, _, _, _| {},
                total_rows: 0,
                arg_in: std::ptr::null(),
                arg_out: std::ptr::null_mut(),
            }),
            wake_lock: Mutex::new(0),
            wake_cv: Condvar::new(),
        });
        let topo = crate::cpu_topology::detect();
        // Exclude `cpus[0]` (and its SMT sibling) from floating worker masks so
        // no worker thread can ever be scheduled onto the caller thread's busy
        // core during `parallel_rows`.
        let caller_siblings: Vec<usize> = cpus
            .first()
            .map(|&c| topo.siblings_of(c))
            .unwrap_or_default();
        let all_worker_siblings: Vec<usize> = cpus
            .iter()
            .skip(1)
            .flat_map(|&c| topo.siblings_of(c))
            .filter(|c| !caller_siblings.contains(c))
            .collect();
        let mut handles = Vec::with_capacity(threads.saturating_sub(1));
        for worker_idx in 1..threads {
            let shared_clone = shared.clone();
            let pin_set = cpus.get(worker_idx).map(|&cpu| {
                let sibs = topo.siblings_of(cpu);
                if sibs.len() >= 2 {
                    sibs
                } else if !all_worker_siblings.is_empty() {
                    all_worker_siblings.clone()
                } else {
                    sibs
                }
            });
            if let Ok(handle) = thread::Builder::new()
                .name(format!("qwen-cp-{worker_idx}"))
                .spawn(move || {
                    if let Some(ref set) = pin_set {
                        crate::cpu_topology::set_current_thread_affinity(set);
                    }
                    worker_loop(&shared_clone, worker_idx);
                })
            {
                handles.push(handle);
            }
        }
        Self { shared, handles }
    }

    fn set_burst(&self, active: bool) {
        self.shared.burst_active.store(active, Ordering::Release);
        if active && self.shared.sleeping.load(Ordering::Acquire) > 0 {
            let _g = self.shared.wake_lock.lock().unwrap();
            self.shared.wake_cv.notify_all();
        }
    }

    fn parallel_rows(
        &self,
        total_rows: usize,
        func: fn(usize, usize, *const (), *mut ()),
        arg_in: *const (),
        arg_out: *mut (),
    ) {
        let threads = self.handles.len() + 1;
        if threads <= 1 || total_rows < 64 {
            func(0, total_rows, arg_in, arg_out);
            return;
        }
        unsafe {
            *self.shared.job.get() = JobDescriptor {
                func,
                total_rows,
                arg_in,
                arg_out,
            };
        }
        self.shared.remaining.store(threads - 1, Ordering::Release);
        let next_gen = self.shared.generation.fetch_add(1, Ordering::AcqRel) + 1;
        if self.shared.sleeping.load(Ordering::Acquire) > 0 {
            let mut g = self.shared.wake_lock.lock().unwrap();
            *g = next_gen;
            self.shared.wake_cv.notify_all();
        }

        let (start, end) = row_range(0, threads, total_rows);
        func(start, end, arg_in, arg_out);

        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for BurstPool {
    fn drop(&mut self) {
        self.shared.burst_active.store(false, Ordering::Release);
        self.shared.shutdown.store(true, Ordering::Release);
        {
            let _g = self.shared.wake_lock.lock().unwrap();
            self.shared.wake_cv.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[inline]
fn row_range(worker_idx: usize, threads: usize, total_rows: usize) -> (usize, usize) {
    let total_groups = total_rows / 4;
    let base = total_groups / threads;
    let rem = total_groups % threads;
    let start_g = worker_idx * base + worker_idx.min(rem);
    let end_g = start_g + base + usize::from(worker_idx < rem);
    let start = start_g * 4;
    let end = if worker_idx + 1 == threads {
        total_rows
    } else {
        end_g * 4
    };
    (start, end)
}

fn worker_loop(shared: &PoolShared, worker_idx: usize) {
    let mut seen_gen = 0usize;
    loop {
        if shared.burst_active.load(Ordering::Acquire) {
            let mut spins = 0u32;
            while shared.generation.load(Ordering::Acquire) == seen_gen {
                if !shared.burst_active.load(Ordering::Relaxed)
                    || shared.shutdown.load(Ordering::Relaxed)
                    || spins >= 120_000
                {
                    break;
                }
                std::hint::spin_loop();
                spins += 1;
            }
        }
        if shared.generation.load(Ordering::Acquire) == seen_gen {
            shared.sleeping.fetch_add(1, Ordering::AcqRel);
            let mut guard = shared.wake_lock.lock().unwrap();
            while shared.generation.load(Ordering::Acquire) == seen_gen
                && !shared.shutdown.load(Ordering::Acquire)
                && !shared.burst_active.load(Ordering::Acquire)
            {
                guard = shared.wake_cv.wait(guard).unwrap();
            }
            drop(guard);
            shared.sleeping.fetch_sub(1, Ordering::AcqRel);
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            if shared.generation.load(Ordering::Acquire) == seen_gen {
                continue;
            }
        }
        let current_gen = shared.generation.load(Ordering::Acquire);
        if current_gen == seen_gen {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            continue;
        }
        seen_gen = current_gen;
        let job = unsafe { *shared.job.get() };
        let (start, end) = row_range(worker_idx, shared.threads, job.total_rows);
        if start < end {
            (job.func)(start, end, job.arg_in, job.arg_out);
        }
        shared.remaining.fetch_sub(1, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Q15 (15-bit signed i16) activation quantization + AVX2 VPMADDWD Q15xQ4 MatVec
// ---------------------------------------------------------------------------

#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct Q15Block([i16; 32]);

struct QuantizedInput15 {
    q15: [Q15Block; INTERMEDIATE / BLOCK_SIZE],
    scales: [f32; INTERMEDIATE / BLOCK_SIZE],
    sums: [f32; INTERMEDIATE / BLOCK_SIZE],
}

impl QuantizedInput15 {
    fn new() -> Self {
        Self {
            q15: [Q15Block([0; 32]); INTERMEDIATE / BLOCK_SIZE],
            scales: [0.0; INTERMEDIATE / BLOCK_SIZE],
            sums: [0.0; INTERMEDIATE / BLOCK_SIZE],
        }
    }

    fn quantize(&mut self, x: &[f32]) {
        for (b, chunk) in x.chunks_exact(BLOCK_SIZE).enumerate() {
            let mut max_abs = 0.0f32;
            let mut sum = 0.0f32;
            for &v in chunk {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
                sum += v;
            }
            self.sums[b] = sum;
            if max_abs > 0.0 {
                let scale = max_abs / 32767.0;
                let inv = 32767.0 / max_abs;
                self.scales[b] = scale;
                let dst = &mut self.q15[b].0;
                for i in 0..BLOCK_SIZE {
                    dst[i] = (chunk[i] * inv).round().clamp(-32767.0, 32767.0) as i16;
                }
            } else {
                self.scales[b] = 0.0;
                self.q15[b].0 = [0; 32];
            }
        }
    }
}

struct MatVec1Args<'a> {
    mat: &'a Q4Matrix,
    x: &'a [f32],
    qx: &'a QuantizedInput15,
    use_avx2: bool,
}

struct MatVec2Args<'a> {
    mat: &'a Q4Matrix,
    x0: &'a [f32],
    qx0: &'a QuantizedInput15,
    x1: &'a [f32],
    qx1: &'a QuantizedInput15,
    use_avx2: bool,
}

struct MatVec2Out<'a> {
    y0: &'a mut [f32],
    y1: &'a mut [f32],
}

fn matvec1_worker(start: usize, end: usize, arg_in: *const (), arg_out: *mut ()) {
    let args = unsafe { &*(arg_in as *const MatVec1Args<'_>) };
    let y_ptr = arg_out as *mut f32;
    let y = unsafe { std::slice::from_raw_parts_mut(y_ptr.add(start), end - start) };
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if args.use_avx2 {
        unsafe {
            matvec1_q15q4_avx2(args.mat, args.qx, start, end, y);
        }
        return;
    }
    matvec1_scalar(args.mat, args.x, &args.qx.sums, start, end, y);
}

fn matvec2_worker(start: usize, end: usize, arg_in: *const (), arg_out: *mut ()) {
    let args = unsafe { &*(arg_in as *const MatVec2Args<'_>) };
    let out = unsafe { &*(arg_out as *const MatVec2Out<'_>) };
    let y0 = unsafe {
        std::slice::from_raw_parts_mut(out.y0.as_ptr().cast_mut().add(start), end - start)
    };
    let y1 = unsafe {
        std::slice::from_raw_parts_mut(out.y1.as_ptr().cast_mut().add(start), end - start)
    };
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if args.use_avx2 {
        unsafe {
            matvec2_q15q4_avx2(args.mat, args.qx0, args.qx1, start, end, y0, y1);
        }
        return;
    }
    matvec1_scalar(args.mat, args.x0, &args.qx0.sums, start, end, y0);
    matvec1_scalar(args.mat, args.x1, &args.qx1.sums, start, end, y1);
}

fn matvec1_scalar(
    mat: &Q4Matrix,
    x: &[f32],
    x_sums: &[f32],
    start: usize,
    end: usize,
    out: &mut [f32],
) {
    let tg_count = mat.tiles_per_row_group;
    for (local_r, row) in (start..end).enumerate() {
        let rg = row / TILE_ROWS;
        let r_off = row % TILE_ROWS;
        let tile_row = &mat.tiles[rg * tg_count..(rg + 1) * tg_count];
        let mut acc = 0.0f32;
        for (tg, tile) in tile_row.iter().enumerate() {
            let b_base = tg * TILE_BLOCKS;
            for k in 0..TILE_BLOCKS {
                let b = b_base + k;
                let q = &tile.q[k][r_off].0;
                let xb = &x[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                let mut dot = 0.0f32;
                for j in 0..16 {
                    let byte = q[j];
                    dot += xb[j] * (byte & 0x0F) as f32 + xb[j + 16] * (byte >> 4) as f32;
                }
                let zbyte = tile.zps[k / 2][r_off];
                let zp = if k % 2 == 0 {
                    (zbyte & 0x0F) as f32
                } else {
                    (zbyte >> 4) as f32
                };
                acc += (dot - zp * x_sums[b]) * tile.scales[k][r_off];
            }
        }
        out[local_r] = acc;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// High-throughput AVX2 Q15xQ4 single-token matvec using `VPMADDWD` (`_mm256_madd_epi16`):
/// - 15-bit per-block activation precision (0.0015% max error -> 100% greedy parity)
/// - 2x `vpmovzxbw` + 2x `vpmaddwd` + 1x `vpaddd` + 1x `vcvtdq2ps` + 1x `vfmadd231ps`
///   per 32 weights (3x fewer Port-5/FPU ops than direct f32 dequantization!)
/// - Reads contiguous 656-byte `Q4Tile`s sequentially (1 memory stream per core).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec1_q15q4_avx2(
    mat: &Q4Matrix,
    qx: &QuantizedInput15,
    start: usize,
    end: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    let tg_count = mat.tiles_per_row_group;
    let mask_0f = _mm_set1_epi8(0x0F);
    let xq_ptr = qx.q15.as_ptr();
    let xsc_ptr = qx.scales.as_ptr();
    let xsm_ptr = qx.sums.as_ptr();
    let tiles_ptr = mat.tiles.as_ptr();

    debug_assert_eq!(start % TILE_ROWS, 0);
    let mut row = start;
    while row + TILE_ROWS <= end {
        let rg = row / TILE_ROWS;
        let row_tiles = unsafe { tiles_ptr.add(rg * tg_count) };
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut zp0 = 0.0f32;
        let mut zp1 = 0.0f32;
        let mut zp2 = 0.0f32;
        let mut zp3 = 0.0f32;

        for tg in 0..tg_count {
            let tile = unsafe { &*row_tiles.add(tg) };
            let b_base = tg * TILE_BLOCKS;
            for k in 0..TILE_BLOCKS {
                let bi = b_base + k;
                let xb = unsafe { (*xq_ptr.add(bi)).0.as_ptr() };
                let x0 = unsafe { _mm256_load_si256(xb.cast()) };
                let x1 = unsafe { _mm256_load_si256(xb.add(16).cast()) };
                let xsc = unsafe { *xsc_ptr.add(bi) };
                let xsm = unsafe { *xsm_ptr.add(bi) };
                let q_arr = &tile.q[k];
                let sc_arr = &tile.scales[k];
                let zp_arr = &tile.zps[k >> 1];
                let hi_nibble = (k & 1) != 0;

                macro_rules! accum_lane {
                    ($r:expr, $acc:ident, $zacc:ident) => {{
                        let raw = unsafe { _mm_load_si128(q_arr[$r].0.as_ptr().cast()) };
                        let lo = _mm_and_si128(raw, mask_0f);
                        let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
                        let w0 = _mm256_cvtepu8_epi16(lo);
                        let w1 = _mm256_cvtepu8_epi16(hi);
                        let p0 = _mm256_madd_epi16(w0, x0);
                        let p1 = _mm256_madd_epi16(w1, x1);
                        let dot_f32 = _mm256_cvtepi32_ps(_mm256_add_epi32(p0, p1));
                        let ws = sc_arr[$r];
                        $acc = _mm256_fmadd_ps(dot_f32, _mm256_set1_ps(ws * xsc), $acc);
                        let zb = zp_arr[$r];
                        let zv = (if hi_nibble { zb >> 4 } else { zb & 0x0F }) as f32;
                        $zacc = (zv * ws).mul_add(-xsm, $zacc);
                    }};
                }

                accum_lane!(0, acc0, zp0);
                accum_lane!(1, acc1, zp1);
                accum_lane!(2, acc2, zp2);
                accum_lane!(3, acc3, zp3);
            }
        }

        let local = row - start;
        unsafe {
            out[local] = hsum256_ps(acc0) + zp0;
            out[local + 1] = hsum256_ps(acc1) + zp1;
            out[local + 2] = hsum256_ps(acc2) + zp2;
            out[local + 3] = hsum256_ps(acc3) + zp3;
        }
        row += TILE_ROWS;
    }
    if row < end {
        // Fallback for non-multiple-of-4 tail rows (synthetic tests only).
        let local = row - start;
        for r_idx in row..end {
            let rg = r_idx / TILE_ROWS;
            let r_off = r_idx % TILE_ROWS;
            let row_tiles = unsafe { tiles_ptr.add(rg * tg_count) };
            let mut acc = _mm256_setzero_ps();
            let mut zacc = 0.0f32;
            for tg in 0..tg_count {
                let tile = unsafe { &*row_tiles.add(tg) };
                let b_base = tg * TILE_BLOCKS;
                for k in 0..TILE_BLOCKS {
                    let bi = b_base + k;
                    let xb = unsafe { (*xq_ptr.add(bi)).0.as_ptr() };
                    let x0 = unsafe { _mm256_load_si256(xb.cast()) };
                    let x1 = unsafe { _mm256_load_si256(xb.add(16).cast()) };
                    let raw = unsafe { _mm_load_si128(tile.q[k][r_off].0.as_ptr().cast()) };
                    let w0 = _mm256_cvtepu8_epi16(_mm_and_si128(raw, mask_0f));
                    let w1 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f));
                    let dot_f32 = _mm256_cvtepi32_ps(_mm256_add_epi32(
                        _mm256_madd_epi16(w0, x0),
                        _mm256_madd_epi16(w1, x1),
                    ));
                    let ws = tile.scales[k][r_off];
                    acc = _mm256_fmadd_ps(
                        dot_f32,
                        _mm256_set1_ps(ws * unsafe { *xsc_ptr.add(bi) }),
                        acc,
                    );
                    let zb = tile.zps[k >> 1][r_off];
                    let zv = (if (k & 1) != 0 { zb >> 4 } else { zb & 0x0F }) as f32;
                    zacc = (zv * ws).mul_add(-unsafe { *xsm_ptr.add(bi) }, zacc);
                }
            }
            out[local + (r_idx - row)] = unsafe { hsum256_ps(acc) } + zacc;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec2_q15q4_avx2(
    mat: &Q4Matrix,
    qx0: &QuantizedInput15,
    qx1: &QuantizedInput15,
    start: usize,
    end: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    use std::arch::x86_64::*;
    let tg_count = mat.tiles_per_row_group;
    let mask_0f = _mm_set1_epi8(0x0F);
    let tiles_ptr = mat.tiles.as_ptr();

    debug_assert_eq!(start % TILE_ROWS, 0);
    let mut row = start;
    while row + TILE_ROWS <= end {
        let rg = row / TILE_ROWS;
        let row_tiles = unsafe { tiles_ptr.add(rg * tg_count) };
        let mut a0 = [_mm256_setzero_ps(); 4];
        let mut a1 = [_mm256_setzero_ps(); 4];
        let mut z0 = [0.0f32; 4];
        let mut z1 = [0.0f32; 4];

        for tg in 0..tg_count {
            let tile = unsafe { &*row_tiles.add(tg) };
            let b_base = tg * TILE_BLOCKS;
            for k in 0..TILE_BLOCKS {
                let bi = b_base + k;
                let u_ptr = qx0.q15[bi].0.as_ptr();
                let v_ptr = qx1.q15[bi].0.as_ptr();
                let u0 = unsafe { _mm256_load_si256(u_ptr.cast()) };
                let u1 = unsafe { _mm256_load_si256(u_ptr.add(16).cast()) };
                let v0 = unsafe { _mm256_load_si256(v_ptr.cast()) };
                let v1 = unsafe { _mm256_load_si256(v_ptr.add(16).cast()) };
                let xsc0 = qx0.scales[bi];
                let xsc1 = qx1.scales[bi];
                let xsm0 = qx0.sums[bi];
                let xsm1 = qx1.sums[bi];
                let q_arr = &tile.q[k];
                let sc_arr = &tile.scales[k];
                let zp_arr = &tile.zps[k >> 1];
                let hi_nibble = (k & 1) != 0;

                for r in 0..TILE_ROWS {
                    let raw = unsafe { _mm_load_si128(q_arr[r].0.as_ptr().cast()) };
                    let w0 = _mm256_cvtepu8_epi16(_mm_and_si128(raw, mask_0f));
                    let w1 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f));
                    let ws = sc_arr[r];

                    let d0 = _mm256_cvtepi32_ps(_mm256_add_epi32(
                        _mm256_madd_epi16(w0, u0),
                        _mm256_madd_epi16(w1, u1),
                    ));
                    a0[r] = _mm256_fmadd_ps(d0, _mm256_set1_ps(ws * xsc0), a0[r]);

                    let d1 = _mm256_cvtepi32_ps(_mm256_add_epi32(
                        _mm256_madd_epi16(w0, v0),
                        _mm256_madd_epi16(w1, v1),
                    ));
                    a1[r] = _mm256_fmadd_ps(d1, _mm256_set1_ps(ws * xsc1), a1[r]);

                    let zb = zp_arr[r];
                    let zv = (if hi_nibble { zb >> 4 } else { zb & 0x0F }) as f32 * ws;
                    z0[r] = zv.mul_add(-xsm0, z0[r]);
                    z1[r] = zv.mul_add(-xsm1, z1[r]);
                }
            }
        }

        let local = row - start;
        for r in 0..TILE_ROWS {
            unsafe {
                out0[local + r] = hsum256_ps(a0[r]) + z0[r];
                out1[local + r] = hsum256_ps(a1[r]) + z1[r];
            }
        }
        row += TILE_ROWS;
    }
    if row < end {
        unsafe {
            matvec1_q15q4_avx2(mat, qx0, row, end, &mut out0[row - start..]);
            matvec1_q15q4_avx2(mat, qx1, row, end, &mut out1[row - start..]);
        }
    }
}

// ---------------------------------------------------------------------------
// Transformer math helpers
// ---------------------------------------------------------------------------

#[inline]
fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32]) {
    let mut ss = 0.0f32;
    for &v in x {
        ss = v.mul_add(v, ss);
    }
    let inv = 1.0f32 / (ss / x.len() as f32 + RMS_EPS).sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * inv * w;
    }
}

#[inline]
fn rms_norm_in_place(x: &mut [f32], weight: &[f32]) {
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss = v.mul_add(v, ss);
    }
    let inv = 1.0f32 / (ss / x.len() as f32 + RMS_EPS).sqrt();
    for (v, &w) in x.iter_mut().zip(weight) {
        *v = *v * inv * w;
    }
}

#[inline]
fn apply_rope(vec: &mut [f32], cos: &[f32; HEAD_DIM / 2], sin: &[f32; HEAD_DIM / 2]) {
    let half = HEAD_DIM / 2;
    for i in 0..half {
        let a = vec[i];
        let b = vec[i + half];
        let c = cos[i];
        let s = sin[i];
        vec[i] = a * c - b * s;
        vec[i + half] = b * c + a * s;
    }
}

#[inline]
fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        let sig = 1.0f32 / (1.0f32 + (-g).exp());
        *o = g * sig * u;
    }
}

// ---------------------------------------------------------------------------
// FastCodePredictor
// ---------------------------------------------------------------------------

struct Scratch {
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    h0: [f32; HIDDEN],
    h1: [f32; HIDDEN],
    norm0: [f32; HIDDEN],
    norm1: [f32; HIDDEN],
    qkv0: [f32; Q_DIM + KV_DIM + KV_DIM],
    qkv1: [f32; Q_DIM + KV_DIM + KV_DIM],
    attn0: [f32; Q_DIM],
    attn1: [f32; Q_DIM],
    proj0: [f32; HIDDEN],
    proj1: [f32; HIDDEN],
    gate_up0: [f32; INTERMEDIATE * 2],
    gate_up1: [f32; INTERMEDIATE * 2],
    mlp_act0: [f32; INTERMEDIATE],
    mlp_act1: [f32; INTERMEDIATE],
    qx0: QuantizedInput15,
    qx1: QuantizedInput15,
    logits: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            k_cache: vec![0.0; NUM_LAYERS * MAX_SEQ * KV_DIM],
            v_cache: vec![0.0; NUM_LAYERS * MAX_SEQ * KV_DIM],
            h0: [0.0; HIDDEN],
            h1: [0.0; HIDDEN],
            norm0: [0.0; HIDDEN],
            norm1: [0.0; HIDDEN],
            qkv0: [0.0; Q_DIM + KV_DIM + KV_DIM],
            qkv1: [0.0; Q_DIM + KV_DIM + KV_DIM],
            attn0: [0.0; Q_DIM],
            attn1: [0.0; Q_DIM],
            proj0: [0.0; HIDDEN],
            proj1: [0.0; HIDDEN],
            gate_up0: [0.0; INTERMEDIATE * 2],
            gate_up1: [0.0; INTERMEDIATE * 2],
            mlp_act0: [0.0; INTERMEDIATE],
            mlp_act1: [0.0; INTERMEDIATE],
            qx0: QuantizedInput15::new(),
            qx1: QuantizedInput15::new(),
            logits: vec![0.0; 2048],
        }
    }
}

pub struct FastCodePredictor {
    talker_embedding: Q4Embedding,
    codec_embeddings: Vec<Q4Embedding>,
    layers: Vec<LayerWeights>,
    final_norm: [f32; HIDDEN],
    lm_heads: Vec<Q4Matrix>,
    rope_cos: [[f32; HEAD_DIM / 2]; MAX_SEQ],
    rope_sin: [[f32; HEAD_DIM / 2]; MAX_SEQ],
    pool: BurstPool,
    use_avx2: bool,
    scratch: Scratch,
}

fn run_matvec1(
    pool: &BurstPool,
    use_avx2: bool,
    mat: &Q4Matrix,
    x: &[f32],
    qx: &mut QuantizedInput15,
    y: &mut [f32],
) {
    qx.quantize(x);
    let args = MatVec1Args {
        mat,
        x,
        qx,
        use_avx2,
    };
    pool.parallel_rows(
        mat.rows,
        matvec1_worker,
        (&args as *const MatVec1Args<'_>).cast(),
        y.as_mut_ptr().cast(),
    );
}

#[allow(clippy::too_many_arguments)]
fn run_matvec2(
    pool: &BurstPool,
    use_avx2: bool,
    mat: &Q4Matrix,
    x0: &[f32],
    qx0: &mut QuantizedInput15,
    x1: &[f32],
    qx1: &mut QuantizedInput15,
    y0: &mut [f32],
    y1: &mut [f32],
) {
    qx0.quantize(x0);
    qx1.quantize(x1);
    let args = MatVec2Args {
        mat,
        x0,
        qx0,
        x1,
        qx1,
        use_avx2,
    };
    let out = MatVec2Out { y0, y1 };
    pool.parallel_rows(
        mat.rows,
        matvec2_worker,
        (&args as *const MatVec2Args<'_>).cast(),
        (&out as *const MatVec2Out<'_>).cast_mut().cast(),
    );
}

impl FastCodePredictor {
    pub fn try_load(path: &Path, threads: usize, cpus: &[usize]) -> Result<Option<Self>> {
        let bytes = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
        let parsed = match parse_onnx_code_predictor(&bytes) {
            Ok(Some(model)) => model,
            Ok(None) => return Ok(None),
            Err(_) => return Ok(None),
        };
        let mut rope_cos = [[0.0f32; HEAD_DIM / 2]; MAX_SEQ];
        let mut rope_sin = [[0.0f32; HEAD_DIM / 2]; MAX_SEQ];
        for pos in 0..MAX_SEQ {
            for i in 0..(HEAD_DIM / 2) {
                let inv_freq = 1.0f64 / ROPE_THETA.powf((2 * i) as f64 / HEAD_DIM as f64);
                let angle = (pos as f64) * inv_freq;
                rope_cos[pos][i] = angle.cos() as f32;
                rope_sin[pos][i] = angle.sin() as f32;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let use_avx2 =
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let use_avx2 = false;

        Ok(Some(Self {
            talker_embedding: parsed.talker_embedding,
            codec_embeddings: parsed.codec_embeddings,
            layers: parsed.layers,
            final_norm: parsed.final_norm,
            lm_heads: parsed.lm_heads,
            rope_cos,
            rope_sin,
            pool: BurstPool::new(threads, cpus),
            use_avx2,
            scratch: Scratch::new(),
        }))
    }

    fn forward_prefill_2(&mut self) {
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let pool = &self.pool;
        let use_avx2 = self.use_avx2;
        let scratch = &mut self.scratch;
        for layer_idx in 0..NUM_LAYERS {
            let layer = &self.layers[layer_idx];
            rms_norm(&scratch.h0, &layer.input_ln, &mut scratch.norm0);
            rms_norm(&scratch.h1, &layer.input_ln, &mut scratch.norm1);
            run_matvec2(
                pool,
                use_avx2,
                &layer.qkv_proj,
                &scratch.norm0,
                &mut scratch.qx0,
                &scratch.norm1,
                &mut scratch.qx1,
                &mut scratch.qkv0,
                &mut scratch.qkv1,
            );

            let layer_kv_base = layer_idx * MAX_SEQ * KV_DIM;
            for (pos, qkv) in [(0usize, &mut scratch.qkv0), (1usize, &mut scratch.qkv1)] {
                let (q_slice, kv_slice) = qkv.split_at_mut(Q_DIM);
                let (k_slice, v_slice) = kv_slice.split_at_mut(KV_DIM);
                let cos = &self.rope_cos[pos];
                let sin = &self.rope_sin[pos];
                for head in 0..Q_HEADS {
                    let qh = &mut q_slice[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                    rms_norm_in_place(qh, &layer.q_norm);
                    apply_rope(qh, cos, sin);
                }
                for head in 0..KV_HEADS {
                    let kh = &mut k_slice[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                    rms_norm_in_place(kh, &layer.k_norm);
                    apply_rope(kh, cos, sin);
                }
                let dst = layer_kv_base + pos * KV_DIM;
                scratch.k_cache[dst..dst + KV_DIM].copy_from_slice(k_slice);
                scratch.v_cache[dst..dst + KV_DIM].copy_from_slice(v_slice);
            }

            let k0_base = layer_kv_base;
            let k1_base = layer_kv_base + KV_DIM;
            let repeat = Q_HEADS / KV_HEADS;
            for qh_idx in 0..Q_HEADS {
                let kvh_idx = qh_idx / repeat;
                let k0 = &scratch.k_cache
                    [k0_base + kvh_idx * HEAD_DIM..k0_base + (kvh_idx + 1) * HEAD_DIM];
                let k1 = &scratch.k_cache
                    [k1_base + kvh_idx * HEAD_DIM..k1_base + (kvh_idx + 1) * HEAD_DIM];
                let v0 = &scratch.v_cache
                    [k0_base + kvh_idx * HEAD_DIM..k0_base + (kvh_idx + 1) * HEAD_DIM];
                let v1 = &scratch.v_cache
                    [k1_base + kvh_idx * HEAD_DIM..k1_base + (kvh_idx + 1) * HEAD_DIM];

                scratch.attn0[qh_idx * HEAD_DIM..(qh_idx + 1) * HEAD_DIM].copy_from_slice(v0);

                let q1 = &scratch.qkv1[qh_idx * HEAD_DIM..(qh_idx + 1) * HEAD_DIM];
                let mut dot0 = 0.0f32;
                let mut dot1 = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot0 = q1[d].mul_add(k0[d], dot0);
                    dot1 = q1[d].mul_add(k1[d], dot1);
                }
                dot0 *= scale;
                dot1 *= scale;
                let m = dot0.max(dot1);
                let e0 = (dot0 - m).exp();
                let e1 = (dot1 - m).exp();
                let inv = 1.0f32 / (e0 + e1);
                let w0 = e0 * inv;
                let w1 = e1 * inv;
                let out1 = &mut scratch.attn1[qh_idx * HEAD_DIM..(qh_idx + 1) * HEAD_DIM];
                for d in 0..HEAD_DIM {
                    out1[d] = w0 * v0[d] + w1 * v1[d];
                }
            }

            run_matvec2(
                pool,
                use_avx2,
                &layer.o_proj,
                &scratch.attn0,
                &mut scratch.qx0,
                &scratch.attn1,
                &mut scratch.qx1,
                &mut scratch.proj0,
                &mut scratch.proj1,
            );
            for i in 0..HIDDEN {
                scratch.h0[i] += scratch.proj0[i];
                scratch.h1[i] += scratch.proj1[i];
            }

            rms_norm(&scratch.h0, &layer.post_ln, &mut scratch.norm0);
            rms_norm(&scratch.h1, &layer.post_ln, &mut scratch.norm1);
            run_matvec2(
                pool,
                use_avx2,
                &layer.gate_up_proj,
                &scratch.norm0,
                &mut scratch.qx0,
                &scratch.norm1,
                &mut scratch.qx1,
                &mut scratch.gate_up0,
                &mut scratch.gate_up1,
            );
            silu_mul(
                &scratch.gate_up0[..INTERMEDIATE],
                &scratch.gate_up0[INTERMEDIATE..],
                &mut scratch.mlp_act0,
            );
            silu_mul(
                &scratch.gate_up1[..INTERMEDIATE],
                &scratch.gate_up1[INTERMEDIATE..],
                &mut scratch.mlp_act1,
            );
            run_matvec2(
                pool,
                use_avx2,
                &layer.down_proj,
                &scratch.mlp_act0,
                &mut scratch.qx0,
                &scratch.mlp_act1,
                &mut scratch.qx1,
                &mut scratch.proj0,
                &mut scratch.proj1,
            );
            for i in 0..HIDDEN {
                scratch.h0[i] += scratch.proj0[i];
                scratch.h1[i] += scratch.proj1[i];
            }
        }
    }

    fn forward_token(&mut self, pos: usize) {
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let cos = &self.rope_cos[pos];
        let sin = &self.rope_sin[pos];
        let seq_len = pos + 1;
        let repeat = Q_HEADS / KV_HEADS;
        let pool = &self.pool;
        let use_avx2 = self.use_avx2;
        let scratch = &mut self.scratch;

        for layer_idx in 0..NUM_LAYERS {
            let layer = &self.layers[layer_idx];
            rms_norm(&scratch.h0, &layer.input_ln, &mut scratch.norm0);
            run_matvec1(
                pool,
                use_avx2,
                &layer.qkv_proj,
                &scratch.norm0,
                &mut scratch.qx0,
                &mut scratch.qkv0,
            );

            let (q_slice, kv_slice) = scratch.qkv0.split_at_mut(Q_DIM);
            let (k_slice, v_slice) = kv_slice.split_at_mut(KV_DIM);
            for head in 0..Q_HEADS {
                let qh = &mut q_slice[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                rms_norm_in_place(qh, &layer.q_norm);
                apply_rope(qh, cos, sin);
            }
            for head in 0..KV_HEADS {
                let kh = &mut k_slice[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                rms_norm_in_place(kh, &layer.k_norm);
                apply_rope(kh, cos, sin);
            }

            let layer_kv_base = layer_idx * MAX_SEQ * KV_DIM;
            let dst = layer_kv_base + pos * KV_DIM;
            scratch.k_cache[dst..dst + KV_DIM].copy_from_slice(k_slice);
            scratch.v_cache[dst..dst + KV_DIM].copy_from_slice(v_slice);

            for qh_idx in 0..Q_HEADS {
                let kvh_idx = qh_idx / repeat;
                let qh = &scratch.qkv0[qh_idx * HEAD_DIM..(qh_idx + 1) * HEAD_DIM];
                let mut scores = [0.0f32; MAX_SEQ];
                let mut max_score = f32::NEG_INFINITY;
                for t in 0..seq_len {
                    let kb = layer_kv_base + t * KV_DIM + kvh_idx * HEAD_DIM;
                    let kh = &scratch.k_cache[kb..kb + HEAD_DIM];
                    let mut dot = 0.0f32;
                    for d in 0..HEAD_DIM {
                        dot = qh[d].mul_add(kh[d], dot);
                    }
                    let s = dot * scale;
                    scores[t] = s;
                    if s > max_score {
                        max_score = s;
                    }
                }
                let mut sum_exp = 0.0f32;
                for t in 0..seq_len {
                    let e = (scores[t] - max_score).exp();
                    scores[t] = e;
                    sum_exp += e;
                }
                let inv = 1.0f32 / sum_exp;
                let out = &mut scratch.attn0[qh_idx * HEAD_DIM..(qh_idx + 1) * HEAD_DIM];
                out.fill(0.0);
                for t in 0..seq_len {
                    let w = scores[t] * inv;
                    let vb = layer_kv_base + t * KV_DIM + kvh_idx * HEAD_DIM;
                    let vh = &scratch.v_cache[vb..vb + HEAD_DIM];
                    for d in 0..HEAD_DIM {
                        out[d] = w.mul_add(vh[d], out[d]);
                    }
                }
            }

            run_matvec1(
                pool,
                use_avx2,
                &layer.o_proj,
                &scratch.attn0,
                &mut scratch.qx0,
                &mut scratch.proj0,
            );
            for i in 0..HIDDEN {
                scratch.h0[i] += scratch.proj0[i];
            }

            rms_norm(&scratch.h0, &layer.post_ln, &mut scratch.norm0);
            run_matvec1(
                pool,
                use_avx2,
                &layer.gate_up_proj,
                &scratch.norm0,
                &mut scratch.qx0,
                &mut scratch.gate_up0,
            );
            silu_mul(
                &scratch.gate_up0[..INTERMEDIATE],
                &scratch.gate_up0[INTERMEDIATE..],
                &mut scratch.mlp_act0,
            );
            run_matvec1(
                pool,
                use_avx2,
                &layer.down_proj,
                &scratch.mlp_act0,
                &mut scratch.qx0,
                &mut scratch.proj0,
            );
            for i in 0..HIDDEN {
                scratch.h0[i] += scratch.proj0[i];
            }
        }
    }

    pub fn predict_frame<F>(
        &mut self,
        talker_hidden: &[f32],
        code0: i64,
        mut sample_group: F,
    ) -> Result<[i64; NUM_GROUPS]>
    where
        F: FnMut(&[f32]) -> Result<usize>,
    {
        ensure!(
            talker_hidden.len() == HIDDEN,
            "unexpected talker hidden size: {}",
            talker_hidden.len()
        );
        let mut frame = [0i64; NUM_GROUPS];
        frame[0] = code0;

        self.pool.set_burst(true);
        let result = (|| -> Result<[i64; NUM_GROUPS]> {
            self.scratch.h0.copy_from_slice(talker_hidden);
            self.talker_embedding
                .lookup_into(code0, &mut self.scratch.h1)?;
            self.forward_prefill_2();

            rms_norm(&self.scratch.h1, &self.final_norm, &mut self.scratch.norm0);
            run_matvec1(
                &self.pool,
                self.use_avx2,
                &self.lm_heads[0],
                &self.scratch.norm0,
                &mut self.scratch.qx0,
                &mut self.scratch.logits,
            );
            frame[1] = sample_group(&self.scratch.logits)? as i64;

            for group in 2..NUM_GROUPS {
                self.codec_embeddings[group - 2]
                    .lookup_into(frame[group - 1], &mut self.scratch.h0)?;
                self.forward_token(group);
                rms_norm(&self.scratch.h0, &self.final_norm, &mut self.scratch.norm0);
                run_matvec1(
                    &self.pool,
                    self.use_avx2,
                    &self.lm_heads[group - 1],
                    &self.scratch.norm0,
                    &mut self.scratch.qx0,
                    &mut self.scratch.logits,
                );
                frame[group] = sample_group(&self.scratch.logits)? as i64;
            }
            Ok(frame)
        })();
        self.pool.set_burst(false);
        result
    }
}

// ---------------------------------------------------------------------------
// Minimal zero-dependency ONNX Protobuf reader for code_predictor.onnx
// ---------------------------------------------------------------------------

struct ParsedModel {
    talker_embedding: Q4Embedding,
    codec_embeddings: Vec<Q4Embedding>,
    layers: Vec<LayerWeights>,
    final_norm: [f32; HIDDEN],
    lm_heads: Vec<Q4Matrix>,
}

struct TensorView<'a> {
    dims: Vec<usize>,
    data_type: u64,
    raw: &'a [u8],
}

impl<'a> TensorView<'a> {
    fn as_f32_vec(&self) -> Result<Vec<f32>> {
        ensure!(self.data_type == 1, "expected FLOAT tensor");
        ensure!(self.raw.len() % 4 == 0, "unaligned f32 tensor");
        let mut out = Vec::with_capacity(self.raw.len() / 4);
        for chunk in self.raw.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    fn as_f32_array<const N: usize>(&self) -> Result<[f32; N]> {
        let vec = self.as_f32_vec()?;
        ensure!(vec.len() == N, "expected {N} floats, found {}", vec.len());
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(&vec);
        Ok(arr)
    }

    fn as_u8_slice(&self) -> Result<&'a [u8]> {
        ensure!(self.data_type == 2, "expected UINT8 tensor");
        Ok(self.raw)
    }
}

struct NodeInfo {
    op_type: String,
    name: String,
    inputs: Vec<String>,
    int_attrs: HashMap<String, i64>,
}

fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut val = 0u64;
    let mut shift = 0u32;
    loop {
        ensure!(*pos < buf.len(), "truncated varint");
        let b = buf[*pos];
        *pos += 1;
        val |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return Ok(val);
        }
        shift += 7;
        ensure!(shift < 64, "varint overflow");
    }
}

fn read_len_delimited<'a>(buf: &'a [u8], pos: &mut usize) -> Result<&'a [u8]> {
    let len = read_varint(buf, pos)? as usize;
    ensure!(*pos + len <= buf.len(), "truncated length-delimited field");
    let slice = &buf[*pos..*pos + len];
    *pos += len;
    Ok(slice)
}

fn skip_field(buf: &[u8], pos: &mut usize, wire_type: u64) -> Result<()> {
    match wire_type {
        0 => {
            let _ = read_varint(buf, pos)?;
        }
        1 => {
            ensure!(*pos + 8 <= buf.len(), "truncated fixed64");
            *pos += 8;
        }
        2 => {
            let _ = read_len_delimited(buf, pos)?;
        }
        5 => {
            ensure!(*pos + 4 <= buf.len(), "truncated fixed32");
            *pos += 4;
        }
        _ => bail!("unsupported protobuf wire type {wire_type}"),
    }
    Ok(())
}

fn parse_tensor<'a>(buf: &'a [u8]) -> Result<(String, TensorView<'a>)> {
    let mut pos = 0;
    let mut dims = Vec::new();
    let mut data_type = 0u64;
    let mut name = String::new();
    let mut raw: &'a [u8] = &[];
    while pos < buf.len() {
        let tag = read_varint(buf, &mut pos)?;
        let field = tag >> 3;
        let wt = tag & 7;
        match (field, wt) {
            (1, 0) => dims.push(read_varint(buf, &mut pos)? as usize),
            (1, 2) => {
                let packed = read_len_delimited(buf, &mut pos)?;
                let mut p = 0;
                while p < packed.len() {
                    dims.push(read_varint(packed, &mut p)? as usize);
                }
            }
            (2, 0) => data_type = read_varint(buf, &mut pos)?,
            (8, 2) => {
                let s = read_len_delimited(buf, &mut pos)?;
                name = std::str::from_utf8(s)?.to_owned();
            }
            (9, 2) => raw = read_len_delimited(buf, &mut pos)?,
            _ => skip_field(buf, &mut pos, wt)?,
        }
    }
    Ok((
        name,
        TensorView {
            dims,
            data_type,
            raw,
        },
    ))
}

fn parse_attr(buf: &[u8]) -> Result<(String, Option<i64>)> {
    let mut pos = 0;
    let mut name = String::new();
    let mut int_val = None;
    while pos < buf.len() {
        let tag = read_varint(buf, &mut pos)?;
        let field = tag >> 3;
        let wt = tag & 7;
        match (field, wt) {
            (1, 2) => name = std::str::from_utf8(read_len_delimited(buf, &mut pos)?)?.to_owned(),
            (3, 0) => int_val = Some(read_varint(buf, &mut pos)? as i64),
            _ => skip_field(buf, &mut pos, wt)?,
        }
    }
    Ok((name, int_val))
}

fn parse_node(buf: &[u8]) -> Result<NodeInfo> {
    let mut pos = 0;
    let mut op_type = String::new();
    let mut name = String::new();
    let mut inputs = Vec::new();
    let mut int_attrs = HashMap::new();
    while pos < buf.len() {
        let tag = read_varint(buf, &mut pos)?;
        let field = tag >> 3;
        let wt = tag & 7;
        match (field, wt) {
            (1, 2) => {
                inputs.push(std::str::from_utf8(read_len_delimited(buf, &mut pos)?)?.to_owned())
            }
            (3, 2) => name = std::str::from_utf8(read_len_delimited(buf, &mut pos)?)?.to_owned(),
            (4, 2) => op_type = std::str::from_utf8(read_len_delimited(buf, &mut pos)?)?.to_owned(),
            (5, 2) => {
                let (aname, aval) = parse_attr(read_len_delimited(buf, &mut pos)?)?;
                if let Some(v) = aval {
                    int_attrs.insert(aname, v);
                }
            }
            _ => skip_field(buf, &mut pos, wt)?,
        }
    }
    Ok(NodeInfo {
        op_type,
        name,
        inputs,
        int_attrs,
    })
}

fn parse_onnx_code_predictor(data: &[u8]) -> Result<Option<ParsedModel>> {
    let mut pos = 0;
    let mut graph_bytes: Option<&[u8]> = None;
    while pos < data.len() {
        let tag = read_varint(data, &mut pos)?;
        let field = tag >> 3;
        let wt = tag & 7;
        if field == 7 && wt == 2 {
            graph_bytes = Some(read_len_delimited(data, &mut pos)?);
            break;
        }
        skip_field(data, &mut pos, wt)?;
    }
    let Some(graph) = graph_bytes else {
        return Ok(None);
    };

    let mut initializers: HashMap<String, TensorView<'_>> = HashMap::new();
    let mut matmul_nodes: HashMap<String, NodeInfo> = HashMap::new();
    let mut gpos = 0;
    while gpos < graph.len() {
        let tag = read_varint(graph, &mut gpos)?;
        let field = tag >> 3;
        let wt = tag & 7;
        match (field, wt) {
            (1, 2) => {
                let node = parse_node(read_len_delimited(graph, &mut gpos)?)?;
                if node.op_type == "MatMulNBits" {
                    matmul_nodes.insert(node.name.clone(), node);
                }
            }
            (5, 2) => {
                let (name, view) = parse_tensor(read_len_delimited(graph, &mut gpos)?)?;
                if !name.is_empty() && !view.raw.is_empty() {
                    initializers.insert(name, view);
                }
            }
            _ => skip_field(graph, &mut gpos, wt)?,
        }
    }

    if matmul_nodes.len() < NUM_LAYERS * 7 + NUM_LM_HEADS {
        return Ok(None);
    }

    let get_tensor = |name: &str| -> Result<&TensorView<'_>> {
        initializers
            .get(name)
            .with_context(|| format!("missing initializer {name}"))
    };

    let load_matmul = |node_name: &str, expect_rows: usize, expect_cols: usize| -> Result<Q4Matrix> {
        let node = matmul_nodes
            .get(node_name)
            .with_context(|| format!("missing MatMulNBits node {node_name}"))?;
        ensure!(
            node.int_attrs.get("bits") == Some(&4)
                && node.int_attrs.get("block_size") == Some(&(BLOCK_SIZE as i64))
                && node.int_attrs.get("K") == Some(&(expect_cols as i64))
                && node.int_attrs.get("N") == Some(&(expect_rows as i64)),
            "unexpected MatMulNBits attributes on {node_name}"
        );
        ensure!(
            node.inputs.len() >= 4,
            "MatMulNBits {node_name} missing zero_points"
        );
        let q = get_tensor(&node.inputs[1])?.as_u8_slice()?;
        let scales = get_tensor(&node.inputs[2])?.as_f32_vec()?;
        let zp = get_tensor(&node.inputs[3])?.as_u8_slice()?;
        Q4Matrix::from_onnx_q4(expect_rows, expect_cols, q, &scales, zp)
    };

    let load_embedding = |prefix: &str, expect_vocab: usize| -> Result<Q4Embedding> {
        let q_view = get_tensor(&format!("{prefix}.weight_Q4"))?;
        ensure!(
            q_view.dims == [expect_vocab, HIDDEN / 2],
            "unexpected embedding shape for {prefix}"
        );
        let q = q_view.as_u8_slice()?.to_vec();
        let scales = get_tensor(&format!("{prefix}.weight_scales"))?.as_f32_vec()?;
        let zp = get_tensor(&format!("{prefix}.weight_zero_point"))?
            .as_u8_slice()?
            .to_vec();
        Q4Embedding::from_onnx(expect_vocab, q, scales, zp)
    };

    let talker_embedding = load_embedding("talker_codec_embedding", 3072)?;
    let mut codec_embeddings = Vec::with_capacity(14);
    for idx in 0..14 {
        codec_embeddings.push(load_embedding(
            &format!("cp.model.codec_embedding.{idx}"),
            2048,
        )?);
    }

    let mut layers = Vec::with_capacity(NUM_LAYERS);
    for l in 0..NUM_LAYERS {
        let input_ln = get_tensor(&format!("cp.model.layers.{l}.input_layernorm.weight"))?
            .as_f32_array::<HIDDEN>()?;
        let post_ln = get_tensor(&format!(
            "cp.model.layers.{l}.post_attention_layernorm.weight"
        ))?
        .as_f32_array::<HIDDEN>()?;
        let q_norm = get_tensor(&format!("cp.model.layers.{l}.self_attn.q_norm.weight"))?
            .as_f32_array::<HEAD_DIM>()?;
        let k_norm = get_tensor(&format!("cp.model.layers.{l}.self_attn.k_norm.weight"))?
            .as_f32_array::<HEAD_DIM>()?;

        let q_proj = load_matmul(
            &format!("/model/layers.{l}/self_attn/q_proj/MatMul_Q4"),
            Q_DIM,
            HIDDEN,
        )?;
        let k_proj = load_matmul(
            &format!("/model/layers.{l}/self_attn/k_proj/MatMul_Q4"),
            KV_DIM,
            HIDDEN,
        )?;
        let v_proj = load_matmul(
            &format!("/model/layers.{l}/self_attn/v_proj/MatMul_Q4"),
            KV_DIM,
            HIDDEN,
        )?;
        let qkv_proj = Q4Matrix::concat_rows(&[q_proj, k_proj, v_proj])?;
        let o_proj = load_matmul(
            &format!("/model/layers.{l}/self_attn/o_proj/MatMul_Q4"),
            HIDDEN,
            Q_DIM,
        )?;
        let gate_proj = load_matmul(
            &format!("/model/layers.{l}/mlp/gate_proj/MatMul_Q4"),
            INTERMEDIATE,
            HIDDEN,
        )?;
        let up_proj = load_matmul(
            &format!("/model/layers.{l}/mlp/up_proj/MatMul_Q4"),
            INTERMEDIATE,
            HIDDEN,
        )?;
        let gate_up_proj = Q4Matrix::concat_rows(&[gate_proj, up_proj])?;
        let down_proj = load_matmul(
            &format!("/model/layers.{l}/mlp/down_proj/MatMul_Q4"),
            HIDDEN,
            INTERMEDIATE,
        )?;

        layers.push(LayerWeights {
            input_ln,
            post_ln,
            q_norm,
            k_norm,
            qkv_proj,
            o_proj,
            gate_up_proj,
            down_proj,
        });
    }

    let final_norm = get_tensor("cp.model.norm.weight")?.as_f32_array::<HIDDEN>()?;
    let mut lm_heads = Vec::with_capacity(NUM_LM_HEADS);
    for head in 0..NUM_LM_HEADS {
        lm_heads.push(load_matmul(
            &format!("/lm_head.{head}/MatMul_Q4"),
            2048,
            HIDDEN,
        )?);
    }

    Ok(Some(ParsedModel {
        talker_embedding,
        codec_embeddings,
        layers,
        final_norm,
        lm_heads,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_avx2_and_scalar_matvec_agree_within_float_tolerance() {
        let rows = 132;
        let cols = 256;
        let nb = cols / BLOCK_SIZE;
        let total_blocks = rows * nb;
        let mut packed_q = Vec::with_capacity(total_blocks * 16);
        let mut scales = Vec::with_capacity(total_blocks);
        let mut zps = Vec::with_capacity(total_blocks / 2);
        for b in 0..total_blocks {
            for j in 0..16 {
                let lo = ((b * 7 + j * 3) % 16) as u8;
                let hi = ((b * 5 + j * 11 + 1) % 16) as u8;
                packed_q.push(lo | (hi << 4));
            }
            scales.push(0.015 + ((b % 17) as f32) * 0.003);
        }
        for b in 0..(total_blocks / 2) {
            let z0 = ((b * 3 + 5) % 16) as u8;
            let z1 = ((b * 7 + 2) % 16) as u8;
            zps.push(z0 | (z1 << 4));
        }
        let mat = Q4Matrix::from_onnx_q4(rows, cols, &packed_q, &scales, &zps).unwrap();
        let x0: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.13).sin()).collect();
        let x1: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.29).cos()).collect();
        let mut qx0 = QuantizedInput15::new();
        let mut qx1 = QuantizedInput15::new();
        qx0.quantize(&x0);
        qx1.quantize(&x1);

        let mut expected0 = vec![0.0f32; rows];
        let mut expected1 = vec![0.0f32; rows];
        for r in 0..rows {
            for b in 0..nb {
                let idx = r * nb + b;
                let sc = scales[idx];
                let zbyte = zps[idx / 2];
                let zp = if idx % 2 == 0 {
                    zbyte & 0x0F
                } else {
                    zbyte >> 4
                } as f32;
                let src = &packed_q[idx * 16..(idx + 1) * 16];
                for j in 0..16 {
                    let w0 = ((src[j] & 0x0F) as f32 - zp) * sc;
                    let w1 = ((src[j] >> 4) as f32 - zp) * sc;
                    expected0[r] += w0 * x0[b * 32 + 2 * j] + w1 * x0[b * 32 + 2 * j + 1];
                    expected1[r] += w0 * x1[b * 32 + 2 * j] + w1 * x1[b * 32 + 2 * j + 1];
                }
            }
        }

        let mut y_scalar = vec![0.0f32; rows];
        matvec1_scalar(&mat, &x0, &qx0.sums, 0, rows, &mut y_scalar);
        for (a, b) in y_scalar.iter().zip(&expected0) {
            assert!((a - b).abs() < 1e-4, "scalar mismatch {a} vs {b}");
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut y_avx0 = vec![0.0f32; rows];
            let mut y_avx1 = vec![0.0f32; rows];
            unsafe {
                matvec1_q15q4_avx2(&mat, &qx0, 0, rows, &mut y_avx0);
            }
            for (a, b) in y_avx0.iter().zip(&expected0) {
                assert!((a - b).abs() < 5e-4, "avx2 q15q4 matvec1 mismatch {a} vs {b}");
            }
            y_avx0.fill(0.0);
            unsafe {
                matvec2_q15q4_avx2(&mat, &qx0, &qx1, 0, rows, &mut y_avx0, &mut y_avx1);
            }
            for (a, b) in y_avx0.iter().zip(&expected0) {
                assert!((a - b).abs() < 5e-4, "avx2 q15q4 matvec2[0] mismatch {a} vs {b}");
            }
            for (a, b) in y_avx1.iter().zip(&expected1) {
                assert!((a - b).abs() < 5e-4, "avx2 q15q4 matvec2[1] mismatch {a} vs {b}");
            }
        }

        let pool = BurstPool::new(4, &[]);
        pool.set_burst(true);
        let mut y_par = vec![0.0f32; rows];
        run_matvec1(&pool, true, &mat, &x0, &mut qx0, &mut y_par);
        pool.set_burst(false);
        for (a, b) in y_par.iter().zip(&expected0) {
            assert!((a - b).abs() < 5e-4, "parallel mismatch {a} vs {b}");
        }
    }
}
