> Historical benchmark record. The commands below used the earlier diagnostic trainer.

# AGE-YOLO benchmark and validation memory

## Review of runs 02 and 03

Both runs used the same trainer revision, model and dataset hashes, seed 42,
batch 16, AMP, eight training workers, and the same recorded backend flags.
They ran consecutively, synchronized first. The optional validation check
started after the native run's training timing ended.

| Measured training window | 02 synchronized | 03 native |
| --- | ---: | ---: |
| Timestamped GPU samples | 12 | 93 |
| Images/s | 65.96 | 8.30 |
| Mean GPU utilization | 78.25% | 11.46% |
| Median batch-body wall time | 0.2392 s | 0.2420 s |
| Longest batch body | 0.2819 s | 20.7231 s |
| Mean sampled power | 66.15 W | 14.99 W |
| Graphics/SM clock range | 2,655–2,707 MHz | 180–2,700 MHz |
| Temperature range | 65–71 °C | 57–71 °C |
| P-states | P4 only | 66 P4, 11 P5, 16 P8 |
| Peak device VRAM | 4,182 MiB | 4,182 MiB |
| Warmup-only projection for 300 epochs | 9.28 h | 73.71 h |

At or above 60% utilization, the native run averaged 66.22 W and 2,685 MHz
SM clock, nearly identical to the synchronized averages of 66.15 W and
2,684 MHz. Active memory clocks were 9,001 MHz in both. In contrast, the
68 native samples at or below 5% utilization averaged 9.20 W. Both runs
reached 71 °C. The power limit field was unavailable in both. These
patterns fit the GPU downclocking during gaps in work, with no positive
evidence of sustained thermal or power throttling. The runs did not
sample throttle-reason flags.

Fourteen native batch bodies exceeded one second and occupied 85.19 s of
the 96.36 s window. The median of the other 36 was 0.2380 s. The first
major stall was batch 44, from 09:35:43.276 to 09:35:57.842. Nine of its
14 overlapping GPU samples were zero. Batch 59 lasted 20.72 s, with 13
zero samples out of 19, then batch 60 took 9.73 s. The longest reported
loader gap was only 0.0748 s. The stalls therefore occurred inside the
measured batch bodies after the main loop received data. The GPU sampler
continued reporting throughout, with at most 1.13 s between samples.

The native loop stalled intermittently, but the records cannot attribute
that to the sync setting. The runs differ in order and host conditions
could have changed. No timestamped guest scheduling, I/O, swap, or Windows
host telemetry was retained. An uninterrupted companion sampler argues
against a complete guest freeze while leaving process-specific contention
or GPU submission waits open. WSL2 itself and Windows host contention are
hypotheses rather than established causes.

The benchmark measures actual work from the native training loop. In native
mode, fences at the boundaries make total elapsed throughput meaningful,
but per-batch host wall time includes CPU work, logging, implicit device
waits, and scheduling gaps. It is not kernel time. Synchronized mode adds
an explicit GPU wait per batch and an immediate CUDA scalar read. Full
training installs neither benchmark callback. It retains the native loss
and progress scalar reads and adds epoch-end validation and checkpoint
saves. The throughput gap cannot be generalized to full training, and
neither runtime projection covers post-warmup accumulation, validation, or
saving. [PyTorch CUDA timing semantics](https://docs.pytorch.org/docs/2.7/notes/cuda.html#asynchronous-execution)

## Warmed native validation result

Run 03 completed validation on all 230 images with the trainer state still
resident. It processed seven FP16 batches of `32 × 3 × 352 × 672` and
one batch of six, with 16 workers. It started with 2,964 MiB free as
reported by CUDA. Peak PyTorch allocated memory was **600 MiB**, peak
reserved was **3,830 MiB**, and sampled device usage peaked at **4,182 MiB
out of 8,151 MiB**. Validation completed in 40.61 s without CUDA OOM.
Twenty-four of its 39 GPU samples were zero, despite the native validator's
per-stage timing synchronizations. This confirms batch-32 fit with the
benchmark's warmed state and adds evidence that intermittent idle periods
are not unique to the unsynchronized training callback. Later checkpoints
can produce a different number of detections and a different NMS load.

## What the first benchmark measured

`experiments/benchmarks/age-yolo-gc10-bench-01/` completed the requested
60 training batches at batch 16. The 50 timed batches averaged 0.789763 s,
20.259 images/s, and 0.001397 s between the previous batch-end callback and
the next batch-start callback. That wait accounts for about 0.18% of the
measured time. The peak device sample was 4,182 MiB out of 8,151 MiB.
PyTorch peaks were 3,406 MiB allocated and 3,830 MiB reserved.

The 39 samples behind the reported 28.87% GPU utilization span
09:27:03.380 through 09:27:42.025. The median was 2%. In that window,
23 samples were at or below 5%, 12 were zero, and 13 were at or above
70%. Two stretches of nine or more consecutive samples stayed at or
below 5%, followed by activity up to 87%. This shows intermittent
GPU activity. NVML utilization is a sampled kernel-active time measure.
[NVIDIA utilization definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)

The original CSV contains utilization and VRAM only. Historical power,
clocks, and temperature are unavailable. An idle spot check after the run
reported P4, about 6.6–6.9 W, 50–51 °C, and 0 MHz graphics/SM clocks.
The power limit query returned `[N/A]`. These readings cannot establish
training throttling. WSL also showed 1.3 GiB of occupied swap after the run.
Occupancy alone cannot establish swapping during the idle stretches.

## Settings and timing limits

| Area | What the record and pinned source show |
| --- | --- |
| Precision | AMP was active. Training uses CUDA autocast and a dynamic gradient scaler. In-training validation uses the FP16 EMA and includes validation loss and NMS. |
| Determinism | Seed 42, `deterministic=true`. The fork requests deterministic PyTorch algorithms in warn-only mode, sets cuDNN deterministic mode, and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It leaves cuDNN algorithm benchmarking at the process default, observed as off in a fresh pinned environment. The original run did not record runtime backend flags. |
| Accumulation | Nominal batch is 64. At batch 16 the native warmup ramps accumulation from 1 toward 4 over 1,377 batches. The first 60 are still at accumulation 1 by the pinned schedule. Training therefore performs optimizer-step work more frequently in this window than after warmup. |
| Data | Eight training workers, 16 validation workers, image cache disabled. The fork's default pins host memory and uses nonblocking image transfers. The original run generated label metadata caches during setup, outside the timed window. |
| Batch instrumentation | The original callback used both `isfinite().all().item()` and `torch.cuda.synchronize()` per batch. The scalar read forces a host wait; the subsequent sync also waits for remaining work. This changes queueing. The reported compute time is elapsed time inside the batch body, including CPU work and synchronization. |
| Native synchronization | The trainer formats CUDA loss scalars for progress each batch. The detection loss and target assigner also branch on CUDA scalar results. Both add host waits independently of the benchmark. The available logs do not quantify their cost. |

Loader delivery at the main loop was small in the instrumented run.
This does not exclude background worker contention, memory pressure, CPU
descheduling, or I/O stalls in other parts of the batch body. Likewise,
explicit synchronization is a confirmed measurement perturbation, but
the available evidence cannot attribute the multi-second GPU idle stretches
to it. The 30.21-hour projection extrapolates LR warmup with accumulation
1 and excludes validation, checkpoint writes, and setup. Treat it as a
training-only extrapolation, with an unknown error relative to a full run.

## Validation memory

The native loader was configured for batch 32 and 16 workers, but the
first benchmark stopped before validation. All validation images are
2048 × 1000. The pinned rectangular loader's stride-32 padding gives
`(32, 3, 352, 672)` for a full batch at nominal image size 640.

A synthetic smoke check ran that shape through the native in-training
FP16 EMA validator, including the native loss and postprocessing. It
completed at **404 MiB peak PyTorch allocated** and **594 MiB peak
reserved**. It used one synthetic batch on a freshly initialized model.
The check did not exercise the real validation loader, trained
prediction density, or GPU allocations left by training and optimizer
steps. These are isolated smoke numbers, not a full-run memory budget.

The manual `--validation-memory` option invokes the exact native
validator on all 230 validation images after the timed batches, with
the current training model, optimizer, EMA, and allocator still present.
It records starting free VRAM, peak allocated and reserved memory,
sampled device VRAM, actual input shapes, and success or CUDA OOM.
Validation time is outside the benchmark's training measurement.
No test data or checkpoint selection is involved. Run 03 completed this
warmed-state check, as recorded above.

## One manual repeat

After authorizing the diagnostic pair, run native first to reverse the
original order. Both runs stop after 60 batches. Omit validation from both
to keep the diagnostic sequence comparable. Keep one terminal, the same AC
connection and power mode, and comparable starting temperatures. Close
competing Windows and WSL workloads. From the repository root:

```sh
source .venv/bin/activate
python -u scripts/train.py --mode benchmark --run-id age-yolo-gc10-bench-04-native-repeat
sleep 30
nvidia-smi --query-gpu=temperature.gpu,pstate --format=csv
python -u scripts/train.py --mode benchmark --run-id age-yolo-gc10-bench-05-sync-repeat --sync-each-batch
```

In a separate terminal, start timestamped guest CPU, I/O, and swap recording
before both runs. Stop the sampler with Ctrl+C. The benchmark records its
own GPU telemetry, including graphics/SM/memory clocks, power, temperature,
P-state, and throttle flags when the driver exposes them. Batch records
now include wall-clock starts and elapsed CPU time on the main thread.
Low main-thread CPU time during a long batch identifies time spent off
CPU, which can include CUDA waits, blocking I/O, or scheduling delays.

```sh
host_log=$(mktemp /tmp/flawtrace-host.XXXXXX)
printf 'Host telemetry: %s\n' "$host_log"
vmstat -w -t 1 | tee "$host_log"
```

To record Windows host activity during the same pair, open Windows PowerShell
and start the following. Stop it with Ctrl+C and retain the printed counter
file. Compare timestamps in the same timezone.

```powershell
$hostLog = Join-Path $env:TEMP ("flawtrace-host-" + (Get-Date -Format yyyyMMdd-HHmmss) + ".blg")
Write-Host $hostLog
Get-Counter -Counter '\Processor(_Total)\% Processor Time','\System\Processor Queue Length','\Memory\Pages Input/sec','\PhysicalDisk(_Total)\Avg. Disk sec/Transfer' -SampleInterval 1 -Continuous |
  Export-Counter -Path $hostLog
```

Review the two timing windows and their per-batch records:

```sh
native_run=experiments/benchmarks/age-yolo-gc10-bench-04-native-repeat
sync_run=experiments/benchmarks/age-yolo-gc10-bench-05-sync-repeat
cat "$native_run/benchmark.json" "$sync_run/benchmark.json"
head -n 5 "$native_run/gpu.csv"
awk -F, 'NR == 1 || $4 > 1' "$native_run/batch_timing.csv"
awk -F, 'NR == 1 || $4 > 1' "$sync_run/batch_timing.csv"
```

## Priorities

1. **Keep the native training configuration.** Retain batch 16, seed 42,
   AMP, deterministic mode, and the current workers and cache settings.
   Confirm repeatable throughput with the pair above. The existing pair
   does not justify adding device-wide fences to full training or choosing
   either projected duration as a stable estimate.
2. **Address a measured host or power bottleneck first.** Stable AC power,
   consistent power mode, and removing competing host workload preserve
   the baseline settings. If telemetry shows memory pressure from workers,
   propose a worker-count comparison before changing it. Image caching and
   additional workers are low priority given the measured loader wait.
3. **Profile native host waits if the stalls persist.** Loss scalar reads
   and live progress are plausible hot spots. Measure their cost before
   any code change. Preserve live progress and the pinned upstream tree.
4. **Treat numerical and sampling changes as new conditions.** Disabling
   deterministic mode, enabling cuDNN autotuning, changing batch size or
   accumulation, or changing worker count can affect reproducibility even
   with seed 42. Keep them out of the baseline unless separately approved,
   benchmarked, and documented. [PyTorch determinism and cuDNN](https://docs.pytorch.org/docs/2.7/notes/randomness.html)
