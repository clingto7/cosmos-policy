# Task 03 training budget and checkpoint pilot

**Run date:** 2026-10-22  
**Training host:** Chuangzhi RTX 4090  
**Training source revision:** `4a21be6`  
**Pilot seed namespace:** `300000000`  
**Final comparison seed namespaces:** `100000000` (`none`, `cov`, `inv`) and `200000000` (`inv_cov`); untouched

## Training measurements

Training continued from the existing iteration 3 checkpoint through iteration 9
using the public Cosmos-Predict2-2B-Video2World weights as the common base.
The dataset manifest contained the 100 audited task 03 demonstrations. Training
used one GPU, batch size 1, gradient accumulation 1, and therefore effective
batch size 1.

| Measure | Result |
| --- | ---: |
| Peak PyTorch GPU allocation | 33.38 GiB |
| Peak NVML GPU use during training | 36.89 GiB |
| Effective batch | 1 |
| Updates from iteration 3 to 9 | 6 |
| Elapsed time, after resume through completion | 128 s |
| Measured throughput, including checkpoint saves | 168.8 updates/hour |
| Iteration 9 checkpoint size | 19 GiB |
| Iteration 9 checkpoint save time | 17 s |

The 17 second save measurement is from the final checkpoint request to its
completion in the training log. The iteration 6 interval took 35.43 seconds
including the optimizer update and checkpoint activity; that interval does not
isolate checkpoint time.

## Pilot protocol

Checkpoint budgets are iterations 3, 6, and 9. Evaluation uses the same
episode seed sequence at each budget, with the `none` channel and a fixed
episode 000000 HDF5 anchor. The runner maps seed base `300000000` to task 03
base seed `300300000`; episode seeds are `300300000` through `300300009`.
These are disjoint from both final comparison seed namespaces above.

Each episode runs at most 400 policy steps (1,200 physics steps), executes
16-action chunks at 20 Hz, and records its seed, checkpoint label, outcome,
stage completion, query count, termination reason, and any error. Success-rate
intervals use the 95% Wilson score interval. The pilot is used to diagnose
failures and compare budgets; the two-interval five-percentage-point plateau
rule is not applied unless the episode count can resolve that difference.

| Budget (updates) | Episodes | Successes | Success rate (95% Wilson CI) | Stage completion | Runtime errors | Episodes/hour |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 10 | 0 | 0% (0%–27.8%) | — | 0 | 36 |
| 6 | Pending | — | — | — | — | — |
| 9 | Pending | — | — | — | — | — |

The iteration 3 run is a smoke-level pilot, not evidence of convergence. It
completed ten full-horizon episodes without runtime errors; the first episode
had zero normalized progress and no completed stage. No plateau claim is made.

## Zero-success diagnosis

The first valid step 3 episode completed all 25 policy queries and the full
1,200 physics-step horizon. It had no runtime error or safety violation, but
completed none of the three glass-placement stages and recorded zero normalized
progress. This points to a policy quality or training-budget issue; it does not
indicate a broken observation-to-action pipeline. The remaining fixed-seed
episodes and later checkpoints will determine whether that diagnosis holds.

## Raw result artifacts

The per-episode result rows and simulator logs are retained on the experiment
host under `/inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/hqb/task03/pilot/`.

## Scale-up training run

The follow-up run uses a single cumulative budget trajectory so the selected
15k, 20k, and 25k checkpoints share the same initialization and data order.
It resumes the complete iteration 9 DCP training state, then trains to total
iteration 25,000. The first nine updates used effective batch 1; the scale-up
run uses microbatch 1 with gradient accumulation 2 (effective batch 2). This
fits the RTX 4090 memory observed at about 40.3 GiB NVML use while increasing
the effective batch over the smoke run.

The remote launcher is
`/inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/hqb/task03/scripts/train_task03_budget_ladder.sh`.
It saves a resumable checkpoint every 1,000 optimizer updates for recovery,
including the selected 15k, 20k, and 25k budgets. Intermediate checkpoints
are temporary and should be removed only after a newer resume point is verified.
Its output is under
`/inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/hqb/task03/output-budget-effbatch2-25k/`;
relaunching the same script resumes from the latest completed checkpoint. The
first optimizer updates were observed running at iteration 11. Checkpoints at
iterations 1,000, 2,000, and 3,000 were saved (about 19 GiB each). Saving
iteration 4,000 failed with `OSError: [Errno 122] Disk quota exceeded`; the
training process then exited. The `latest_checkpoint.txt` pointer still names
iteration 3,000, while the iteration 4,000 directory is incomplete and was
left untouched. No requested scale-up evaluation budget has been reached.

After the instance was recycled, the current instance had no training process.
The earlier run had written checkpoints to the HDD project mount, whose quota
was exhausted. On 2026-10-23, the proof output (92 GiB), scale-up output
(55 GiB), and 100 task 03 HDF5 files (17 GiB) were copied to the project's
private HDD2 mount and verified with a checksum-mode rsync comparison. The HDD
output directories were replaced by symlinks to HDD2, then the verified HDD
copies were removed; HDD available space rose from 51 GiB to 197 GiB. The
training manifest now points to HDD2 HDF5 paths; the dataset loader found
100 entries and 74,016 samples, and the latest complete scale-up checkpoint
remains iteration 3,000. Training has not been restarted. Later evaluation
should use the fixed pilot seeds and keep the five-point plateau rule
unsupported unless the sample size can resolve it.
