# FlawTrace on UPES HPC

Run setup and submit jobs from the shared HPC checkout on `mgmt01`.
## Install and prepare GC10-DET

From the repository root on `mgmt01`:

```bash
bash scripts/setup_hpc.sh
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
if [[ ! -d data/gc10-det ]]; then python scripts/prepare_data.py; fi
python scripts/check_data.py
if [[ ! -d data/gc10-det-aug42-x4 ]]; then python scripts/augment_data.py; fi
python scripts/augment_data.py --verify
python scripts/train.py --config configs/hpc.yaml --preflight
```

Setup creates or reuses only `flawtrace-gc10`. It installs Python 3.12.14,
the project pins including PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128,
and Conda's `bsdtar`. The HPC build uses `opencv-python-headless==4.11.0.86`
to import `cv2` without X11. Rerun the setup command above to migrate an
existing environment.
The verified trainer checks AMP with AGE-YOLO itself and initializes model
weights from YAML on fresh runs.

Training uses the deterministic AUG42 ×4 split under `data/gc10-det-aug42-x4`
through `configs/gc10_augmented.yaml`. The dataset contains 7,336 training
images and 11,524 training objects. Validation contains 230 images and 345
objects. Run the standalone verifiers above after preparing or changing the dataset.

## GPU verification and training

PBS requests one GPU. Confirm which CUDA index the job is authorized
to use before setting `DEVICE`. The recorded run on `gpu02` used
`cuda:5`. Device indices are relative to the job’s CUDA visibility,
so verify the assignment for each new run. The launcher inherits the
submission environment through `#PBS -V`. The trainer reports the selected
device and available memory at startup.

`configs/hpc.yaml` uses batch **64**, a deviation from the official **16**.
The job keeps learning rate **0.01**, maximum 300 epochs, 640 px, SGD,
patience 50, seed 42, eight training workers, and the official online
augmentation and AMP settings. The native trainer requests batch **128**
for validation. The audited split contains 230 images, so the native loader
uses a validation batch size of **128**. The fork's nominal batch size
of 64 gives expected accumulation **1** and effective decay **0.0005** on
the decayed weight group with other groups at zero. The run records the
observed values. CUDA out-of-memory errors stop the job and appear in the log.

The checked-in configuration records `hpc-full-06`. Choose an unused run ID
for each new experiment. Submit from the repository root.

```bash
job=$(qsub submit.sh)
qstat -f "$job"
tail -F _main_train.log
```

Use Ctrl-C to leave `tail`. The log appears when the job starts and shows
epochs, batches, losses, GPU memory, and validation metrics. Inspect
`qstat -xf "$job"` after it exits. For live GPU statistics, run
`nvidia-smi -l 1` in a shell on the assigned compute node if permitted.

Run files are under `experiments/<run-id>/`. They include `run.json`,
`environment.json`, `effective_config.json`, `terminal.log`, configuration
copies, the fork's `args.yaml` and `results.csv`, checkpoints, and plots. The launcher redirects Python output to `_main_train.log` and truncates
that file at the next launch. The per-run `terminal.log` remains available.

## Recover an interrupted run

Wait for the prior job to exit. Use the same run ID, directory, batch, and
dataset. Native resume restores the saved epoch and optimizer state.
If the native `last.pt` is missing or unreadable, the launcher tries
the separate atomic recovery copy from the most recent saved epoch.
If the results CSV extends beyond the recovered checkpoint, the launcher
archives that CSV with the attempt metadata and resumes the active CSV from
the saved epoch. Earlier job logs remain available.

```bash
run=experiments/hpc-full-06
ls -lh "$run/weights/"
sed -i 's/^RESUME: false$/RESUME: true/' configs/hpc.yaml
resume_job=$(qsub submit.sh)
qstat -f "$resume_job"
tail -F "$run/terminal.log"
```

A completed run is closed to further training. If the job stopped before its
first usable checkpoint, start over with a fresh run ID. For a quick review,
read `results.csv` and the status in `run.json`.

## PBS limits to confirm

`submit.sh` requests one node, one GPU, 16 CPUs, and 32 GB of host RAM.
The GPU check requests 10 minutes. The visible PBS server and queue
configuration did not advertise a training walltime default or ceiling.
Confirm the full-job walltime and any per-user resource policies with the
cluster operator before submission. Use PBS job history to review resource usage after a run.

```bash
qstat -Qf gpu
qstat -f "$job"
qstat -xf "$job"
```
