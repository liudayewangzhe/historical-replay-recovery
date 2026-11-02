# Cost-matched recovery from durable backdoors in federated learning

Code for the paper "Is Historical Replay Worth Its Cost? A Cost-Matched Study of Recovery from
Durable Backdoors in Federated Learning". Everything runs on one GPU with PyTorch 2.8 in
deterministic mode; all methods of a seed run in one process from the same attacked prefix,
client partition and minibatch schedules.

## Layout

- `src/` experiment code
  - `train_holdout.py`: trains a federated model with full participation until the server
    validation accuracy first reaches a target and saves that onset checkpoint.
  - `dr_recovery.py`: attack window, every recovery method, continuation (oracle identities).
  - `dr_partial.py`: the same with per-round partial participation.
  - `dr_detect_recovery.py`: recovery triggered by the deployed update-statistics detector.
  - `dr_pilot_durability.py`: attack-strength calibration (attack followed by filtering).
  - `dr_lib.py` (attack, setup, Neurotoxin-style mask), `dr_baselines.py` (FedEraser-style,
    Crab-style, server fine-tuning), `fedrecover_adapter.py` (FedRecover-style),
    `run_covariance.py` (sampled, paired, capped and fitted replay), `run_recovery.py`,
    `run_paired.py`, `holdout_data.py`, `imagenet_gn_init.py`, `check_holdout.py`.
- `scripts/` the queue scripts that produced the reported runs.
- `analysis/` tables and figures from the per-seed CSV files.

## Results

`results_sync/results/<run>/s<seed>/` holds the per-seed outputs behind every table and figure
(`recovery.csv`, `continuation.csv`, `attack.csv`, `config.json`); model checkpoints are not included.
The analysis scripts read these files and write to `analysis/out/`, `manuscript/figures/` and
`manuscript/sections/`, so they run unchanged from the repository root, e.g.
`python analysis/statistics.py`.

## Reproducing

1. Data: torchvision CIFAR-10, CIFAR-100 and Fashion-MNIST under `data/`.
2. Checkpoints, e.g. CIFAR-10 seed 932:

       python src/train_holdout.py --dataset cifar10 --data data --out results/holdout_cifar10_s932 \
         --device cuda --seeds 932 --clients 50 --malicious 10 --rounds 400 --local-epochs 1 \
         --batch 64 --lr .03 --validation-fraction .1 --eval-interval 5 --target-accuracy .75 \
         --initialization imagenet1k_gn

   Fashion-MNIST uses `--lr .1 --target-accuracy .85 --initialization random`
   (Dirichlet 0.1 cohort: `--alpha 0.1 --target-accuracy .8`); CIFAR-100 uses `--target-accuracy .5`.
3. Recovery (main study, CIFAR-10):

       python src/dr_recovery.py --checkpoint results/holdout_cifar10_s932/checkpoint_seed932.pt \
         --out results/dr_main_cifar_a/s932 --delays 8 16 --continuation 16 --scale 1

   Fashion-MNIST: `--delays 8 16 32 64 --scale 3`; Neurotoxin-style: `--scale 10 --neurotoxin 0.5`.
   Cost-matched study: `scripts/run_c_full.sh`; partial participation: `scripts/run_c_q3.sh`,
   `scripts/run_c_partial05.sh`; late onset estimate and repeated sample draws:
   `scripts/run_c_fmnist_extra.sh`; FedRecover-style schedules: `scripts/run_c_fr_sens.sh`;
   deployed detector: `scripts/run_detect_cifar10.sh`.
4. Analysis: `analysis/analyze_dr.py <run>`, `analysis/statistics.py`, `analysis/synthesis.py`,
   `analysis/make_summary_table.py`, `analysis/make_figures.py`, `analysis/make_tables.py <run>`,
   `analysis/cost_matched.py`, `analysis/partial.py`, `analysis/val_check.py`, `analysis/onset_draws.py`,
   `analysis/fig_overview.py`.

Seeds 931 (CIFAR-10), 950 (CIFAR-100) and 1771 (Fashion-MNIST) were used only for attack
calibration and are excluded from every analysis.

Storing the retained uploads of a 16-round CIFAR-10 window needs about 29 GB of host memory
(39 GB peak per process).

## License

MIT (see `LICENSE`).
