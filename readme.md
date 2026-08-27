# MUL

Class-level machine unlearning on long-tailed datasets. The idea is to remove one class from a trained classifier and see what it actually costs, measured separately for head, mid and tail classes instead of one averaged number.

Research code, so expect rough edges.

## Setup

PyTorch + torchvision, plus numpy/matplotlib for the plots.

Datasets, checkpoints and outputs are gitignored. Paths in the scripts are relative to the parent of the repo, so run everything from one directory above `MUL/`:

```
.
└── MUL/
    ├── Datasets/   # long-tailed splits
    ├── Models/     # Teachers/, Students/, Oracles/
    ├── Plots/
    └── Scripts/
```

You need a trained teacher at `Models/Teachers/{dataset}/teacher_im{imb_factor}.pth` before unlearning anything. ResNet-18 for CIFAR-10, ResNet-50 for the rest, with `conv1`/`maxpool` swapped out for the 32x32 inputs.

## Scripts

`imp_master.py` is the main one. It distills the teacher into a student with the forget class masked out of the teacher's distribution.

- `--pipeline imp` rebalances the teacher's probability mass across head/mid/tail before masking
- `--pipeline nor` just masks, no rebalancing
- `--pipeline orcl` trains the retrain-from-scratch oracle

```bash
python MUL/Scripts/imp_master.py --dataset cifar100 --imb-factor 200 --forget-class tail --clustering-type mvm --pipeline imp
```

Baselines: `neggrad.py`, `scrub.py`, `lcodec.py`, `dtd.py`, `ul.py`. `oracle_master.py` runs the oracle on its own.

`get_statistics.py` computes per-class stats and produces the `mean` / `mvm` head-mid-tail clusterings. `gradcam_final.py` and `tsne_final.py` are for the visualizations.

## Common flags

| Flag | Values |
|---|---|
| `--dataset` | `cifar10`, `cifar100`, `food101`, `places365` |
| `--imb-factor` | imbalance ratio, default `200` |
| `--forget-class` | `head`, `mid`, `tail`, or a class index |
| `--clustering-type` | `manual`, `mean`, `mvm` |
| `--seed`, `--device` | |

The baseline scripts also take method-specific flags (`--scrub-*`, `--lcodec-*`, `--ul-*`), all with defaults from the respective papers.