# DOCTR as 2D CUT data

`prepare_doctr_cut.py` exports OCT B-scans from `mydata` into the folder
structure expected by this repository's `UnalignedDataset`. Domain A is
`SELFF-OCT-preproc_v2`; domain B is `SD-OCT`. It samples 64 B-scans uniformly
from each volume and exports them as 256 × 256 grayscale PNGs. Resizing keeps
the original aspect ratio; edge pixels fill the unused area. The existing
loader converts these PNGs to three-channel RGB when reading them. No mask or
boundary annotation is used for CUT training.

The subject split bundled with DOCTR is kept: 29 subjects in `trainA/B`, 7
in `validA/B`, and 10 in `testA/B`. Both eyes of a subject stay in the same
split. A and B are loaded independently by `UnalignedDataset`; the common
subject and eye identifiers in filenames do not make training pairs.

Run from the repository root:

```bash
.contrastive-unpaired-translation/bin/python scripts/prepare_doctr_cut.py
.contrastive-unpaired-translation/bin/python train.py \
  --dataroot ./datasets/doctr_cut_2d --name doctr_selff2sd_cut \
  --model cut --dataset_mode unaligned --CUT_mode CUT \
  --preprocess none --no_flip --gpu_ids -1 \
  --display_id -1 --num_threads 0
.contrastive-unpaired-translation/bin/python test.py \
  --dataroot ./datasets/doctr_cut_2d --name doctr_selff2sd_cut \
  --model cut --dataset_mode unaligned --CUT_mode CUT \
  --preprocess none --no_flip --gpu_ids -1
```

`--preprocess none` is necessary because the exported PNGs have already been
resized with their aspect ratio preserved and extended to 256 × 256. The default
`resize_and_crop` would resize them again. `--no_flip` keeps retinal orientation
fixed. The examples use CPU; CUDA training can use a valid GPU ID instead.
`--display_id -1` disables the optional Visdom server while retaining HTML
results and loss logs. `--num_threads 0` keeps loading in the main process,
which is a reliable default on macOS.

Generated files go to `datasets/doctr_cut_2d`, which is ignored by Git. The
original NIfTI volumes in `mydata` are not changed. `manifest.csv` records
the original B-scan index of each image. The 2D model does not enforce
consistency between neighboring B-scans, and the mask/boundary data should be
used separately to assess anatomy after translation.
