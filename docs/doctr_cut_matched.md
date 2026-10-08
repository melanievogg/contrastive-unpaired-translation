# Geometry/resolution matched DOCTR export

The baseline script and dataset are not modified. Run commands from the CUT repository root, using its Python environment. Install exporter dependencies with `python -m pip install -r scripts/requirements-doctr-matched.txt`. The batch checker additionally uses the existing CUT dependencies.

## Inputs and safety

`--source`, `--output`, and `--split-dir` are required. The source contains subject directories directly. Only `<subject>/<OD|OS>/{SELFF-OCT-preproc_v2,SD-OCT}/data.nii.gz` and SD `boundaries.npy` are read. Baseline PNGs are never inputs. Split files are the unchanged `train_subjects_0.txt`, `valid_subjects_0.txt`, `test_subjects_0.txt` from the baseline workflow; they must cover the source subjects exactly and be disjoint. No new split is generated.

Choose `--subjects 002`, `--max-subjects 1`, or explicitly `--all-subjects`. Existing output directories are rejected, including previous partial exports. Use a separate fresh directory for each run. Source/output overlap and the baseline output are rejected. NIfTIs are only opened for reading. `--dry-run` checks configuration, split membership and image-file presence and lists expected inputs without opening NIfTI data or writing output. Actual processing prints input paths, shapes and zooms during preflight, before export starts.

## Geometry and assumptions

Axes are the supplied DOCTR convention: x=A-scans, y=axial, z=B-scans. Orthogonal affine columns and consistency with header zooms are checked. Units must be mm, or unknown (explicitly reported and assumed mm for DOCTR). Affines are recorded; scanner world origins/orientations are not treated as inter-device anatomical registration. Acquisition axes must have the same anatomical direction convention. This performs physical sampling/FOV matching, not registration or exact foveal alignment.

SELFF remains complete. For each SD axis, the resampled size is `floor((original_size-1)*original_spacing/target_spacing)+1`, with target spacing from that eye's SELFF header. The output voxel-centre support is centred within the original support. Linear 3D interpolation uses `input_coordinate = output_coordinate * target_spacing/original_spacing + offset`. No per-volume normalization is applied; inputs must decode to uint8, interpolation is float32, and the final crop is rounded/clipped to uint8. No 2D resizing is performed. SD must cover the complete SELFF target shape without padding.

The two lateral crops are geometrically centred, with at most half a target voxel of integer crop rounding. The SD axial crop is one constant interval for the whole volume. `boundaries.npy` is required to verify its safety. It must contain nine surfaces whose values are **zero-based axial voxel coordinates**, not mm or one-based indices. The default is the confirmed DOCTR convention `--boundary-layout lzx`: `(layer, B-scan, A-scan)`. SELFF shape `(9,209,276)` establishes this convention, which also applies to square SD arrays `(9,512,512)`. Optional `auto` accepts only a unique permutation to `(9, original_x, original_z)` and therefore rejects square SD arrays. `l` denotes the surface axis. Invalid/missing/nonfinite/out-of-volume coordinates fail explicitly; no missing-value sentinel is guessed.

All surfaces over the retained lateral footprint (including neighbouring source samples) contribute to the retinal min/max. Coordinates are transformed with the same resampling scale and offset. The geometrical axial crop is kept if safe; otherwise it is shifted to the nearest integer position containing the full retinal range, plus optional `--axial-margin-mm`. If no such interval exists, processing stops. SELFF is not axially cropped. The default margin is zero. The real boundary coordinate convention must be confirmed during local QC; shape alone cannot prove it.

Both domains have identical target spacing, shape, and physical FOV per eye. FOV is reported as `shape * spacing` (voxel-cell extent); centre-to-centre extent is `(shape-1)*spacing`. Cross-eye target shapes/spacings must also agree (spacing tolerance `rtol=1e-5, atol=1e-8`), because unpaired CUT can combine different eyes. Native width/height must be divisible by four: this CUT loader otherwise resizes even with `--preprocess none`. Mismatches abort rather than silently changing geometry.

64 distinct indices are sampled with rounded `np.linspace` only after cropping. Manifest original indices for resampled SD are nearest source indices; `original_bscan_coordinate` retains the fractional coordinate. Crop bounds are half-open in resampled coordinates. The CSV also records affines, offsets, shapes, spacings, FOV, sizes and paths. `summary.json` includes counts and geometry for every volume. QC contains separate A/B PNGs at first/middle/last sample positions for up to five eyes. They are inspection images, not paired training data.

## Mac: first one eye

Set `SPLIT_DIR` to the directory containing the **existing** split files. The value below is the old script's relative default; change it if those files live elsewhere.

```bash
SPLIT_DIR="mydata/code/Segmentation/nnunet/subject_split"
python scripts/prepare_doctr_cut_matched.py \
  --source "/Volumes/BilderMelli/Praktikum/mydata/" \
  --output datasets/doctr_cut_2d_matched_smoke \
  --split-dir "$SPLIT_DIR" --subjects 002 --eyes OD --boundary-layout lzx --dry-run

python scripts/prepare_doctr_cut_matched.py \
  --source "/Volumes/BilderMelli/Praktikum/mydata/" \
  --output datasets/doctr_cut_2d_matched_smoke \
  --split-dir "$SPLIT_DIR" --subjects 002 --eyes OD --boundary-layout lzx
```

Check the smoke manifest's split and use that phase in the loader check (`train`, `valid`, or `test`):

```bash
python scripts/check_doctr_cut_batch.py --dataroot datasets/doctr_cut_2d_matched_smoke --phase train
```

For actual SELFF shape `(276,200,209)` the expected batch is `(1,3,200,276)`. Inspect QC and confirm boundary semantics before proceeding. Then run the same exporter to a separate new full output:

```bash
python scripts/prepare_doctr_cut_matched.py \
  --source "/Volumes/BilderMelli/Praktikum/mydata/" \
  --output datasets/doctr_cut_2d_matched \
  --split-dir "$SPLIT_DIR" --all-subjects --boundary-layout lzx
```

## CUT training

A=SELFF, B=SD. The UnalignedDataset draws domains independently. These commands use the repository's usual CUDA setup; select a suitable GPU configuration for your training machine.

```bash
python train.py --dataroot datasets/doctr_cut_2d_matched --name doctr_matched_AtoB --model cut --CUT_mode CUT --direction AtoB --preprocess none --batch_size 1 --no_flip
python train.py --dataroot datasets/doctr_cut_2d_matched --name doctr_matched_BtoA --model cut --CUT_mode CUT --direction BtoA --preprocess none --batch_size 1 --no_flip
```

## Synthetic verification

```bash
python -m unittest discover -s tests -p test_prepare_doctr_cut_matched.py
```

Tests create tiny NIfTIs under a temporary directory inside `tests/`, then clean them up. They verify linear sampling, a necessary axial shift, impossible retinal coverage, source protection, dry-run, 64 PNGs/domain, manifest/QC, and actual CUT batch loading without resizing. No real DOCTR data are accessed. Real-data QC and a complete export must be run locally; they have not been performed here.

## Original SD boundary overlay QC

For each of the first `--qc-eyes` eyes (default 5), QC additionally saves
`<subject>_<eye>_SD_original_zNNNN_raw.png`, `_boundaries.png`, and
`_boundaries.json`. The selected original SD z index defaults to `shape[2]//2`;
choose another with `--qc-bscan-index 256`. This is an original-volume index,
not a matched/cropped index. Both images have the native original B-scan size.

The raw image is exactly `volume[:, :, z].T`. For stored `lzx` boundaries,
`boundaries[:, z, :]` supplies one axial y value per x/A-scan. Internally the
canonical `(layer,x,z)` representation uses `bounds[:, :, z]`, which is equivalent.
The overlay draws ILM in green and OB_RPE in magenta without resampling or crop.
The JSON records the z index, colors, surface indices, and original y coordinates.
Compare the raw and overlay PNGs to check alignment. Training PNGs remain untouched.

Surface identity is separate from axis order: QC defaults to ILM=surface 0 and
OB_RPE=surface 8 (zero-based first/last surfaces). Confirm these labels against your
surface definitions; if needed supply `--ilm-layer` and `--ob-rpe-layer`.
All nine surfaces still determine the crop independently of the two QC labels.
Synthetic tests use square lateral axes and different x/z variation to expose
an accidental axis swap, and verify the overlay's axial y coordinates.
