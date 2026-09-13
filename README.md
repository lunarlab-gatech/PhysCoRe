# PhysCoRe

### [CoRL 2026] PhysCoRe: Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics

[Haocheng Yin\*](https://haochengyin.github.io/) · [Shuohan Tao\*](https://shuohantao.github.io/) · [Yongsheng Chen](https://ce.gatech.edu/directory/person/yongsheng-chen) · [Lu Gan](https://ganlumomo.github.io/)

Georgia Institute of Technology

<sub>\*Equal contribution</sub>

[[Project Page]](https://lunarlab-gatech.github.io/PhysCoRe-website/) · [[Paper]](https://arxiv.org/abs/2607.20653) · [[Dataset]](https://huggingface.co/datasets/GeorgiaTech/PhysCoRe)

## Environment setup

We run our code on Linux with an RTX 5090 GPU. Please adjust the PyTorch index URL and
the `TORCH_CUDA_ARCH_LIST` values below to match the CUDA version and GPU architecture on
your own machine. Run the steps in order, since each one builds on the previous.

```bash
conda create -n physcore python=3.10 && conda activate physcore

# 1. PyTorch, CUDA 12.8 build
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 2. everything else
pip install -r requirements.txt

# 3. restore the contrib build of OpenCV, which supervision replaces
pip uninstall -y opencv-python && pip install --force-reinstall opencv-contrib-python

# 4. PyTorch3D, built from source (12.0 = sm_120)
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git"

# 5. the two Gaussian-splat CUDA extensions, for rendering the confidence field.
#    Build source only, renamed so it is not mistaken for gaussian_splatting/.
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git cloned-gaussian-splatting
( cd cloned-gaussian-splatting/submodules/diff-gaussian-rasterization \
  && TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation . )
( cd cloned-gaussian-splatting/submodules/simple-knn \
  && TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation . )

# 6. sparse 3D convolutions, used by RfD. Pick the wheel for your CUDA version;
#    the cu126 build below also runs on the cu128 runtime.
pip install spconv-cu126
```

### Install GroundingDINO and SAM 2 dependencies

Only the 2D stage needs these, so you can skip this section while working from the
released cases. Segmentation relies on GroundingDINO and SAM 2, both of which are
installed from the Grounded-SAM-2 distribution along with their checkpoints:

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git && cd Grounded-SAM-2
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation -e .
TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation -e grounding_dino
cd ..

mkdir -p checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cp ../Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py .
cd ..
```

## Input data layout

Our recordings, trained weights, Gaussian splats and configs are all in the
[dataset repository](https://huggingface.co/datasets/GeorgiaTech/PhysCoRe). Copy its
contents into the root of this checkout and every stage below finds what it needs. Those
cases already include their masks and 3D tracks, so you can skip the 2D stage and start at
[3D Process Data](#3d-process-data).

A case is one directory of time-aligned RGB-D streams from multiple calibrated cameras:

```
<data_root>/<case_name>/
├── calibrate.pkl   # pickled list of N camera-to-world 4x4 matrices
├── metadata.json   # intrinsics (N x 3 x 3), WH, frame_num, serial_numbers
├── color/          # <cam>.mp4, and the same frames as <cam>/<frame>.png
└── depth/          # <cam>/<frame>.npy, uint16 depth in millimeters
```

Record your own cases in this layout, or use one of the released ones as a template; the
dataset README lists every file a case directory holds.

## 2D Process Data

This stage segments the object and the manipulator in every view, tracks them densely
through the sequence, and lifts the tracks into 3D. The released cases already contain its
output, so run it only on your own recordings. Point `CHECKPOINT_DIR` at the directory
holding the three files downloaded above, or the run exits immediately:

```bash
export CHECKPOINT_DIR=$PWD/checkpoints

python datagen/process2d/process_data.py \
  --base_path <data_root> \            # parent directory containing <case_name>/
  --case_name <case_name> \            # the case directory to process
  --category "<object prompt>" \       # object prompt for GroundingDINO
  --controller "<manipulator>" \       # "hand" (default) or "robot gripper"
  --headless                           # render Open3D offscreen (no display)
```

Use `--headless` on a machine without a display, such as a remote GPU node: the lifting,
mask and tracking stages otherwise open an on-screen Open3D window and stop.

## 3D Process Data

This stage completes the observed surface into a closed shape with a Poisson solver, fills
the interior with particles, and writes one training episode.

```bash
python datagen/convert3d/convert_to_episode.py \
  --source_dir <data_root>/<case_name> \        # the case processed in the previous stage
  --output_dir <episode_root>/<case_name> \     # where the episode is written
  --controller_mask_label "<manipulator>"       # must match --controller in 2D Process Data
```

The particle resolution is tuned per object. We used three presets, and the default one
covers ropes and plasticine:

| object | flags |
|---|---|
| default | `--particle_voxel_size 0.008 --target_total_particles 8000 --mask_erode_pixels 2` |
| plush toy | `--particle_voxel_size 0.008 --target_total_particles 12000 --mask_erode_pixels 2` |
| cloth | `--particle_voxel_size 0.004 --target_total_particles 16000 --mask_erode_pixels 2 --interior_to_shell_max_distance 0.004` |

Cloth needs both of its own values: the sheet is thin, so the default 8 mm spacing leaves
almost nothing inside it, and `--interior_to_shell_max_distance` keeps the interior from
being filled where the real sheet does not reach. Raise `--target_total_particles` whenever
the spacing gets finer, or the limit throws away the extra detail. `--mask_erode_pixels`
trims the mask outline, where a pixel sees both the object and the background behind it and
its depth lands between the two.

Each run writes `episode_0000/` into the output directory.

## Augment Data

This stage re-simulates every converted episode under randomly sampled material fields,
which turns a handful of recordings into a training set.

```bash
python datagen/augment/run_recipe.py \
  --config configs/augment_recipe.yaml \   # the augmentation recipe
  (--dry-run)                              # optional, print the planned work and exit
```

In a group's physics block, tune `manipulation_controller_grid_contact_radius` for your own
objects. The controller grips every particle inside that radius, so a larger one holds the
object more reliably but deforms it near the grasp, and a smaller one leaves fewer
artifacts but risks sliding off.

Episodes land in `data_episodes_augmented/episodes/<source>_<group>/`, with one video each
in `data_episodes_augmented/video/` unless you pass `--no-viz`. GPU reductions are not
deterministic, so the same seed reproduces closely but not exactly, and a stiff material
occasionally diverges instead of settling. Check the videos and regenerate the episodes
that blew up.

## MfM Training and Validation

Training runs on the augmented episodes from the previous stage.

```bash
python train_MfM.py --config configs/train_MfM.yaml
```

Validation then scores a trained checkpoint against the real recordings, so point its
checkpoint path at your own weights or at the released `MfM_checkpoint.pt`, and keep the
physics block in step with the checkpoint you are scoring.

```bash
python validate_MfM.py --config configs/validate_MfM.yaml
```

If a rollout video shows the controller detached from the object, raise
`manipulation_controller_grid_contact_radius` in `configs/validate_MfM.yaml`, either the
value in the `rollout` block or a per-case entry under `per_sample_rollout_config`. A
larger radius grips more particles and keeps hold of the object.

### Visualizing the confidence field

The renderer recolors an episode's Gaussian splat by the predicted confidence and warps it
along a saved rollout, so you need both in advance: a splat, which this repository does not
produce, and a trajectory from a validation run with trajectory saving enabled. Pass the
checkpoint and episode that produced that trajectory.

```bash
python render_MfM_confidence_3dgs.py --config configs/render_MfM_confidence_3dgs.yaml
```

Confidence ranges differ between cases, so `norm_hi`, the ceiling of the color map, is
tuned per case: too high a ceiling leaves the object uniformly dark, too low a one
saturates it to a flat yellow. The dataset README lists the value behind each released
overlay video.

## RfD Training and Validation

RfD predicts a velocity correction for every MPM grid node, which is added to the node
velocities the solver computed before the particles read them back. Before you start
training, set `refiner_checkpoint` in the config to the MfM weights from the previous
stage, which RfD reads frozen and never updates.

```bash
python train_RfD.py --config configs/train_RfD.yaml
```

Validation then scores every checkpoint that training saved. It finds them through
`train.output_dir`, so that field has to match the one in `configs/train_RfD.yaml`. The
results are merged into the training history and plots, and every episode gets a video.

```bash
python validate_RfD.py --config configs/validate_RfD.yaml
```

## Citation

We hope this code is useful for your research. If it contributes to your work, please
consider citing:

```bibtex
@inproceedings{yin2026physcore,
  title     = {PhysCoRe: Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics},
  author    = {Yin, Haocheng and Tao, Shuohan and Chen, Yongsheng and Gan, Lu},
  booktitle = {Conference on Robot Learning (CoRL)},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026}
}
```
