## Rendering-aware OT Distillation + Hard-Budget Reseed (Implementation Plan)

### Code map (confirmed in repo)
- **Gaussian parameter storage**: `scene/gaussian_model.py`
  - Positions: `GaussianModel._xyz`
  - Rotation (quaternion): `GaussianModel._rotation` (normalized via `rotation_activation`)
  - Scale (log space): `GaussianModel._scaling` (activated by `exp`)
  - Opacity (logit): `GaussianModel._opacity` (activated by `sigmoid`)
  - SH/color: `GaussianModel._features_dc`, `GaussianModel._features_rest`
  - 3D covariance from scale+rotation: `GaussianModel.get_covariance()` → `utils.general_utils.build_scaling_rotation`
- **Rasterizer interface**: `gaussian_renderer/__init__.py`
  - `render(...)` builds `GaussianRasterizationSettings` and calls `diff_gaussian_rasterization.GaussianRasterizer`
  - Returns: `render`, `viewspace_points`, `visibility_filter`, `radii`, `depth`
- **CUDA rasterizer**: `submodules/diff-gaussian-rasterization`
  - Tile duplication and sorting:
    - `cuda_rasterizer/rasterizer_impl.cu`: `duplicateWithKeys`, `identifyTileRanges`
    - `cuda_rasterizer/forward.cu`: `renderCUDA` does front-to-back alpha compositing
  - 2D covariance projection + conic params:
    - `cuda_rasterizer/forward.cu`: `computeCov2D`, `conic_opacity` packing
- **Training entry**: `train.py`
  - L1 + DSSIM loss: `utils/loss_utils.py` (`l1_loss`, `ssim`)
  - Pipeline/optimization params: `arguments/__init__.py`

### Step-by-step plan (baseline, minimal changes)
1. **KD-tree student init (new script)**  
   - Add `init_student_kdtree.py` to load teacher PLY (`GaussianModel.load_ply`), build a KD-tree partition of `teacher._xyz` (CPU), and create exactly `n` clusters (median splits by widest axis).  
   - For each cluster, compute weighted averages (weights = teacher opacity) of:
     - Mean (`xyz`)  
     - 3D covariance (`L @ L^T` from scale+rotation, then eigen-decompose to get rotation+scale)  
     - SH coefficients (`features_dc`, `features_rest`)  
     - Opacity (average in sigmoid space, convert to logit)  
   - Emit a student PLY in the repo’s standard format.

2. **Renderer stats for OT (CUDA change + Python wrapper)**  
   - Extend the CUDA rasterizer to accumulate **per-tile per-instance mass** during `renderCUDA`:
     - For each pixel’s contribution, `atomicAdd(mass[instance_idx], alpha * T)`.
   - Return the following extra stats to Python:
     - `point_list` (per-tile duplicated Gaussian indices)
     - `ranges` (start/end per tile)
     - `instance_mass` (per duplicated instance)
   - Implement `render_with_stats(...)` in `gaussian_renderer/__init__.py` using a new `GaussianRasterizerWithStats` wrapper in `diff_gaussian_rasterization/__init__.py`.

3. **Student training (new script)**  
   - Add `train_student_ot.py`:
     - Load dataset cameras via `Scene`, load teacher and student PLYs as `GaussianModel` (teacher frozen).
     - Render teacher and student per-view using `render_with_stats`.
     - Distillation loss: reuse `l1_loss` + `ssim` with teacher render as target.
     - OT loss: per-tile Sinkhorn divergence using `geomloss.SamplesLoss` on feature vectors
       `z = [u_x, u_y, log_area]` (projected in Python using camera matrices).
     - Total loss `L = L_img + beta * L_ot + gamma * reg` (minimal opacity/scale regularizers).
     - **No densification**; maintain exactly `n` Gaussians.

4. **Hard-budget reseed (training-time)**  
   - Maintain EMA of per-Gaussian mass (scatter-add from per-instance mass).
   - Every `K` iterations, replace the bottom `D` Gaussians with teacher parameters from high-residual tiles
     (residual = `|I_S - I_T|`, tile-aggregated).
   - Reset Adam moments for replaced indices to avoid stale momentum.

5. **Config + docs**  
   - Add a JSON config in `configs/student_ot.json` for `n`, `beta`, `gamma`, `K`, `top_k`, Sinkhorn params, and reseed settings.
   - Provide a minimal usage doc describing init, training, and evaluation commands.
