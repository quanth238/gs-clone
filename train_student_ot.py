#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from typing import Dict, Tuple

import torch
from torch import nn

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render_with_stats
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except Exception:
    GEOMLOSS_AVAILABLE = False

BLOCK_X = 16
BLOCK_Y = 16


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def setup_exposure(gaussians: GaussianModel, cameras):
    gaussians.exposure_mapping = {cam.image_name: idx for idx, cam in enumerate(cameras)}
    gaussians.pretrained_exposures = None
    exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cameras), 1, 1)
    gaussians._exposure = nn.Parameter(exposure.requires_grad_(True))


def build_rotation(q):
    norm = torch.sqrt((q * q).sum(dim=1, keepdim=True))
    q = q / norm.clamp(min=1e-8)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def build_cov3d(scales, rotations):
    rot = build_rotation(rotations)
    diag = torch.zeros((scales.shape[0], 3, 3), device=scales.device, dtype=scales.dtype)
    diag[:, 0, 0] = scales[:, 0]
    diag[:, 1, 1] = scales[:, 1]
    diag[:, 2, 2] = scales[:, 2]
    L = torch.bmm(rot, diag)
    return torch.bmm(L, L.transpose(1, 2))


def transform_point4x3(points, matrix):
    x = points[:, 0] * matrix[0, 0] + points[:, 1] * matrix[1, 0] + points[:, 2] * matrix[2, 0] + matrix[3, 0]
    y = points[:, 0] * matrix[0, 1] + points[:, 1] * matrix[1, 1] + points[:, 2] * matrix[2, 1] + matrix[3, 1]
    z = points[:, 0] * matrix[0, 2] + points[:, 1] * matrix[1, 2] + points[:, 2] * matrix[2, 2] + matrix[3, 2]
    return torch.stack([x, y, z], dim=1)


def transform_point4x4(points, matrix):
    x = points[:, 0] * matrix[0, 0] + points[:, 1] * matrix[1, 0] + points[:, 2] * matrix[2, 0] + matrix[3, 0]
    y = points[:, 0] * matrix[0, 1] + points[:, 1] * matrix[1, 1] + points[:, 2] * matrix[2, 1] + matrix[3, 1]
    z = points[:, 0] * matrix[0, 2] + points[:, 1] * matrix[1, 2] + points[:, 2] * matrix[2, 2] + matrix[3, 2]
    w = points[:, 0] * matrix[0, 3] + points[:, 1] * matrix[1, 3] + points[:, 2] * matrix[2, 3] + matrix[3, 3]
    return torch.stack([x, y, z, w], dim=1)


def project_gaussians(gaussians: GaussianModel, camera) -> Tuple[torch.Tensor, torch.Tensor]:
    means = gaussians.get_xyz
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    cov3d = build_cov3d(scales, rotations)

    viewmatrix = camera.world_view_transform
    projmatrix = camera.full_proj_transform
    tanfovx = torch.tensor(math.tan(camera.FoVx * 0.5), device=means.device, dtype=means.dtype)
    tanfovy = torch.tensor(math.tan(camera.FoVy * 0.5), device=means.device, dtype=means.dtype)
    focal_x = camera.image_width / (2.0 * tanfovx)
    focal_y = camera.image_height / (2.0 * tanfovy)

    p_view = transform_point4x3(means, viewmatrix)
    p_hom = transform_point4x4(means, projmatrix)
    p_proj = p_hom[:, :3] / (p_hom[:, 3:4].clamp(min=1e-8))

    u_x = ((p_proj[:, 0] + 1.0) * camera.image_width - 1.0) * 0.5
    u_y = ((p_proj[:, 1] + 1.0) * camera.image_height - 1.0) * 0.5

    limx = 1.3 * tanfovx
    limy = 1.3 * tanfovy
    txtz = p_view[:, 0] / p_view[:, 2].clamp(min=1e-8)
    tytz = p_view[:, 1] / p_view[:, 2].clamp(min=1e-8)
    t_x = torch.clamp(txtz, -limx, limx) * p_view[:, 2]
    t_y = torch.clamp(tytz, -limy, limy) * p_view[:, 2]
    t_z = p_view[:, 2]

    J = torch.zeros((means.shape[0], 3, 3), device=means.device, dtype=means.dtype)
    J[:, 0, 0] = focal_x / t_z
    J[:, 0, 2] = -(focal_x * t_x) / (t_z * t_z)
    J[:, 1, 1] = focal_y / t_z
    J[:, 1, 2] = -(focal_y * t_y) / (t_z * t_z)

    W = torch.tensor([
        [viewmatrix[0, 0], viewmatrix[1, 0], viewmatrix[2, 0]],
        [viewmatrix[0, 1], viewmatrix[1, 1], viewmatrix[2, 1]],
        [viewmatrix[0, 2], viewmatrix[1, 2], viewmatrix[2, 2]],
    ], device=means.device, dtype=means.dtype)
    W = W.unsqueeze(0).repeat(means.shape[0], 1, 1)

    T = torch.bmm(W, J)
    cov = torch.bmm(T.transpose(1, 2), torch.bmm(cov3d.transpose(1, 2), T))
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 0, 1]
    log_area = torch.log(det.clamp(min=1e-8))

    u = torch.stack([u_x, u_y], dim=1)
    return u, log_area


def aggregate_mass(point_list, instance_mass, num_gaussians):
    mass = torch.zeros((num_gaussians,), device=instance_mass.device)
    if point_list.numel() == 0:
        return mass
    mass.scatter_add_(0, point_list.long(), instance_mass)
    return mass


def reset_optimizer_state(optimizer, indices):
    for group in optimizer.param_groups:
        param = group["params"][0]
        state = optimizer.state.get(param)
        if state is None:
            continue
        for key in ["exp_avg", "exp_avg_sq"]:
            if key in state:
                state[key][indices] = 0


def compute_tile_residuals(residual_map, tile_grid_x, tile_grid_y):
    tile_residuals = []
    for ty in range(tile_grid_y):
        for tx in range(tile_grid_x):
            y0 = ty * BLOCK_Y
            y1 = min((ty + 1) * BLOCK_Y, residual_map.shape[0])
            x0 = tx * BLOCK_X
            x1 = min((tx + 1) * BLOCK_X, residual_map.shape[1])
            tile_residuals.append(residual_map[y0:y1, x0:x1].mean())
    return torch.stack(tile_residuals)


def training(dataset, opt, pipe, config, teacher_path, student_path, debug):
    safe_state(False)
    student = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    student.load_ply(student_path)
    teacher = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    teacher.load_ply(teacher_path)
    teacher.active_sh_degree = dataset.sh_degree

    scene = Scene(dataset, GaussianModel(dataset.sh_degree))
    setup_exposure(student, scene.getTrainCameras())
    setup_exposure(teacher, scene.getTrainCameras())
    student.training_setup(opt)

    for tensor in [teacher._xyz, teacher._features_dc, teacher._features_rest, teacher._opacity, teacher._scaling, teacher._rotation]:
        tensor.requires_grad_(False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not GEOMLOSS_AVAILABLE:
        raise RuntimeError("geomloss is required for OT loss. Please install it in your environment.")

    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=config["sinkhorn"]["blur"],
        reach=config["sinkhorn"]["reach"],
        scaling=config["sinkhorn"]["scaling"],
        debias=True,
    )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_mass = torch.zeros((student.get_xyz.shape[0],), device="cuda")
    ema_decay = config["reseed"]["ema_decay"]

    for iteration in range(1, config["training"]["iterations"] + 1):
        student.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        batch = []
        for _ in range(config["training"]["batch_size"]):
            rand_idx = random.randint(0, len(viewpoint_indices) - 1)
            batch.append(viewpoint_stack.pop(rand_idx))
            viewpoint_indices.pop(rand_idx)

        total_loss = 0.0
        total_img_loss = 0.0
        total_ot_loss = 0.0
        reseed_view_data = None

        for view_idx, viewpoint_cam in enumerate(batch):
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            student_pkg = render_with_stats(viewpoint_cam, student, pipe, bg, separate_sh=False)
            with torch.no_grad():
                teacher_pkg = render_with_stats(viewpoint_cam, teacher, pipe, bg, separate_sh=False)

            student_img = student_pkg["render"]
            teacher_img = teacher_pkg["render"].detach()

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                student_img = student_img * alpha_mask
                teacher_img = teacher_img * alpha_mask

            Ll1 = l1_loss(student_img, teacher_img)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(student_img.unsqueeze(0), teacher_img.unsqueeze(0))
            else:
                ssim_value = ssim(student_img, teacher_img)
            img_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            if view_idx == 0:
                reseed_view_data = (student_img, teacher_img, teacher_pkg, viewpoint_cam)

            tile_ranges = student_pkg["ranges"]
            tile_point_list = student_pkg["point_list"]
            tile_mass = student_pkg["instance_mass"]

            teacher_ranges = teacher_pkg["ranges"]
            teacher_point_list = teacher_pkg["point_list"]
            teacher_mass = teacher_pkg["instance_mass"]

            student_u, student_log_area = project_gaussians(student, viewpoint_cam)
            with torch.no_grad():
                teacher_u, teacher_log_area = project_gaussians(teacher, viewpoint_cam)

            ranges_cpu = tile_ranges.detach().cpu().numpy()
            teacher_ranges_cpu = teacher_ranges.detach().cpu().numpy()

            tile_ot_loss = torch.tensor(0.0, device=student.get_xyz.device)
            if ranges_cpu.shape[0] > 0 and teacher_ranges_cpu.shape[0] > 0:
                for tile_id, (start, end) in enumerate(ranges_cpu):
                    if end <= start:
                        continue
                    t_start, t_end = teacher_ranges_cpu[tile_id]
                    if t_end <= t_start:
                        continue

                    masses = tile_mass[start:end]
                    t_masses = teacher_mass[t_start:t_end]
                    if masses.numel() < 2 or t_masses.numel() < 2:
                        continue

                    k = min(config["ot"]["top_k"], masses.numel())
                    tk = min(config["ot"]["top_k"], t_masses.numel())
                    mass_vals, mass_idx = torch.topk(masses, k=k, sorted=False)
                    t_mass_vals, t_mass_idx = torch.topk(t_masses, k=tk, sorted=False)

                    student_ids = tile_point_list[start:end][mass_idx].long()
                    teacher_ids = teacher_point_list[t_start:t_end][t_mass_idx].long()

                    x = torch.cat([student_u[student_ids], student_log_area[student_ids, None]], dim=1)
                    y = torch.cat([teacher_u[teacher_ids], teacher_log_area[teacher_ids, None]], dim=1)
                    alpha = mass_vals
                    beta = t_mass_vals
                    tile_ot_loss = tile_ot_loss + sinkhorn(alpha, x, beta, y)

            ot_loss = tile_ot_loss / max(1, len(ranges_cpu))

            total_img_loss = total_img_loss + img_loss
            total_ot_loss = total_ot_loss + ot_loss
            total_loss = total_loss + img_loss + config["ot"]["beta"] * ot_loss

            student_mass = aggregate_mass(tile_point_list, tile_mass, student.get_xyz.shape[0])
            ema_mass = ema_decay * ema_mass + (1 - ema_decay) * student_mass.detach()

        reg_loss = config["reg"]["gamma"] * ((student.get_scaling ** 2).mean() + (student.get_opacity ** 2).mean())
        total_loss = total_loss / len(batch) + reg_loss
        total_loss.backward()

        if iteration < config["training"]["iterations"]:
            student.optimizer.step()
            student.optimizer.zero_grad(set_to_none=True)
            student.exposure_optimizer.step()
            student.exposure_optimizer.zero_grad(set_to_none=True)

        if iteration % config["reseed"]["interval"] == 0 and reseed_view_data is not None:
            student_img, teacher_img, teacher_pkg, reseed_cam = reseed_view_data
            residual = (student_img - teacher_img).abs().mean(dim=0)
            tile_grid_x = (reseed_cam.image_width + BLOCK_X - 1) // BLOCK_X
            tile_grid_y = (reseed_cam.image_height + BLOCK_Y - 1) // BLOCK_Y
            tile_residuals = compute_tile_residuals(residual, tile_grid_x, tile_grid_y)

            reseed_count = min(config["reseed"]["count"], student.get_xyz.shape[0])
            _, worst_idx = torch.topk(ema_mass, k=reseed_count, largest=False)

            ranges = teacher_pkg["ranges"]
            point_list = teacher_pkg["point_list"]
            mass = teacher_pkg["instance_mass"]
            ranges_cpu = ranges.detach().cpu().numpy()
            if ranges_cpu.shape[0] == 0:
                continue

            _, tile_candidates = torch.topk(tile_residuals, k=min(config["reseed"]["tile_candidates"], tile_residuals.numel()))
            tile_candidates = tile_candidates.tolist()

            with torch.no_grad():
                for k_idx, student_idx in enumerate(worst_idx):
                    tile_id = tile_candidates[k_idx % len(tile_candidates)]
                    start, end = ranges_cpu[tile_id]
                    if end <= start:
                        continue
                    tile_mass = mass[start:end]
                    tile_indices = point_list[start:end]
                    if tile_mass.numel() == 0:
                        continue
                    pick = torch.argmax(tile_mass)
                    teacher_idx = tile_indices[pick].long()

                    jitter = torch.randn((3,), device=student.get_xyz.device) * config["reseed"]["jitter_xyz"]
                    student._xyz[student_idx] = teacher.get_xyz[teacher_idx] + jitter
                    student._rotation[student_idx] = teacher._rotation[teacher_idx]
                    student._scaling[student_idx] = teacher._scaling[teacher_idx]
                    student._opacity[student_idx] = teacher._opacity[teacher_idx]
                    student._features_dc[student_idx] = teacher._features_dc[teacher_idx]
                    student._features_rest[student_idx] = teacher._features_rest[teacher_idx]

                reset_optimizer_state(student.optimizer, worst_idx)

            if debug:
                print(f"[Iter {iteration}] Reseeded {reseed_count} Gaussians.")

        if iteration % config["training"]["log_interval"] == 0:
            print(f"[Iter {iteration}] loss={total_loss.item():.6f} img={total_img_loss.item():.6f} ot={total_ot_loss.item():.6f}")

        if iteration % config["training"]["save_interval"] == 0:
            save_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
            os.makedirs(save_dir, exist_ok=True)
            student.save_ply(os.path.join(save_dir, "point_cloud.ply"))

        assert student.get_xyz.shape[0] == config["budget"]["n"], "Student Gaussian count changed."


def main():
    parser = argparse.ArgumentParser(description="Train student with OT distillation and hard-budget reseed.")
    model = ModelParams(parser)
    pipe = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--teacher_model", required=True, help="Teacher PLY path.")
    parser.add_argument("--student_model", required=True, help="Student PLY path.")
    parser.add_argument("--config", default="configs/student_ot.json", help="Config JSON path.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    dataset = model.extract(args)
    opt = opt.extract(args)
    pipe = pipe.extract(args)

    config = load_config(args.config)
    config["budget"]["n"] = int(config["budget"]["n"])

    training(dataset, opt, pipe, config, args.teacher_model, args.student_model, args.debug)


if __name__ == "__main__":
    main()
