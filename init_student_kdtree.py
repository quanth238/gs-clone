#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn

from scene.gaussian_model import GaussianModel


@dataclass
class ClusterStats:
    count: int
    weight_sum: float


def normalize_quaternion(q):
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-8, None)


def quaternion_to_rotation(q):
    q = normalize_quaternion(q)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot = np.zeros((q.shape[0], 3, 3), dtype=np.float32)
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


def rotation_matrix_to_quaternion(rot):
    q = np.zeros((rot.shape[0], 4), dtype=np.float32)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    for i in range(rot.shape[0]):
        if trace[i] > 0:
            s = 0.5 / np.sqrt(trace[i] + 1.0)
            q[i, 0] = 0.25 / s
            q[i, 1] = (rot[i, 2, 1] - rot[i, 1, 2]) * s
            q[i, 2] = (rot[i, 0, 2] - rot[i, 2, 0]) * s
            q[i, 3] = (rot[i, 1, 0] - rot[i, 0, 1]) * s
        else:
            if rot[i, 0, 0] > rot[i, 1, 1] and rot[i, 0, 0] > rot[i, 2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot[i, 0, 0] - rot[i, 1, 1] - rot[i, 2, 2])
                q[i, 0] = (rot[i, 2, 1] - rot[i, 1, 2]) / s
                q[i, 1] = 0.25 * s
                q[i, 2] = (rot[i, 0, 1] + rot[i, 1, 0]) / s
                q[i, 3] = (rot[i, 0, 2] + rot[i, 2, 0]) / s
            elif rot[i, 1, 1] > rot[i, 2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot[i, 1, 1] - rot[i, 0, 0] - rot[i, 2, 2])
                q[i, 0] = (rot[i, 0, 2] - rot[i, 2, 0]) / s
                q[i, 1] = (rot[i, 0, 1] + rot[i, 1, 0]) / s
                q[i, 2] = 0.25 * s
                q[i, 3] = (rot[i, 1, 2] + rot[i, 2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rot[i, 2, 2] - rot[i, 0, 0] - rot[i, 1, 1])
                q[i, 0] = (rot[i, 1, 0] - rot[i, 0, 1]) / s
                q[i, 1] = (rot[i, 0, 2] + rot[i, 2, 0]) / s
                q[i, 2] = (rot[i, 1, 2] + rot[i, 2, 1]) / s
                q[i, 3] = 0.25 * s
    return normalize_quaternion(q)


def build_covariance(scales, rotations):
    rot = quaternion_to_rotation(rotations)
    diag = np.zeros((scales.shape[0], 3, 3), dtype=np.float32)
    diag[:, 0, 0] = scales[:, 0]
    diag[:, 1, 1] = scales[:, 1]
    diag[:, 2, 2] = scales[:, 2]
    L = np.matmul(rot, diag)
    return np.matmul(L, np.transpose(L, (0, 2, 1)))


def inverse_sigmoid(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def kd_tree_partition(points, target_leaves):
    leaves: List[np.ndarray] = [np.arange(points.shape[0])]
    while len(leaves) < target_leaves:
        sizes = np.array([len(idx) for idx in leaves])
        split_idx = int(np.argmax(sizes))
        indices = leaves.pop(split_idx)
        if len(indices) <= 1:
            leaves.append(indices)
            if len(leaves) == target_leaves:
                break
            continue

        pts = points[indices]
        spread = np.ptp(pts, axis=0)
        axis = int(np.argmax(spread))
        order = indices[np.argsort(pts[:, axis])]
        mid = len(order) // 2
        leaves.append(order[:mid])
        leaves.append(order[mid:])
    return leaves


def build_student_from_clusters(teacher, clusters, min_scale=1e-4):
    xyz = teacher.get_xyz.detach().cpu().numpy()
    opacity = teacher.get_opacity.detach().cpu().numpy().reshape(-1, 1)
    scales = teacher.get_scaling.detach().cpu().numpy()
    rotations = teacher.get_rotation.detach().cpu().numpy()
    features_dc = teacher.get_features_dc.detach().cpu().numpy()
    features_rest = teacher.get_features_rest.detach().cpu().numpy()

    cov3d = build_covariance(scales, rotations)

    cluster_xyz = []
    cluster_scales = []
    cluster_rots = []
    cluster_opacity = []
    cluster_dc = []
    cluster_rest = []
    stats = []

    for indices in clusters:
        weights = opacity[indices]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            weights = np.ones_like(weights)
            weight_sum = float(np.sum(weights))
        w = weights / weight_sum

        mean = np.sum(xyz[indices] * w, axis=0)
        cov = np.sum(cov3d[indices] * w[:, None, None], axis=0)

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, min_scale ** 2, None)
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 0] *= -1
        scales_cluster = np.sqrt(eigvals)
        rot_cluster = rotation_matrix_to_quaternion(eigvecs[None])[0]

        cluster_xyz.append(mean)
        cluster_scales.append(scales_cluster)
        cluster_rots.append(rot_cluster)
        cluster_opacity.append(np.sum(opacity[indices] * w, axis=0))
        cluster_dc.append(np.sum(features_dc[indices] * w[:, None, None], axis=0))
        cluster_rest.append(np.sum(features_rest[indices] * w[:, None, None], axis=0))
        stats.append(ClusterStats(count=len(indices), weight_sum=weight_sum))

    return (
        np.stack(cluster_xyz, axis=0),
        np.stack(cluster_scales, axis=0),
        np.stack(cluster_rots, axis=0),
        np.stack(cluster_opacity, axis=0),
        np.stack(cluster_dc, axis=0),
        np.stack(cluster_rest, axis=0),
        stats,
    )


def save_student_ply(out_path, sh_degree, xyz, scales, rotations, opacity, features_dc, features_rest):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gaussians = GaussianModel(sh_degree)
    gaussians.active_sh_degree = sh_degree
    gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._features_rest = nn.Parameter(torch.tensor(features_rest, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._opacity = nn.Parameter(torch.tensor(inverse_sigmoid(opacity), dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.tensor(np.log(scales), dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.tensor(rotations, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians.save_ply(out_path)


def main():
    parser = argparse.ArgumentParser(description="KD-tree init for student Gaussians.")
    parser.add_argument("--teacher_model", required=True, help="Path to teacher PLY.")
    parser.add_argument("--n", type=int, required=True, help="Target number of Gaussians.")
    parser.add_argument("--out", required=True, help="Output PLY path for the student.")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree of the teacher/student.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    teacher = GaussianModel(args.sh_degree)
    teacher.load_ply(args.teacher_model)

    points = teacher.get_xyz.detach().cpu().numpy()
    clusters = kd_tree_partition(points, args.n)
    if len(clusters) != args.n:
        raise RuntimeError(f"KD-tree produced {len(clusters)} clusters, expected {args.n}.")

    xyz, scales, rotations, opacity, features_dc, features_rest, stats = build_student_from_clusters(teacher, clusters)

    if args.debug:
        sizes = [s.count for s in stats]
        print(f"Cluster sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.2f}")
        print(f"Opacity stats: min={opacity.min():.4f} max={opacity.max():.4f}")
        meta = {"cluster_sizes": sizes}
        with open(os.path.splitext(args.out)[0] + "_clusters.json", "w") as f:
            json.dump(meta, f, indent=2)

    save_student_ply(args.out, args.sh_degree, xyz, scales, rotations, opacity, features_dc, features_rest)
    print(f"Saved student PLY to {args.out}")


if __name__ == "__main__":
    main()
