# Rendering-aware OT Distillation + Hard-Budget Reseed (Baseline)

## 1) Initialize a student with exactly **n** Gaussians
```bash
python init_student_kdtree.py \
  --teacher_model /path/to/teacher/point_cloud.ply \
  --n 10000 \
  --out /path/to/student_init/point_cloud.ply \
  --sh_degree 3 \
  --debug
```

## 2) Train the student with distillation + OT + reseed
```bash
python train_student_ot.py \
  --source_path /path/to/dataset \
  --model_path /path/to/output \
  --teacher_model /path/to/teacher/point_cloud.ply \
  --student_model /path/to/student_init/point_cloud.ply \
  --config configs/student_ot.json
```

The student is kept at exactly **n** Gaussians (no densification). The teacher stays frozen.

## 3) Evaluate (render outputs + metrics)
Use the existing renderer/eval tools, e.g.:
```bash
python render.py --model_path /path/to/output --source_path /path/to/dataset
python metrics.py --model_path /path/to/output --source_path /path/to/dataset
```
