import os
import sys
import subprocess
import argparse
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_7d")

# 全部约束
ALL_C = {'cont', 'thermo', 'mom', 'energy', 'phase', 'covar'}

#实验分组清单
CORE_EXPS = [
    ("lam1e-1_seed42", 42, True, 0.1, True, 3, None),
    ("lam1e-1_seed123", 123, True, 0.1, True, 3, None),
    ("lam1e-1_seed2024", 2024, True, 0.1, True, 3, None),
    ("seed42_nophys", 42, False, 0.0, True, 3, None),
    ("seed123_nophys", 123, False, 0.0, True, 3, None),
    ("seed2024_nophys", 2024, False, 0.0, True, 3, None),
]

ABLATION_EXPS = [
    ("lam1e-1_noaug_seed42", 42, True, 0.1, False, 3, None),
    ("lam1e-1_noaug_seed123", 123, True, 0.1, False, 3, None),
    ("lam1e-1_noaug_seed2024", 2024, True, 0.1, False, 3, None),
    ("seed42_baseline", 42, False, 0.0, False, 3, None),
    ("seed123_baseline", 123, False, 0.0, False, 3, None),
    ("seed2024_baseline", 2024, False, 0.0, False, 3, None),
]

CONSTRAINT_EXPS = [
    ("abl2_noenergy_42", 42, True, 0.1, True, 3, ALL_C - {'energy'}),
    ("abl2_noenergy_123", 123, True, 0.1, True, 3, ALL_C - {'energy'}),
    ("abl2_noenergy_2024", 2024, True, 0.1, True, 3, ALL_C - {'energy'}),
    ("abl2_nomom_42", 42, True, 0.1, True, 3, ALL_C - {'mom'}),
    ("abl2_nomom_123", 123, True, 0.1, True, 3, ALL_C - {'mom'}),
    ("abl2_nomom_2024", 2024, True, 0.1, True, 3, ALL_C - {'mom'}),
    ("abl2_2only_42", 42, True, 0.1, True, 3, {'energy', 'mom'}),
    ("abl2_2only_123", 123, True, 0.1, True, 3, {'energy', 'mom'}),
    ("abl2_2only_2024", 2024, True, 0.1, True, 3, {'energy', 'mom'}),
]

LAMBDA_EXPS = [
    ("lam1e-2_seed42", 42, True, 0.01, True, 3, None),
    ("lam1e-2_seed123", 123, True, 0.01, True, 3, None),
    ("lam1e-2_seed2024", 2024, True, 0.01, True, 3, None),
    ("lam3e-1_seed42", 42, True, 0.3, True, 3, None),
    ("lam3e-1_seed123", 123, True, 0.3, True, 3, None),
    ("lam3e-1_seed2024", 2024, True, 0.3, True, 3, None),
]

LAYER_EXPS = [
    ("lam1e-1_L2_seed42", 42, True, 0.1, True, 2, None),
    ("lam1e-1_L2_seed123", 123, True, 0.1, True, 2, None),
    ("lam1e-1_L2_seed2024", 2024, True, 0.1, True, 2, None),
    ("lam1e-1_L4_seed42", 42, True, 0.1, True, 4, None),
    ("lam1e-1_L4_seed123", 123, True, 0.1, True, 4, None),
    ("lam1e-1_L4_seed2024", 2024, True, 0.1, True, 4, None),
]


def run_one(exp):
    exp_name, seed, physics, scale, augment, layers, constraints = exp

    # 🔧 断点续跑核心逻辑：判断文件夹下有没有 summary.npy
    summary_path = os.path.join(CKPT_DIR, exp_name, "summary.npy")
    if os.path.exists(summary_path):
        print(f" ⏭️ 跳过 {exp_name} (已存在)")
        return

    env = os.environ.copy()
    env["EXP_NAME"] = exp_name
    env["SEED"] = str(seed)
    env["USE_PHYSICS"] = str(physics)
    env["PHYSICS_SCALE"] = str(scale)
    env["USE_AUGMENT"] = str(augment)
    env["NUM_LAYERS"] = str(layers)
    env["ENABLED_CONSTRAINTS"] = "" if constraints is None else ",".join(sorted(constraints))

    print(f"\n{'=' * 60}")
    print(f"🚀 准备运行: {exp_name}")
    print(f"   参数: Seed={seed} | Physics={physics}(λ={scale}) | Aug={augment} | L={layers}")
    print(f"{'=' * 60}")

    t0 = time.time()
    try:
        subprocess.run([sys.executable, "-m", "model.train"], check=True, env=env, cwd=SCRIPT_DIR)
        print(f"✅ {exp_name} 完成，耗时 {(time.time() - t0) / 60:.1f} 分钟")
    except subprocess.CalledProcessError as e:
        print(f"❌ {exp_name} 失败！建议检查显存或代码。")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["core", "ablation", "constraint", "lambda", "layer", "all"], default="core")
    parser.add_argument("--start", type=int, default=1, help="从当前批次的第几个开始")
    args = parser.parse_args()

    if args.stage == "core":
        exps = CORE_EXPS
    elif args.stage == "ablation":
        exps = ABLATION_EXPS
    elif args.stage == "constraint":
        exps = CONSTRAINT_EXPS
    elif args.stage == "lambda":
        exps = LAMBDA_EXPS
    elif args.stage == "layer":
        exps = LAYER_EXPS
    else:
        exps = CORE_EXPS + ABLATION_EXPS + CONSTRAINT_EXPS + LAMBDA_EXPS + LAYER_EXPS

    print(f"🎯 当前阶段 [{args.stage}] 共 {len(exps)} 组实验待跑。")

    for i, exp in enumerate(exps, 1):
        if i < args.start:
            continue
        print(f"\n进度: [{i}/{len(exps)}]")
        run_one(exp)

    print(f"\n🎉 [{args.stage}] 阶段跑完！")