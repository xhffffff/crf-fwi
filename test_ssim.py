import os
import glob
import re
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from unet import UnetConcat
import random
from model import RectifiedFlow

# ================= 配置区域 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_PATH = r"...\rf\cfb_100000.pth"   # rf\cfb_100000.pth
DATA_DIR = r"...\rf\cfb\test_data"   # C:\pycharm_project\paper\rf\cfb\test_data
SAVE_RESIDUAL = False
RESIDUAL_SAVE_DIR = r"...\residuals"
SAMPLE_STEPS = 10
snr_db = 1
CFG_SCALE = 1

# 🔥 SSIM 过滤阈值
SSIM_THRESHOLD = 0.0  # SSIM 低于此值的样本将被过滤

# 固定随机数种子
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

# ================= 🔥 关键修复：去除 _orig_mod. 前缀 =================
def strip_orig_mod_prefix(state_dict):
    """去除 torch.compile() 添加的 _orig_mod. 前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# ================= 指标计算函数 =================
def calculate_image_metrics(img1, img2, save_residual=False, residual_path="residual.png"):
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # [-1, 1] -> [0, 1]
    img1_np = (img1 + 1.0) / 2.0
    img2_np = (img2 + 1.0) / 2.0
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    residual = img1_np - img2_np
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))

    if img1_np.ndim == 4:
        ssim_scores = []
        for i in range(img1_np.shape[0]):
            s = ssim(img1_np[i][0], img2_np[i][0], data_range=1)
            ssim_scores.append(s)
        ssim_score = np.mean(ssim_scores)
    else:
        ssim_score = ssim(img1_np, img2_np, data_range=1)

    if save_residual and img1_np.ndim > 2:
        res_single = residual[0]
        res_min, res_max = res_single.min(), res_single.max()
        if res_max - res_min > 0:
            residual_normalized = ((res_single - res_min) / (res_max - res_min) * 255).astype(np.uint8)
            Image.fromarray(residual_normalized).save(residual_path)

    return {"MAE": mae, "RMSE": rmse, "SSIM": ssim_score}

# ================= 🔥 模型初始化 =================
print("🔧 初始化模型...")
model = UnetConcat(
        channels=1,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=1,
        seis_channels=64,
        cond_drop_prob=0.3,
        time_pool_segmented=True,
    ).to(DEVICE)

# ================= 🔥 权重加载 =================
print(f"🔍 正在加载权重：{WEIGHT_PATH}")

if not os.path.exists(WEIGHT_PATH):
    for ext in [".pth", ".pt", ".ckpt"]:
        if os.path.exists(WEIGHT_PATH + ext):
            WEIGHT_PATH += ext
            break

checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict):
    if 'ema_model' in checkpoint:
        state_dict = checkpoint['ema_model']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
else:
    state_dict = checkpoint

print("🔧 检查并去除 _orig_mod. 前缀...")
has_orig_mod = any(k.startswith('_orig_mod.') for k in state_dict.keys())
if has_orig_mod:
    print("⚠️ 检测到 torch.compile() 前缀，正在处理...")
    state_dict = strip_orig_mod_prefix(state_dict)
    print("✅ 前缀处理完成")
else:
    print("✅ 无需处理前缀")

model.load_state_dict(state_dict, strict=False)
print("🎉 模型权重加载成功！")

# ================= 🔥 创建 RectifiedFlow 采样器 =================
print("🚀 创建 RectifiedFlow 采样器...")
sampler = RectifiedFlow(
    net=model,
    device=DEVICE,
    channels=1,
    image_size=70,
    num_classes=1,
    use_logit_normal_cosine=True,
    logit_normal_loc=0.0,
    logit_normal_scale=1.0,
    timestep_min=1e-4,
    timestep_max=1.0 - 1e-4,
)

# ================= 数据加载 =================
def load_data_pairs(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    print(f"数据目录: {data_dir}")
    print(f"找到文件: {files}")

    vel_files = sorted([f for f in files if "vel" in f.lower()])
    seis_files = sorted([f for f in files if "seis" in f.lower()])

    if len(vel_files) == 0 or len(seis_files) == 0:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 vel 或 seis 文件")

    def extract_id(filename):
        match = re.search(r'(?:seis|vel)(\d+)', os.path.basename(filename).lower())
        return int(match.group(1)) if match else -1

    vel_by_id = {extract_id(f): f for f in vel_files if extract_id(f) >= 0}
    seis_by_id = {extract_id(f): f for f in seis_files if extract_id(f) >= 0}

    common_ids = sorted(set(vel_by_id.keys()) & set(seis_by_id.keys()))
    if len(common_ids) == 0:
        raise ValueError(
            f"无法匹配 seis 和 vel 文件！\n"
            f"  vel IDs: {sorted(vel_by_id.keys())}\n"
            f"  seis IDs: {sorted(seis_by_id.keys())}\n"
            f"  请确保测试数据目录中的 seis 和 vel 文件数字编号一致"
        )

    print(f"匹配到的类别ID: {common_ids}")

    seis_global_max = None
    pairs = []
    for cid in common_ids:
        v_path = vel_by_id[cid]
        s_path = seis_by_id[cid]
        print(f"  配对: {os.path.basename(s_path)} <-> {os.path.basename(v_path)}")

        vel_data = np.load(v_path)
        seis_data = np.load(s_path)

        abs_max = np.abs(seis_data).max()
        if seis_global_max is None or abs_max > seis_global_max:
            seis_global_max = abs_max

        pairs.append({
            "vel": vel_data,
            "seis": seis_data,
            "seis_raw": seis_data.copy(),
            "vel_path": v_path,
            "seis_path": s_path
        })

    if seis_global_max is None or seis_global_max == 0:
        seis_global_max = 1.0
    print(f"动态计算的 seis_global_max: {seis_global_max:.4f}")

    for pair in pairs:
        seis_data = pair.pop("seis_raw")
        pair["seis"] = seis_data / seis_global_max
        pair["seis"] = np.clip(pair["seis"], -1, 1)

        vel_data = pair["vel"]
        vel_data = (vel_data - 1500.0) / (5500.0 - 1500.0)
        vel_data = vel_data * 2.0 - 1.0
        vel_data = np.clip(vel_data, -1, 1)
        pair["vel"] = vel_data

    return pairs

data_pairs = load_data_pairs(DATA_DIR)
print(f"📂 找到 {len(data_pairs)} 对测试数据")

if SAVE_RESIDUAL and not os.path.exists(RESIDUAL_SAVE_DIR):
    os.makedirs(RESIDUAL_SAVE_DIR)

# ================= 🔥 测试循环（SSIM 过滤） =================
all_mae, all_rmse, all_ssim = [], [], []
filtered_count = 0  # 🔥 记录被过滤的样本数
total_count = 0     # 🔥 记录总样本数

from noise import add_gaussian_noise, add_uniform_noise, add_noisy_traces, add_spike_noise, add_realistic_field_noise, add_multiples

with torch.no_grad():
    for idx, pair in enumerate(data_pairs):
        print(f"\n--- 测试第 {idx + 1}/{len(data_pairs)} 组 ---")

        vel_np = pair['vel']
        seis_np = pair['seis']

        if vel_np.ndim == 3: vel_np = np.expand_dims(vel_np, axis=1)
        if seis_np.ndim == 3: seis_np = np.expand_dims(seis_np, axis=1)

        n_samples = vel_np.shape[0]
        print(f"📊 样本数量：{n_samples}, 速度形状：{vel_np.shape}, 地震形状：{seis_np.shape}")

        vel_tensor = torch.from_numpy(vel_np).float().to(DEVICE)
        seis_tensor = torch.from_numpy(seis_np).float().to(DEVICE)

        batch_mae, batch_rmse, batch_ssim = [], [], []

        BATCH_SIZE = 8
        for i in tqdm(range(0, n_samples, BATCH_SIZE), desc="Sampling"):
            end_idx = min(i + BATCH_SIZE, n_samples)
            batch_seis = seis_tensor[i:end_idx]
            # batch_seis = add_gaussian_noise(batch_seis, snr_db=snr_db)
            batch_vel_gt = vel_tensor[i:end_idx]
            batch_size = batch_seis.shape[0]

            class_labels = torch.zeros(batch_size, dtype=torch.long).to(DEVICE)

            # 🔥 使用 sampler.sample() 进行采样
            pred_vel, all_steps = sampler.sample(
                batch_size=batch_size,
                class_labels=class_labels,
                cfg_scale=CFG_SCALE,
                sample_steps=SAMPLE_STEPS,
                return_all_steps=True,
                seis=batch_seis
            )


            # 计算指标
            metrics = calculate_image_metrics(batch_vel_gt, pred_vel)
            batch_mae.append(metrics['MAE'])
            batch_rmse.append(metrics['RMSE'])
            batch_ssim.append(metrics['SSIM'])


        if len(batch_mae) > 0:
            group_mae = np.mean(batch_mae)
            group_rmse = np.mean(batch_rmse)
            group_ssim = np.mean(batch_ssim)
        else:
            group_mae, group_rmse, group_ssim = 0, 0, 0

        print(f"📈 组结果 -> MAE: {group_mae:.6f}, RMSE: {group_rmse:.6f}, SSIM: {group_ssim:.4f}")

        all_mae.append(group_mae)
        all_rmse.append(group_rmse)
        all_ssim.append(group_ssim)

import matplotlib.pyplot as plt

# ================= 随机 10 个样本可视化 =================
print("\n🔍 准备随机抽取 10 个样本进行可视化...")

# 从唯一的数据文件中提取 500 个样本
pair = data_pairs[0]
vel_np = pair['vel']
seis_np = pair['seis']

# 统一维度 -> (N, C, H, W)
if vel_np.ndim == 3:
    vel_np = np.expand_dims(vel_np, axis=1)
if seis_np.ndim == 3:
    seis_np = np.expand_dims(seis_np, axis=1)

n_total = vel_np.shape[0]
N_VIS = min(4, n_total)
vis_indices = np.random.choice(n_total, size=N_VIS, replace=False)
print(f"随机抽取索引: {vis_indices}，总样本数: {n_total}")

# 提取选中的 10 个样本
vis_vel = vel_np[vis_indices]  # (10, 1, H, W)
vis_seis = seis_np[vis_indices]


# 转 tensor 并批量推理
vel_tensor = torch.from_numpy(vis_vel).float().to(DEVICE)
seis_tensor = torch.from_numpy(vis_seis).float().to(DEVICE)
# seis_tensor = add_gaussian_noise(seis_tensor, snr_db=snr_db)
class_labels = torch.zeros(N_VIS, dtype=torch.long).to(DEVICE)

model.eval()
with torch.no_grad():
    pred_vel = sampler.sample(
        batch_size=N_VIS,
        class_labels=class_labels,
        cfg_scale=CFG_SCALE,
        sample_steps=SAMPLE_STEPS,
        return_all_steps=False,
        seis=seis_tensor
    )


# 反归一化到 [0, 1] 用于可视化
def to_img(t):
    arr = t.cpu().numpy()
    arr = (arr + 1.0) / 2.0
    return np.clip(arr, 0, 1)


gt_imgs = to_img(vel_tensor)[:, 0]  # (10, H, W)
pred_imgs = to_img(pred_vel)[:, 0]  # (10, H, W)
res_imgs = np.abs(gt_imgs - pred_imgs)  # (10, H, W)

# 地震数据逐样本归一化到 [0,1]（仅用于可视化，保留道集相对形态）
seis_imgs = seis_tensor.cpu().numpy()[:, 0]
seis_imgs_vis = []
for i in range(N_VIS):
    s = seis_imgs[i]
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    seis_imgs_vis.append(s)
seis_imgs_vis = np.stack(seis_imgs_vis)

# 绘图
fig, axes = plt.subplots(N_VIS, 4, figsize=(16, 4.2 * N_VIS))
if N_VIS == 1:
    axes = axes.reshape(1, -1)

for row in range(N_VIS):
    idx = vis_indices[row]

    # 1. 地震输入
    ax = axes[row, 0]
    ax.imshow(seis_imgs_vis[row], aspect='auto', cmap='seismic', interpolation='nearest')
    ax.set_title(f'Sample {idx} | Seismic Input', fontsize=10)
    ax.axis('off')

    # 2. 真实速度 + 3. 预测速度：统一颜色条
    gt_row = gt_imgs[row]      # (H, W)
    pred_row = pred_imgs[row]  # (H, W)
    # 计算统一范围
    vmin = min(gt_row.min(), pred_row.min())
    vmax = max(gt_row.max(), pred_row.max())

    ax = axes[row, 1]
    im1 = ax.imshow(gt_row, aspect='auto', cmap='jet', interpolation='nearest',
                    vmin=vmin, vmax=vmax)
    ax.set_title('GT Velocity', fontsize=10)
    ax.axis('off')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[row, 2]
    im2 = ax.imshow(pred_row, aspect='auto', cmap='jet', interpolation='nearest',
                    vmin=vmin, vmax=vmax)
    ax.set_title('Pred Velocity', fontsize=10)
    ax.axis('off')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # 4. 绝对残差
    ax = axes[row, 3]
    im3 = ax.imshow(res_imgs[row], aspect='auto', cmap='hot', interpolation='nearest')
    mae_val = np.mean(res_imgs[row])
    ax.set_title(f'|Residual| (MAE:{mae_val:.4f})', fontsize=10)
    ax.axis('off')
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f'Random {N_VIS} Samples Visualization (Steps={SAMPLE_STEPS}, CFG={CFG_SCALE})',
             fontsize=14, y=1.00)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(WEIGHT_PATH), 'vis.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"✅ 可视化已保存: {save_path}")
plt.show()
# ================= 可视化结束 =================


# ================= 汇总 =================
print("\n" + "=" * 50)
print("🎉 测试完成")
print("=" * 50)
print(f"📊 总样本数：{total_count + filtered_count}")

print("=" * 50)
print(f"平均 MAE : {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
print(f"平均 RMSE: {np.mean(all_rmse):.6f} ± {np.std(all_rmse):.6f}")
print(f"平均 SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")

print("=" * 50)