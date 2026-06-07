from unet import UnetConcat
import torch
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import random
from model import RectifiedFlow
from torch.utils.data import DataLoader
from comet_ml import Experiment
import os
import torch.optim as optim
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from utils.test_data_slicer import load_test_batch_direct
from utils.OptimizedSeisDataset import OptimizedSeisDataset
from ssim_improments import LearningRateController


def calculate_image_metrics(img1, img2):
    """计算两张图像的MAE、RMSE、SSIM指标"""
    img1_np = (img1.cpu().numpy() + 1.0) / 2.0
    img2_np = (img2.cpu().numpy() + 1.0) / 2.0
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    residual = img1_np - img2_np
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))
    ssim_score = ssim(img1_np, img2_np, data_range=1)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "SSIM": ssim_score
    }


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


def main():
    n_steps = 100000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    image_size = 70
    image_channels = 1
    num_classes = 1

    os.makedirs('/images_openfwi/', exist_ok=True)
    os.makedirs('/results_openfwi/', exist_ok=True)
    checkpoint_root_path = r'/checkpoint/'
    os.makedirs(checkpoint_root_path, exist_ok=True)

    experiment = Experiment(
        api_key='8QcemfBuqZM3Pmrj4vWvMf8fi',
        project_name="cfb-test"
    )

    unet_path = os.path.join(os.path.dirname(__file__), "unet.py")
    if os.path.exists(unet_path):
        experiment.log_code(file_name=unet_path)

    folder = r"/openfwi/data"

    ds = OptimizedSeisDataset(folder=folder, cache_strategy='memory_map')
    print(f"数据集总样本数：{len(ds)}")
    sample_x, sample_y, sample_seis = ds[0]
    print(f"样本x形状：{sample_x.shape}")
    print(f"样本seis形状：{sample_seis.shape}")
    train_sampler = torch.utils.data.RandomSampler(ds)
    train_dataloader = DataLoader(ds,
                                  batch_size=batch_size,
                                  num_workers=8,
                                  pin_memory=True,
                                  prefetch_factor=4,
                                  persistent_workers=True,
                                  sampler=train_sampler)

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = cycle(train_dataloader)

    model = UnetConcat(
        channels=1,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=num_classes,
        seis_channels=64,
        cond_drop_prob=0.3,
    ).to(device)

    model = torch.compile(model)

    ema_model = deepcopy(model).eval()
    ema_decay = 0.9999

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    lr_controller = LearningRateController(
        initial_lr=3e-4,
        warmup_steps=20000,
        cosine_period=80000,
        min_lr_ratio=0.3,
        restart_ratio=0.5,
        decay_ratio=1.0
    )

    def lr_lambda_wrapper(step):
        return lr_controller.get_lr(step) / lr_controller.initial_lr

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda_wrapper)

    sampler = RectifiedFlow(
        ema_model,
        device=device,
        channels=image_channels,
        image_size=image_size,
        use_logit_normal_cosine=True,
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
        timestep_min=1e-8,
        timestep_max=1.0 - 1e-8,
    ).to(device)

    scaler = torch.cuda.amp.GradScaler()

    start_step = 0
    print("从头开始训练")

    def update_ema(ema_model, model, decay):
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.lerp_(p.data, 1 - decay)

    def sample_and_log_images(step, save_images=False):
        """采样并记录图像 - 评估训练集和测试集的SSIM"""
        print(f"Sampling images at step {step}...")

        ema_model.eval()
        with torch.no_grad():
            train_data = next(train_dataloader)
            train_x = train_data[0][:2].to(device)
            train_y = train_data[1][:2].to(device)
            train_seis = train_data[2][:2].to(device)

            train_ssim_list = []
            for i in range(2):
                x_gt = train_x[i:i + 1]
                seis_single = train_seis[i:i + 1]

                samples = sampler.sample_each_class(1, return_all_steps=False, seis=seis_single)
                pred = samples[0].squeeze(0).squeeze(0)
                gt = x_gt.squeeze(0).squeeze(0).squeeze(0)

                metrics = calculate_image_metrics(pred, gt)
                train_ssim_list.append(metrics["SSIM"])

            train_ssim_result = np.mean([s.item() if hasattr(s, 'item') else float(s) for s in train_ssim_list])
            print(f"Train SSIM: {train_ssim_result:.4f}")
            experiment.log_metric("train_avg/SSIM", train_ssim_result, step=step)

            test_data_dir = r"/openfwi/data/test_data"

            seis_scaling_factor = ds.seis_global_max
            seis_batch, vel_batch, sample_ids = load_test_batch_direct(
                test_data_dir=test_data_dir,
                device=device,
                seis_global_max=seis_scaling_factor
            )

            test_ssim_list = []
            test_mae_list = []
            test_rmse_list = []

            for i, sample_id in enumerate(sample_ids):
                seis_single = seis_batch[i:i + 1]
                vel_gt = vel_batch[i:i + 1]

                samples = sampler.sample_each_class(1, return_all_steps=False, seis=seis_single)
                pred = samples[0].squeeze(0).squeeze(0)
                gt = vel_gt.squeeze(0).squeeze(0).squeeze(0)

                gt = (gt - 1500.0) / (5500.0 - 1500.0)
                gt = gt * 2.0 - 1.0

                metrics = calculate_image_metrics(pred, gt)
                test_ssim_list.append(metrics["SSIM"])
                test_mae_list.append(metrics["MAE"])
                test_rmse_list.append(metrics["RMSE"])

                experiment.log_metric(
                    name=f"test_ssim/{sample_id}",
                    value=metrics["SSIM"].item() if hasattr(metrics["SSIM"], 'item') else float(metrics["SSIM"]),
                    step=step,
                )

            test_ssim_result = np.mean([s.item() if hasattr(s, 'item') else float(s) for s in test_ssim_list])
            test_mae = np.mean([m.item() if hasattr(m, 'item') else float(m) for m in test_mae_list])
            test_rmse = np.mean([r.item() if hasattr(r, 'item') else float(r) for r in test_rmse_list])

            print(f"Test SSIM: {test_ssim_result:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

            experiment.log_metric("test_avg/SSIM", test_ssim_result, step=step)
            experiment.log_metric("test_avg/MAE", test_mae, step=step)
            experiment.log_metric("test_avg/RMSE", test_rmse, step=step)

        return train_ssim_result, test_ssim_result

    losses = []
    sigma_min = 1e-06
    training_cfg_rate = 0.2
    use_immiscible = True
    gradient_clip = 1.0

    with tqdm(range(start_step, n_steps), dynamic_ncols=True, initial=start_step, total=n_steps) as pbar:
        pbar.set_description("Training Unet on openfwi")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()

            x1 = data[0].to(device)
            seis = data[2].to(device)
            b = x1.shape[0]

            t = torch.rand(b, device=device)

            alpha_t = t.view(b, 1, 1, 1)
            sigma_t = (1 - (1 - sigma_min) * t).view(b, 1, 1, 1)

            if use_immiscible:
                k = 4
                z_candidates = torch.randn(b, k, image_channels, image_size, image_size, device=x1.device,
                                           dtype=x1.dtype)
                x1_flat = x1.flatten(start_dim=1)
                z_candidates_flat = z_candidates.flatten(start_dim=2)
                distances = torch.norm(x1_flat.unsqueeze(1) - z_candidates_flat, dim=2)
                min_distances, min_indices = torch.min(distances, dim=1)
                batch_indices = torch.arange(b, device=x1.device)
                z = z_candidates[batch_indices, min_indices]
            else:
                z = torch.randn_like(x1)

            x_t = sigma_t * z + alpha_t * x1
            u_positive = x1 - (1 - sigma_min) * z

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model.forward(x_t, t, seis=seis)

                l1_loss = F.l1_loss(pred, u_positive)
                loss = l1_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            update_ema(ema_model, model, ema_decay)

            loss_val = loss.item()
            grad_val = grad_norm.item()
            losses.append(loss_val)
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": loss_val,
                "grad_norm": grad_val,
                "lr": f"{current_lr:.2e}"
            })
            if step % 10 == 0:
                experiment.log_metric("loss", loss_val, step=step)
                experiment.log_metric("grad_norm", grad_val, step=step)
                experiment.log_metric("learning_rate", current_lr, step=step)
                avg_loss = sum(losses[-10:]) / min(10, len(losses))
                experiment.log_metric("avg_loss_10", avg_loss, step=step)

            if step % 1000 == 0 and step >= 96000:
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"\nStep: {step + 1}/{n_steps} | avg_loss: {avg_loss:.4f}")
                losses.clear()
                _, _ = sample_and_log_images(step, save_images=False)

            if step >= n_steps - 1:
                print(f"\n训练完成：已运行 {step + 1} 步")
                break

            if step % 10000 == 0 and step > 0:
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"\nStep: {step + 1}/{n_steps} | avg_loss: {avg_loss:.4f}")
                losses.clear()
                _, _ = sample_and_log_images(step, save_images=False)

    # ============================================================
    # 训练循环结束后：只保存最终权重
    # ============================================================
    final_step = n_steps  # 或者 step + 1，如果你需要实际步数
    checkpoint_path = os.path.join(checkpoint_root_path, f"model_fwi_step_{final_step}.pth")
    state_dict = {
        "CureFaultB": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "config": {
            "dataset": "openfwi",
            "image_size": image_size,
            "image_channels": image_channels,
            "sigma_min": sigma_min,
            "training_cfg_rate": training_cfg_rate,
            "gradient_clip": gradient_clip,
            "ema_decay": ema_decay,
        }
    }
    torch.save(state_dict, checkpoint_path)
    experiment.log_model(name=f"model_step_{final_step}", file_or_folder=checkpoint_path)
    experiment.end()
    print(f"最终权重已保存: {checkpoint_path}")


if __name__ == "__main__":
    main()