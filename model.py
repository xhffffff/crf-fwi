import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
import math

# from dit import DiT
from unet import  UnetConcat as Unet


class LogitNormalCosineScheduler:
    """
    Combined Logit-Normal timestep sampling + Cosine interpolation scheduling.
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0, min_t: float = 1e-4, max_t: float = 1.0 - 1e-4):
        self.loc = loc
        self.scale = scale
        self.min_t = min_t
        self.max_t = max_t
        base_normal = Normal(loc, scale)
        self.logit_normal = TransformedDistribution(base_normal, SigmoidTransform())

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        t = self.logit_normal.sample((batch_size,)).to(device)
        t = torch.clamp(t, self.min_t, self.max_t)
        return t

    def get_cosine_schedule_params(self, t: torch.Tensor, sigma_min: float = 1e-6) -> tuple:
        t_cos = 0.5 * (1 - torch.cos(math.pi * t))
        alpha_t = t_cos
        sigma_t = 1 - t_cos * (1 - sigma_min)
        return alpha_t, sigma_t

    def get_velocity_target(self, x1: torch.Tensor, z: torch.Tensor, sigma_min: float = 1e-6) -> torch.Tensor:
        u = x1 - (1 - sigma_min) * z
        return u

    def create_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        t_span = torch.linspace(0, 1, num_steps + 1, device=device)
        t_span = 0.5 * (1 - torch.cos(math.pi * t_span))
        return t_span

    create_cosine_schedule = create_schedule


class LinearScheduler:
    """
    Linear (uniform) timestep schedule.
    All steps have equal dt = 1/N. No endpoint compression.
    """

    def __init__(self, min_t: float = 1e-4, max_t: float = 1.0 - 1e-4):
        self.min_t = min_t
        self.max_t = max_t

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        t = torch.rand(batch_size, device=device)
        t = torch.clamp(t, self.min_t, self.max_t)
        return t

    def get_cosine_schedule_params(self, t: torch.Tensor, sigma_min: float = 1e-6) -> tuple:
        alpha_t = t
        sigma_t = 1 - t * (1 - sigma_min)
        return alpha_t, sigma_t

    def get_velocity_target(self, x1: torch.Tensor, z: torch.Tensor, sigma_min: float = 1e-6) -> torch.Tensor:
        u = x1 - (1 - sigma_min) * z
        return u

    def create_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(0, 1, num_steps + 1, device=device)

    create_cosine_schedule = create_schedule


def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5
use_median_correction=False,

class RectifiedFlow(nn.Module):
    def __init__(
            self,
            net: Unet,
            device="cuda",
            channels=3,
            image_size=32,
            logit_normal_sampling_t=True,
            use_logit_normal_cosine=True,
            schedule_type="cosine",
            logit_normal_loc=0.0,
            logit_normal_scale=1.0,
            timestep_min=1e-4,
            timestep_max=1.0 - 1e-4,
    ):
        super().__init__()
        self.net = net
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.logit_normal_sampling_t = logit_normal_sampling_t
        self.use_logit_normal_cosine = use_logit_normal_cosine
        self.schedule_type = schedule_type

        self._init_scheduler(timestep_min, timestep_max, logit_normal_loc, logit_normal_scale)

    def _init_scheduler(self, t_min, t_max, loc, scale):
        if self.schedule_type == "cosine":
            self.scheduler = LogitNormalCosineScheduler(
                loc=loc, scale=scale, min_t=t_min, max_t=t_max
            )
        elif self.schedule_type == "linear":
            self.scheduler = LinearScheduler(min_t=t_min, max_t=t_max)
        else:
            self.scheduler = LinearScheduler(min_t=t_min, max_t=t_max)

    def forward(self, x, c=None):
        pass

    def get_timestep_schedule(self, sample_steps: int, schedule_type: str = None):
        st = schedule_type or self.schedule_type
        if st == "linear":
            return torch.linspace(0, 1, sample_steps + 1, device=self.device)
        else:
            return self.scheduler.create_schedule(sample_steps, self.device)

    def _integration_step(self, z, t, dt, seis):
        v_t = self.net(z, t, seis=seis)
        z_next = z + dt * v_t
        return z_next, v_t

    @torch.no_grad()
    def sample(self, batch_size=None,  sample_steps=10,
               return_all_steps=False, seis=None, schedule_type=None,
               use_median_correction=use_median_correction):
        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)

        images = []
        t_span = self.get_timestep_schedule(sample_steps, schedule_type=schedule_type)

        t = t_span[0]
        dt = t_span[1] - t_span[0]
        x1_estimates = []

        for step in range(1, len(t_span)):
            z_old = z
            z, v_t = self._integration_step(z, t, dt, seis)

            if use_median_correction and 0.1 <= float(t) <= 0.95:
                x1_est = z_old + (1 - float(t)) * v_t
                x1_estimates.append(x1_est)

            t = t + dt

            if return_all_steps:
                images.append(z.clone())

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        if use_median_correction and len(x1_estimates) >= 2:
            stacked = torch.stack(x1_estimates, dim=0)
            z_final = stacked.median(dim=0).values.clip(-1, 1)
        else:
            z_final = z.clip(-1, 1)

        if return_all_steps:
            return z_final, torch.stack(images)
        return z_final

    @torch.no_grad()
    def sample_each_class(self, n_per_class, sample_steps=10,
                          return_all_steps=False, seis=None, schedule_type=None,
                          use_median_correction=use_median_correction):
        if not self.use_cond:
            raise ValueError("Cannot sample each class when num_classes is None")

        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        z = torch.randn(self.num_classes * n_per_class, self.channels, self.image_size, self.image_size,
                        device=self.device)

        images = [z.clone()] if return_all_steps else []
        t_span = self.get_timestep_schedule(sample_steps, schedule_type=schedule_type)

        t = t_span[0]
        dt = t_span[1] - t_span[0]
        x1_estimates = []


        for step in range(1, len(t_span)):
            z_old = z
            z, v_t = self._integration_step(z, t, dt, seis)

            if use_median_correction and 0.1 <= float(t) <= 0.95:
                x1_est = z_old + (1 - float(t)) * v_t
                x1_estimates.append(x1_est)

            t = t + dt

            if return_all_steps:
                images.append(z.clone())

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        if use_median_correction and len(x1_estimates) >= 2:
            stacked = torch.stack(x1_estimates, dim=0)
            z_final = stacked.median(dim=0).values.clip(-1, 1)
        else:
            z_final = z.clip(-1, 1)

        if return_all_steps:
            return z_final, torch.stack(images)
        return z_final

    @classmethod
    def from_checkpoint(cls, checkpoint_path, net, device="cuda"):
        """
        Create RectifiedFlow sampler from training checkpoint.
        Automatically loads the correct scheduler parameters.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})

        # Extract scheduler parameters from checkpoint
        use_logit_normal_cosine = config.get('timestep_sampling') == 'logit_normal_cosine'

        sampler = cls(
            net=net,
            device=device,
            channels=config.get('image_channels', 3),
            image_size=config.get('image_size', 32),
            use_logit_normal_cosine=use_logit_normal_cosine,
            logit_normal_loc=config.get('logit_normal_loc', 0.0),
            logit_normal_scale=config.get('logit_normal_scale', 1.0),
            timestep_min=config.get('timestep_min', 1e-4),
            timestep_max=config.get('timestep_max', 1.0 - 1e-4),
        )

        return sampler



# Example usage
if __name__ == "__main__":
    # Test the fixed implementation
    from unet import Unet

    model = Unet(
        channels=1,
        dim=64,
        dim_mults=(1, 2, 4, 8),
    )

    # Create sampler with logit-normal + cosine scheduling
    sampler = RectifiedFlow(
        net=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        channels=3,
        image_size=64,
        use_logit_normal_cosine=True,
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
    )

    # Test sampling
    with torch.no_grad():
        samples = sampler.sample(batch_size=4, sample_steps=20)
        print(f"Generated samples shape: {samples.shape}")

        class_samples = sampler.sample_each_class(n_per_class=2, sample_steps=20)
        print(f"Class samples shape: {class_samples.shape}")