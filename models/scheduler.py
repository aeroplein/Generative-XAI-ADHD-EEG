import torch
from utils.logger import LoggerHelper

class DiffusionScheduler:
    """
    Docstring for DiffusionScheduler
    """

    def __init__(self, num_time_steps: int = 1000, beta_start: float=1e-4, beta_end: float=0.02, device: str="cpu"):
        self.logger = LoggerHelper.get_logger()
        self.num_time_steps = num_time_steps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_time_steps).to(device)
        self.alphas = 1.0-self.betas
        self.alpha_bars=torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars=torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars=torch.sqrt(1-self.alpha_bars)
        self.logger.info(f"DiffusionScheduler initialized with {num_time_steps} steps on {device}")

    def add_noise(self, original_signal: torch.Tensor, time_steps: torch.Tensor, noise: torch.Tensor=None):
            if noise is None:
                 noise=torch.randn_like(original_signal, device=self.device)
            sqrt_alpha_t=self.sqrt_alpha_bars[time_steps].view(-1, 1, 1)
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_bars[time_steps].view(-1, 1, 1)
            noisy_signal=(sqrt_alpha_t*original_signal)+(sqrt_one_minus_alpha_t*noise)
            return noisy_signal, noise
    