import torch
import torch.nn as nn

class DiffusionEngine:
    """
    Docstring for DiffusionEngine
    """

    def __init__(self, model, scheduler):
        self.model=model
        self.scheduler=scheduler
        self.criterion=nn.MSELoss()

    def compute_loss(self, clean_signal,  reduction='mean'):
        """
        Docstring for compute_loss
        
        :param self: Description
        :param clean_signal: Description
        :param reduction: Description
        """

        batch_size = clean_signal.shape[0]
        device=clean_signal.device

        t=torch.randint(0, self.scheduler.num_time_steps, (batch_size,), device=device).long()

        noisy_signal, real_noise = self.scheduler.add_noise(clean_signal, t)

        predicted_noise = self.model(noisy_signal, t)

        loss=self.criterion(predicted_noise, real_noise)

        return loss