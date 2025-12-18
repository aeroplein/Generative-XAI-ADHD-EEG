import torch
import torch.nn as nn
from utils.logger import logger

class TimeEmbedding(nn.Module):

    def __init__(self, embed_dim:int):
        """
        Docstring for __init__
        
        :param self: Description
        :param embed_dim: Description
        :type embed_dim: int
        """
        super().__init__()
        # input from 1 dim a single scalar time step to embed_dim num of dimentions.
        # this is basically to give our model the capacity to encode complex info abt specific time
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    # t is in shape [batch_size] for example [32]
    # if we use like here, t.unsqueeze(-1) it changes the shape to [32, 1]
    # we need to do this to match the input features which is 1.
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t.unsqueeze(-1).float())

class D4PMDenoiser(nn.Module):

    def __init__(self, input_dim: int, time_steps:int=100, device:str="cpu", weights_path:str=None):
        # here we use the inherited class' constructor
        super().__init__()
        self.device=torch.device(device)
        self.time_steps=time_steps
        self.logger=logger.get_logger()

        hidden = 256

        self.time_embed=TimeEmbedding(hidden)

        # now we are gonna employ a simple denoising network, which is mlp based.
        # and by using this structure we basically aim to predict the noise which is brilliant
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden, hidden), # hidden is our time embedding and by doing so 
            #we can look at the time and eeg signal data side by side.
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        ) # so the output here wilil represent the predicted noise present in the signal.

        self.betas = torch.linspace(1e-4, 0.02, time_steps).to(device)

        self.to(device)

        if weights_path:
            self._load_pretrained_weights(weights_path)
        
    def _load_pretrained_weights(self, path:str):
        try:
            self.logger.info("Loading pretrained D4PM weights from {path}...")
            checkpoint = torch.load(path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)

        except Exception as e:
            self.logger.error(f"Loading weights failed with error: {e}")
            


    # we wont be able to denoise x if we dont know t.
    # here we concatenate them because we are allowing the neural network
    #  to adjust the weights based on how noisy the image is.
    def forward(self, x:torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t) # here we run mlp and get the vector
        t_emb=t_emb.expand(x.shape[0], -1) # duplicates that single time vector 32 times.
        # soo we provide eeg signal in the batch their corresponding time vector.
        x_in=torch.cat([x, t_emb], dim=-1) # concatenates these 2 into.
        return self.net(x_in)
    
    @torch.no_grad()
    def denoise(self, x: torch.Tensor, return_cpu:bool=True) -> torch.Tensor:
        """
        Reverse diffusion process (inference) to obtain clean EEG again
        from Gaussian Noise.
        It iteratively removes noise from input tensor x across defined
        time steps. 
        
        :param self: instance of diffusion model class containing betas, noise schedule
        and neural network.
        :param x: starting tensor initialized with Gaussian Noise.
        shape is (batch_size, channels, signal_length)
        :type x: torch.Tensor
        :return: denoised EEG signal sequence or feature map.
        :rtype: Tensor
        """
        self.eval()
        x=x.to(self.device)

        x_t = x + torch.randn_like(x) * 0.1 # were addng a very weak noise here

        for step in reversed(range(self.time_steps)):
            t=torch.full((x_t.shape[0],), float(step), device=self.device, dtype=torch.float32)
            beta = self.betas[step]

            noise_pred = self.forward(x_t, t)
            noise = torch.randn_like(x_t, device=self.device) if step > 0 else 0

            x_t = x_t -beta * noise_pred + torch.sqrt(beta) * noise

        if return_cpu:
            return x_t.cpu()
        else:
            return x_t
        

        

