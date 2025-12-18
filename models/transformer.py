import torch
import torch.nn as nn
import math
from models.embeddings import EEGPatchEmbedding

class SinusoidalTimeEmbeddings(nn.Module):
    """
    sinusoidal embeddings give continuity, extrapolation, generalization on unseen steps,
    no learned parameters. standard in ddpm.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Convert a scalar time step t into a vector representation.

        1.split embedding space into sine+cosine
        2.use different frequencies
        3.encode time step as phase shifts

        basically this method is for the model to understand the time
    
        
        :param self: Description
        :param time_steps: noise level indicator
        :type time_steps: torch.Tensor (batch_size,)
        :return: Description
        :rtype: Tensor, (batch_size, embedding_dim)

        """

        device = time_steps.device

        half_dim = self.embedding_dim//2
        # half the dimensions are sin, half are cos 
        # if we have embedding_dim of 128, output is 64 sin + 64 cos = 128 dims
        # we do this bhcs sine alone is ambiguoug and cosine complements it
        # mentioned in transformer paper actually.
        # the reason for log10000 it defines freq.
        # early d,ms low freq latr dims high freq
        # we want the freqs logly not linly
        # we divide it by halfdim-1 bcs spreads freq evenly in log-space
        frequency_scale=math.log(10000)/(half_dim-1)

        frequencies = torch.exp(torch.arange(half_dim, device=device)*-frequency_scale)

        angular_terms = time_steps[:, None] * frequencies[None,:]

        time_embedding = torch.cat([angular_terms.sin(), angular_terms.cos()], dim=-1)

        return time_embedding
    
class EEGDiffusionTransformer(nn.Module):
    """
    Transformer backbone.
    For diffusion based EEG denoising.

    Converts raw EEG signals into patch embeddings
    Diffusion time step info is added
    Models temporal dependencies using transformer encoder
    predicts noise or clean signal for each EEG patch

    only the neural network architecure is handled.
    
    """
    def __init__(self, num_eeg_channels: int=19,
                    model_dim: int = 128,
                    num_attention_heads: int=8,
                    num_transformer_layers: int = 4,
                    patch_length: int = 25,
                    num_diffusion_steps: int = 1000
                    ):
        """
        Docstring for __init__
        
        :param self: Description
        :param num_eeg_channels: number of electrodes
        :type num_eeg_channels: int
        :param model_dim: transformer's internal feature size
        :type model_dim: int
        :param num_attention_heads: Description
        :type num_attention_heads: int
        :param num_transformer_layers: Description
        :type num_transformer_layers: int
        :param patch_length: Description
        :type patch_length: int
        :param num_diffusion_steps: Description
        :type num_diffusion_steps: int
        """
        
        super().__init__()
        self.model_dim = model_dim
        # we extract features here and tokenize this is necessary for to be fed into our transformer model we convert eeg signals into tokens
        self.eeg_patch_embedding = EEGPatchEmbedding(in_channels=num_eeg_channels, model_dim=model_dim, patch_size=patch_length, positional_encoding="learnable")

        # this is for understanding in which step are we of diffusion
        # and how much noise we have

        self.time_step_embedding_net = nn.Sequential(SinusoidalTimeEmbeddings(model_dim),nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.num_eeg_channels=num_eeg_channels
        self.patch_length=patch_length

        encoder_layer = nn.TransformerEncoderLayer(
        d_model=model_dim,
        nhead=num_attention_heads,
        dim_feedforward=model_dim * 4,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_transformer_layers
        )


        # maps transformer features back to eeg signal space
        self.output_projection = nn.Linear(model_dim, num_eeg_channels * patch_length)

    def forward(self, eeg_signal: torch.Tensor, diffusion_step: torch.Tensor) -> torch.Tensor:
        """
        Docstring for forward
        
        :param self: Description
        :param eeg_signal: Description
        :type eeg_signal: torch.Tensor
        :param diffusion_step: Description
        :type diffusion_step: torch.Tensor
        :return: Description
        :rtype: Tensor
        """
        # we tokenize it here. we have spatial info-channels and local temporal info(patch_length)
        # each eeg patch is a token now.
        patch_embeddings = self.eeg_patch_embedding(eeg_signal)
        # where we are in the diffusion process, how noisy is the signal, a global context not patch specific.
        time_step_embeddings = self.time_step_embedding_net(diffusion_step)

        # here we inject time step info into every patch token.
        time_step_conditioned_embeddings = (patch_embeddings + time_step_embeddings.unsqueeze(1))

        transformer_features = self.transformer_encoder(time_step_conditioned_embeddings)

        output = self.output_projection(transformer_features)
        output=output.view(output.shape[0], -1, self.num_eeg_channels, self.patch_length)
        output = output.permute(0, 2, 1, 3)
        output=output.reshape(output.shape[0], self.num_eeg_channels, -1)

        return output
        



