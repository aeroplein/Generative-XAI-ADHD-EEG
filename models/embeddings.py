import torch
import torch.nn as nn
from utils.logger import LoggerHelper

class EEGPatchEmbedding(nn.Module):
    """
    Docstring for EEGPatchEmbedding
    """
    
    def __init__(self, in_channels: int, model_dim, patch_size: int, positional_encoding: str = "learnable", signal_length: int = 256):
        """
        Docstring for __init__
        
        :param self: Description
        :param in_channels: Number of electrodes in EEG device
        :type in_channels: int
        :param model_dim: vector length
        The voltage data is taken and converted into a vector with length of 128.
        These are not voltage, but they are the extracted features.
        The bigger the values, the better the model learns complex relationships.
        :param patch_size: How many pieces the long signal will be divided into
        :type patch_size: int
        :param positional_encoding: How is the position information will be held.
        Transformer models do not know whether a piece is the start or end of the signal.
        It learns itself.
        :type positional_encoding: str
        :param signal_length: the length of the window that goes through the model.
        :type signal_length: int
        """
        super().__init__()
        
        self.logger = LoggerHelper.get_logger()
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.positional_encoding = positional_encoding

        
        if signal_length % patch_size != 0:
            error_msg = f"Window size ({signal_length} cannot be divided into ({patch_size}))"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.num_patches = signal_length//patch_size

        patch_vector_dim = in_channels * patch_size

        self.patch_projection = nn.Linear(patch_vector_dim, model_dim)

        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, model_dim)*0.02)

        self.logger.info(f"{__name__} initialized: {in_channels} channels, {self.num_patches} patches.")

    def forward(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """
        Docstring for forward
        
        :param self: Description
        :param eeg_signal: Description
        :type eeg_signal: torch.Tensor
        :return: Description
        :rtype: Tensor
        """

        # batch_size: how many EEG samples
        # num_patches: how many pieces into the signal is divided
        # num_channels: EEG channel number like 19
        # patch_size: step number per patch
        # every patch looks like this:
        # [num_channels × patch_size]
        # but it cannot 


        batch_size, in_channels, signal_length = eeg_signal.shape

        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {in_channels}.")
        

        num_patches = signal_length // self.patch_size


        # we divide signal_length into small time blocks.
        patches = eeg_signal.view(batch_size, self.in_channels,num_patches, self.patch_size)

        # we changed their orders
        patches = patches.permute(0, 2, 1, 3).contiguous()
        
        # in each patch, we change it into one vector including the channel x tşme info
        # like this:
        # (channels, patch_size)
        # later: (channels * patch_size)
        # the latest version then becomes like this:
        # (batch_size, num_patches, channels*patch_size)
        patches_flattened = patches.view(batch_size, num_patches, -1)

        # it is the latent embedding that the transformer can understand
        # and the final shape becomes 
        # (batch_szie, num_patches, model_dim)
        patch_embeddings = self.patch_projection(patches_flattened)

        

        patch_embeddings = patch_embeddings + self.positional_embedding

        return patch_embeddings
        
        


