import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.dataset import EEGDataset
from models.transformer import EEGDiffusionTransformer
from models.scheduler import DiffusionScheduler
from utils.logger import LoggerHelper

def main():

    logger=LoggerHelper.get_logger()
    logger.info("Starting process.")

    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    possible_paths = [
        os.path.join(BASE_DIR, "pediatric", "X_final.npy"),
        os.path.join(BASE_DIR, "X_final.npy")
    ]

    data_path=None
    for path in possible_paths:
        if os.path.exists(path):
            data_path=path
            break
    
    if data_path is None:
        logger.error("Data not found.")
        return
    
    dataset=EEGDataset(data_path)
    indices=np.random.choice(len(dataset), 5, replace=False)

    model=EEGDiffusionTransformer(
        num_eeg_channels=19,
        model_dim=128,
        num_attention_heads=8,
        num_transformer_layers=4,
        patch_length=16,
        num_diffusion_steps=1000

    ).to(DEVICE)

    checkpoints_path=os.path.join(BASE_DIR, "checkpoints_finetuned", "child_model_epoch_20.pt")

    if not os.path.exists(checkpoints_path):
        logger(f"Model file not found: {checkpoints_path}")
        return
    
    model.load_state_dict(torch.load(checkpoints_path, map_location=DEVICE))
    model.eval()
    logger.info("Model and data is ready.")

    scheduler=DiffusionScheduler(num_time_steps=1000, device=DEVICE)

    results_dir=os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Visualization graphs:")

    with torch.no_grad():
        for i, index in enumerate(indices):
            original_signal=dataset[index].unsqueeze(0).to(DEVICE)
            t=torch.tensor([500], device=DEVICE).long()
            noisy_signal, _ = scheduler.add_noise(original_signal, t)
            predicted_noise=model(noisy_signal, t)
            reconstructed_signal=noisy_signal-predicted_noise

            channel_index=0
            original=original_signal[0, channel_index, :].cpu().numpy()
            noisy=noisy_signal[0, channel_index, :].cpu().numpy()
            reconstructed=reconstructed_signal[0, channel_index,:].cpu().numpy()

            plt.figure(figsize=(15,5))

            plt.subplot(1,3,1)
            plt.plot(original, color='hotpink')
            plt.title(f"Original signal (child {index})")

            plt.subplot(1,3,2)
            plt.plot(noisy, color='green', alpha=0.7)
            plt.title("Noisy (t=500)")

            plt.subplot(1,3,3)
            plt.plot(reconstructed, color='blue')
            plt.title("Reconstructed")

            save_file=os.path.join(results_dir, f"result_sample_{i}.png")
            plt.savefig(save_file)
            plt.close()
            logger.info(f"Graph saved: {save_file}")

    logger.info("Test completed. check results folder.")

if __name__ == "__main__":
    main()


