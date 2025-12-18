import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import welch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models.transformer import EEGDiffusionTransformer
from utils.dataset import EEGDataset

def ddpm_sampling_loop(model, shape, device, logger, save_interval=100):
    img = torch.randn(shape, device=device)
    num_times_steps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    
    betas = torch.linspace(beta_start, beta_end, num_times_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    snapshot = {}

    model.eval()
    with torch.no_grad():
        for t in reversed(range(num_times_steps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            predicted_noise = model(img, t_tensor)
            
            alpha = alphas[t]
            alpha_hat = alphas_cumprod[t]
            beta = betas[t]

            noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)

            coefficient = beta / torch.sqrt(1 - alpha_hat)
            mean = (1 / torch.sqrt(alpha)) * (img - (coefficient * predicted_noise))

            img = mean + (torch.sqrt(beta) * noise)

            if t % save_interval == 0 or t == 0:
                snapshot[t] = img.cpu().numpy()
                logger.info(f"Denoising step {t} completed.")
                
    return img, snapshot

def plot_psd_comparison(real_signal, synthetic_signal, save_dir):
    fs = 128
    
    def normalize_signal(sig):
        return (sig - np.mean(sig)) / np.std(sig)

    real_signal_norm = normalize_signal(real_signal.flatten())
    synthetic_signal_norm = normalize_signal(synthetic_signal.flatten())

    f_real, Pxx_real = welch(real_signal_norm, fs=fs, nperseg=128)
    f_syn, Pxx_syn = welch(synthetic_signal_norm, fs=fs, nperseg=128)

    plt.figure(figsize=(10, 6))
    plt.semilogy(f_real, Pxx_real, label="Real ADHD Data (Normalized)", color="hotpink", alpha=0.8, linewidth=2)
    plt.semilogy(f_syn, Pxx_syn, label='Synthetic Data (Normalized)', color='blue', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.title("Generative XAI: Biological Validity Proof (Normalized PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Density (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "xai_frequency_proof_fixed.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_diffusion_evolution(snapshots, save_dir):
    steps = sorted(snapshots.keys(), reverse=True)

    selected_steps = [steps[0], steps[len(steps)//2], steps[-1]] 

    plt.figure(figsize=(15, 5))

    for i, step in enumerate(selected_steps):
        signal = snapshots[step] 

        if signal.ndim == 3:
            signal = signal[0]

        channel_data = signal[0, :] 

        plt.subplot(1, 3, i+1)
        plt.plot(channel_data, color='purple')
        plt.title(f"Time Step: {step}")
        plt.ylim(-3, 3) 
        plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, "diffusion_evolution.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def main():
    try:
        from utils.logger import LoggerHelper
        logger = LoggerHelper.get_logger()
    except ImportError:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("XAI")

    logger.info("=== STARTING GENERATIVE XAI ===")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")

    possible_paths = [
        os.path.join(BASE_DIR, "pediatric", "X_final.npy"),
        os.path.join(BASE_DIR, "X_final.npy")
    ]
    data_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if data_path is None:
        logger.error("Data file not found!")
        return
    
    real_data = np.load(data_path)
    real_sample = real_data[np.random.randint(len(real_data))]
    logger.info("Real data loaded.")

    model = EEGDiffusionTransformer(
        num_eeg_channels=19, model_dim=128, num_attention_heads=8, 
        num_transformer_layers=4, patch_length=16, num_diffusion_steps=1000
    ).to(DEVICE)
    
    ckpt_path = os.path.join(BASE_DIR, "checkpoints_finetuned", "child_model_epoch_20.pt")
    if not os.path.exists(ckpt_path):
        logger.error("Model file not found!")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    logger.info("Model weights loaded.")

    logger.info("Generating synthetic signal...")
    sample_shape = (1, 19, 256)
    synthetic_tensor, snapshot = ddpm_sampling_loop(model, sample_shape, DEVICE, logger)
    synthetic_signal = synthetic_tensor.cpu().numpy()[0]

    save_dir = os.path.join(BASE_DIR, "results", "xai")
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Plotting diffusion evolution...")
    plot_diffusion_evolution(snapshot, save_dir)

    logger.info("Performing frequency analysis...")
    psd_path = plot_psd_comparison(real_sample[0], synthetic_signal[0], save_dir)
    
    logger.info(f"Proof graph saved: {psd_path}")
    logger.info("Process completed!")

if __name__ == "__main__":
    main()