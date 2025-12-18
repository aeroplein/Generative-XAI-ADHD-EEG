import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
import sys
import numpy as np

from utils.logger import LoggerHelper
from utils.dataset import EEGDataset

from models.engine import DiffusionEngine
from models.transformer import EEGDiffusionTransformer
from models.scheduler import DiffusionScheduler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def main():
    logger=LoggerHelper.get_logger()
    logger.info("Pretraining...")

    BATCH_SIZE=64
    LEARNING_RATE=1e-4
    EPOCHS=15
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {DEVICE}")
    
    x_path = os.path.join(BASE_DIR, "adult", "X_adult.npy")

    if not os.path.exists(x_path):
        logger.error(f"File not found: {x_path}")
        logger.error(f"Please first run \'adult_dataset_prep.ipynb\.")
        return
    
    try:
        train_dataset=EEGDataset(x_path)
        train_loader=DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    except Exception as e:
        logger.error(f"Dataset initialization failed with error: {e}")
        return
    
    model=EEGDiffusionTransformer(
        num_eeg_channels=19,
        model_dim=128,
        num_attention_heads=8,
        num_transformer_layers=4,
        patch_length=16,
        num_diffusion_steps=1000

    ).to(DEVICE)

    logger.info(f"Model parameter number: {sum(p.numel() for p in model.parameters()):,}")

    scheduler=DiffusionScheduler(num_time_steps=1000, device=DEVICE)
    engine=DiffusionEngine(model, scheduler)
    optimizer=AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    logger.info("Training is starting...")

    loss_history = []

    for epoch in range(EPOCHS):
        total_loss=0

        for batch_index, signals in enumerate(train_loader):
            signals=signals.to(DEVICE)
            optimizer.zero_grad()
            loss=engine.compute_loss(signals)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()

            if batch_index%50==0:
                logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Step {batch_index} Loss: {loss.item():.4f}")

    avg_loss=total_loss/len(train_loader)
    logger.info(f"Epoch {epoch+1} is done. Average loss: {avg_loss:.4f}")
    loss_history.append(avg_loss)

    os.makedirs("checkpoints", exist_ok=True)
    if (epoch+1)%5==0:
        torch.save(model.state_dict(), f"checkpoints/adult_model_epoch_{epoch+1}.pt")

        logger.info("Pretraining is completed.")

    np.save("train_loss_history.npy", np.array(loss_history)) 
    logger.info("Loss history saved to train_loss_history.npy")

if __name__ == "__main__":
    main()

        
