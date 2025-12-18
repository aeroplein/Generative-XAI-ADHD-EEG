import os
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.logger import LoggerHelper
from utils.dataset import EEGDataset

from models.engine import DiffusionEngine
from models.transformer import EEGDiffusionTransformer
from models.scheduler import DiffusionScheduler

def main():

    logger=LoggerHelper.get_logger()
    logger.info("Fine tuning for pediatric dataset")

    BATCH_SIZE=64
    LEARNING_RATE=5e-5
    EPOCHS=20
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")

    possible_paths=[
        os.path.join(BASE_DIR, "pediatric", "X_final.npy"),
        os.path.join(BASE_DIR, "X_final")
    ]


    pediatric_data_path=os.path.join(BASE_DIR, "pediatric", "X_final.npy")

    if not os.path.exists(pediatric_data_path):
        logger.error(f"Pediatric data cannot be found. The path: {pediatric_data_path}")
        return
    
    try:
        train_dataset=EEGDataset(pediatric_data_path)
        train_loader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        logger.info(f"Pediatric data is loaded. Total samples: {len(train_dataset)}")

    except Exception as e:
        logger.error(f"Dataset loading failed with error: {e}")
        return
    
    model=EEGDiffusionTransformer(
        num_eeg_channels=19,
        model_dim=128,
        num_attention_heads=8,
        num_transformer_layers=4,
        patch_length=16,
        num_diffusion_steps=1000
    ).to(DEVICE)

    pretrained_path=os.path.join(BASE_DIR, "checkpoints", "adult_model_epoch_15.pt")


    if os.path.exists(pretrained_path):
        logger.info(f"Pretrained adult model is loading:{pretrained_path}")
        try:
            state_dict=torch.load(pretrained_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            logger.info("Successfully loaded. Now the model knows the adult knowledge.") ##????????
        except Exception as e:
            logger.error(f"Model weight loading failed with error: {e}")
            return
    else:
        logger.warning(f"Pretrained model could not be found in {pretrained_path}")
        logger.warning("The training will start from scratch-no transfer learning.")

    scheduler=DiffusionScheduler(num_time_steps=1000, device=DEVICE)
    engine=DiffusionEngine(model, scheduler)
    optimizer=AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    logger.info("Fine tuning started...")
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
                logger.info(f"Epoch[{epoch+1}/{EPOCHS}] step {batch_index} loss: {loss.item():.4f}")

        avg_loss=total_loss/len(train_loader)
        logger.info(f"Epoch[{epoch+1}/{EPOCHS}] completed. Average loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)

        checkpoints_dir=os.path.join(BASE_DIR, "checkpoints_finetuned")
        os.makedirs(checkpoints_dir, exist_ok=True)

        if(epoch+1)%5==0:
            save_path=os.path.join(checkpoints_dir, f"child_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Checkpont saved: {save_path}")

    logger.info("Fine tuning training completed on child dataset.")
    np.save("train_loss_history.npy", np.array(loss_history)) # <--- BUNU EKLE
    logger.info("Loss history saved to train_loss_history.npy")

if __name__ == "__main__":
    main()







            

