import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 

from DatasetBuilder import GeoCoordDataset
from model import SimpleCNN

if __name__ == "__main__":
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    MODEL_SAVE_PATH = "saved_models_pytorch"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    geo_dataset = GeoCoordDataset()

    num_dataloader_workers = os.cpu_count() // 2 if os.cpu_count() > 1 else 0
    print(f"DataLoader: Using {num_dataloader_workers} workers.")

    train_dataloader = geo_dataset.get_dataloader(
        shuffle=True,
        num_workers=num_dataloader_workers
    )

    model = SimpleCNN(geo_dataset.IMG_SIZE).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    training_losses = []
    print("Starting training")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}",
                    total=len(train_dataloader), leave=True) 

        for batch_idx, (images, coords) in pbar: 
            images, coords = images.to(device), coords.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"}) 

            if (batch_idx + 1) % 1000 == 0:
                total_batches_processed = epoch * len(train_dataloader) + (batch_idx + 1)
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"model_batch_{total_batches_processed:05d}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"\n--- Checkpoint saved at Batch {total_batches_processed}: {checkpoint_path}") 

        epoch_loss = running_loss / len(train_dataloader.dataset)
        training_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Total Loss: {epoch_loss:.4f}")

    print("\nTraining complete!")

    FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "final_trained_model.pth")
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model weights saved to: {FINAL_MODEL_PATH}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), training_losses, label="Training Loss", color="blue")
    plt.title("Training Loss Over Epochs (PyTorch)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Mean Squared Error)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()