import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

INTERNAL_DATA = "dataset/train"
MODEL_SAVE_PATH = "saved_models/"
IMG_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', DEVICE)


class Network(nn.Module):
    def __init__(self, num_classes=94):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (128, 256, 256)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (256, 128, 128)

            # Global average pool to 1×1
            nn.AdaptiveAvgPool2d((1, 1))  # -> (256, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # -> (256,)
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


transform = transforms.Compose([
    transforms.RandomApply(
        [transforms.RandomResizedCrop(
            256,  # final size
            scale=(0.8, 1.0),  # keep 80–100% of area
            ratio=(0.95, 1.05))  # near‑square, keeps distortion low
        ],
        p=0.2
    ),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])
print('loading data')
train_ds = datasets.ImageFolder(root=INTERNAL_DATA, transform=transform)
print(train_ds.classes)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
model = Network().to(DEVICE)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

torch.backends.cudnn.benchmark = True
model = torch.compile(model, mode="max-autotune")
scaler = torch.amp.GradScaler('cuda')

print('training')
loss_history = []
i = 1
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0
    model.train()

    for images, labels in tqdm(train_dl,
                               desc=f"Epoch {epoch}/{EPOCHS}",
                               leave=False,
                               ncols=80):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        epoch_loss += loss.item() * batch_size

        if i % 32278 == 0:
            ckpt = f"{MODEL_SAVE_PATH}/model_batch_{i:05d}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"saved checkpoint {ckpt}")
        i += 1

    epoch_loss /= len(train_ds)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch:02d}/{EPOCHS} — loss: {epoch_loss:.4f}")

final_dir = os.path.join(MODEL_SAVE_PATH, "final_trained_model")
os.makedirs(final_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
print(f"saved final model to {final_dir}")

epochsRange = range(1, EPOCHS + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochsRange, loss_history, label="Training loss", color="blue")
plt.title("Training Over Loss Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss (Mean Squared Error)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
