import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow
import time

from model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/processed"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001


def main():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=f"{DATA_DIR}/train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root=f"{DATA_DIR}/val",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = get_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    mlflow.set_experiment("Cats_vs_Dogs")

    with mlflow.start_run():

        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LR)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), MODEL_DIR / "model.pt")
        mlflow.log_artifact(str(MODEL_DIR / "model.pt"))

        print("Training complete.")


if __name__ == "__main__":
    main()
