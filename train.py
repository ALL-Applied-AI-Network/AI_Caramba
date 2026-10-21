import torch
import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Resize images to the size EfficientNet expects
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load training and validation datasets
train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8
)

# Count recipe classes automatically
num_classes = len(train_dataset.classes)

print("Classes:", train_dataset.classes)
print("Number of classes:", num_classes)

# Load EfficientNet-B0
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=num_classes
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# Training loop
for epoch in range(10):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/10 "
        f"Loss = {running_loss:.4f}"
    )

# Save trained model
torch.save(
    model.state_dict(),
    "recipe_model.pth"
)

print("Training complete!")
