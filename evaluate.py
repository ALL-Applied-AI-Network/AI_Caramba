import torch
import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    "dataset/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8
)

num_classes = len(test_dataset.classes)

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=num_classes
)

model.load_state_dict(
    torch.load("recipe_model.pth")
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
