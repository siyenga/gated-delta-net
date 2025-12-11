"""
Run inference on MNIST using a trained GatedDeltaNet model.
"""

import torch
from torchvision import datasets, transforms
from gateddelta_mnist_full  import GatedDeltaNetClassifier, images_to_patches_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------
# 1. Load MNIST test set
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_test = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# pick any MNIST index to test
index = 6
img, label = mnist_test[index]
print("Actual Label:", label)

# reshape to batch form: (1,1,28,28)
img = img.unsqueeze(0).to(device)

# ---------------------------------------------------------
# 2. Convert image → patches (required by GatedDeltaNet)
# ---------------------------------------------------------
tokens = images_to_patches_tensor(img).to(device)
print("Tokens shape:", tokens.shape)

# ---------------------------------------------------------
# 3. Load the trained GatedDeltaNet model
# ---------------------------------------------------------
model = GatedDeltaNetClassifier(
    patch_size=7,
    in_channels=1,
    d_model=64,
    num_layers=6,
    hidden_dim=256,
    num_classes=10,
    dropout=0.1
).to(device)

# load weights
checkpoint_path = "checkpoints/gateddelta_best.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ---------------------------------------------------------
# 4. Run inference
# ---------------------------------------------------------
with torch.no_grad():
    logits = model(tokens)
    pred = logits.argmax(dim=1).item()

print("Predicted Label:", pred)
