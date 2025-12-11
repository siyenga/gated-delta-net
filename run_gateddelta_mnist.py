# run_gateddelta_mnist.py
import sys
import os
import torch
import torch.nn as nn
import numpy as np

# --- adjust path if needed ---
# This file assumes GatedDeltaNet.py sits in the same directory.
from importlib import util
#spec = util.spec_from_file_location("user_model", os.path.join(os.path.dirname(__file__), "GatedDeltaNet.py"))
spec = util.spec_from_file_location("user_model",  "GatedDeltaNet.py")
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)
GatedDeltaNet = getattr(mod, "GatedDeltaNet")

print("Imported GatedDeltaNet:", GatedDeltaNet)

# Attempt to load MNIST using tensorflow (no torchvision dependency)
use_mnist = True
try:
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("MNIST loaded, test set shape:", x_test.shape)
except Exception as e:
    print("Could not download/load MNIST via tensorflow. Falling back to random data. Error:", e)
    use_mnist = False

# Convert images into patches: 7x7 non-overlapping => 4x4 grid => 16 tokens each dim=49
def images_to_patches_numpy(imgs_np):
    # imgs_np: (B, 28, 28) or (B,1,28,28)
    if imgs_np.ndim == 4:
        imgs_np = imgs_np[:,0,:,:]
    B = imgs_np.shape[0]
    patches = []
    for i in range(0,28,7):
        for j in range(0,28,7):
            p = imgs_np[:, i:i+7, j:j+7]  # (B,7,7)
            p = p.reshape(B, -1)          # (B,49)
            patches.append(p)
    tokens = np.stack(patches, axis=1)  # (B,16,49)
    return tokens

# Prepare a small batch
B = 8
if use_mnist:
    imgs = x_test[:B].astype(np.float32) / 255.0  # (B,28,28)
    labels = y_test[:B].astype(np.int64)
else:
    imgs = np.random.rand(B,28,28).astype(np.float32)
    labels = np.random.randint(0,10,size=(B,)).astype(np.int64)

tokens_np = images_to_patches_numpy(imgs)  # (B,16,49)
tokens = torch.from_numpy(tokens_np)       # float32
labels = torch.from_numpy(labels)

print("tokens_np.shape:", tokens_np.shape)

# Model hyperparams to match patch dims
d_in = 49     # token feature dim
d_out = 64    # internal output dim (must match model expectations if any)
num_heads = 8
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Instantiate user's model
try:
    model = GatedDeltaNet(d_in=d_in, d_out=d_out, dropout=dropout, num_heads=num_heads)
except TypeError as e:
    print("Constructor signature mismatch:", e)
    print("Try inspecting GatedDeltaNet.__init__ to pass the right args.")
    raise

model = model.to(device)
model.eval()

# Small classifier head: pool token outputs -> class logits
classifier = nn.Linear(d_out, 10).to(device)

tokens = tokens.to(device)
labels = labels.to(device)

with torch.no_grad():
    out = model(tokens)    # your model may return tensor or (tensor, extras)
    # Accept either (B, T, D) or (B, D) outputs
    if isinstance(out, (tuple, list)):
        seq = out[0]
    else:
        seq = out

    print("Model forward returned tensor with shape:", getattr(seq, "shape", None))
    if seq.dim() == 3:
        # sequence case: (B, T, D)
        pooled = seq.mean(dim=1)   # (B, D)
    elif seq.dim() == 2:
        pooled = seq
    else:
        raise RuntimeError("Unexpected output dims from model: got shape " + str(seq.shape))

    logits = classifier(pooled)
    preds = logits.argmax(dim=1)

    print("Preds:", preds.cpu().numpy())
    print("Labels:", labels.cpu().numpy())

print("Done. If this fails, paste the error and I'll help debug.")
