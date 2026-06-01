import os
import io
import time
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

ZIP_PATH   = Path(__file__).parent.parent / "captcha-dataset.zip"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_H      = 27
IMG_W      = 73
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3
N_DIGITS   = 4
N_CLASSES  = 10

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


class CaptchaDataset(Dataset):
    def __init__(self, zip_path, split, transform=None):
        self.transform = transform
        self.samples = []
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist()
                     if n.startswith(split + "/") and n.endswith(".png")]
            for name in names:
                label = [int(c) for c in Path(name).stem]
                self.samples.append((zf.read(name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        img = Image.open(io.BytesIO(data)).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


transform_train = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# CNN con backbone compartido y 4 cabezas (una por digito)
class CaptchaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.heads = nn.ModuleList([nn.Linear(512, N_CLASSES) for _ in range(N_DIGITS)])

    def forward(self, x):
        x = self.fc(self.backbone(x))
        return [h(x) for h in self.heads]


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum = 0.0
    correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = sum(criterion(out[i], labels[:, i]) for i in range(N_DIGITS))
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = torch.stack([o.argmax(1) for o in out], dim=1)
        correct += (preds == labels).all(dim=1).sum().item()
        total += labels.size(0)

    return loss_sum / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = total = 0
    all_labels, all_probs = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        loss = sum(criterion(out[i], labels[:, i]) for i in range(N_DIGITS))
        loss_sum += loss.item()

        probs = torch.stack([o.softmax(1) for o in out], dim=1)
        preds = probs.argmax(-1)
        correct += (preds == labels).all(dim=1).sum().item()
        total += labels.size(0)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    classes = list(range(N_CLASSES))
    aucs = []
    for pos in range(N_DIGITS):
        y_true  = all_labels[:, pos]
        y_score = all_probs[:, pos, :]
        if len(np.unique(y_true)) < 2:
            continue
        auc = roc_auc_score(
            label_binarize(y_true, classes=classes),
            y_score,
            multi_class="ovr",
            average="macro",
        )
        aucs.append(auc)

    return loss_sum / len(loader), correct / total, float(np.mean(aucs)), all_labels, all_probs


def main():
    train_ds = CaptchaDataset(ZIP_PATH, "train", transform_train)
    val_ds   = CaptchaDataset(ZIP_PATH, "val",   transform_val)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model     = CaptchaCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {n_params:,}")

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_auc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{EPOCHS} | loss tr={tr_loss:.4f} va={va_loss:.4f} | "
              f"acc tr={tr_acc:.4f} va={va_acc:.4f} | auc={va_auc:.4f} | {time.time()-t0:.1f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), Path(__file__).parent / "best_model.pth")

    model.load_state_dict(torch.load(Path(__file__).parent / "best_model.pth", weights_only=True))
    print(f"\nMejor val accuracy: {best_acc:.4f}")

    # evaluacion final
    _, acc, auc_global, all_labels, all_probs = evaluate(model, val_loader, criterion)
    classes = list(range(N_CLASSES))
    print(f"Accuracy (4 digitos): {acc:.4f}")
    for pos in range(N_DIGITS):
        y_true  = all_labels[:, pos]
        y_score = all_probs[:, pos, :]
        auc = roc_auc_score(label_binarize(y_true, classes=classes), y_score,
                            multi_class="ovr", average="macro")
        digit_acc = (y_score.argmax(1) == y_true).mean()
        print(f"  Digito {pos+1}: acc={digit_acc:.4f}  auc={auc:.4f}")
    print(f"AUC global: {auc_global:.4f}")

    pth_path = Path(__file__).parent / "best_model.pth"
    print(f"Tamaño modelo: {pth_path.stat().st_size / 1024**2:.2f} MB")

    model.eval()
    dummy = torch.randn(1, 1, IMG_H, IMG_W).to(DEVICE)
    with torch.no_grad():
        for _ in range(50):
            model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(200):
            t0 = time.perf_counter()
            model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    print(f"Inferencia: {np.mean(times):.3f} ms +/- {np.std(times):.3f} ms")

    class OnnxWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return torch.stack(self.m(x), dim=1)

    wrapper = OnnxWrapper(model).to(DEVICE)
    wrapper.eval()
    onnx_path = Path(__file__).parent / "captcha_cnn.onnx"
    torch.onnx.export(
        wrapper,
        torch.randn(1, 1, IMG_H, IMG_W).to(DEVICE),
        str(onnx_path),
        input_names=["image"],
        output_names=["digits"],
        dynamic_axes={"image": {0: "batch"}, "digits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX guardado: {onnx_path}  ({onnx_path.stat().st_size / 1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
