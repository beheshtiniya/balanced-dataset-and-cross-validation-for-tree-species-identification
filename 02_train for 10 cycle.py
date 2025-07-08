# --- کد کامل با اصلاح early stopping فقط بر اساس عدم بهبود در val_loss ---

import os
import pandas as pd
import torch
import torchvision
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from tqdm import tqdm
import subprocess

CUDA_LAUNCH_BLOCKING = 1.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

class trDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        self.root = root
        self.phase = phase
        self.targets = pd.read_csv(os.path.join(root, '{}_labels.csv'.format(phase)))
        self.imgs = self.targets['filename']

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images_rename', self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img)
        filename = self.imgs[idx]

        box_list = self.targets[self.targets['filename'] == self.imgs[idx]][['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.tensor(box_list, dtype=torch.float32)

        label_list = self.targets[self.targets['filename'] == self.imgs[idx]][['class']].values.squeeze(1)
        labels = torch.tensor(label_list, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return img, target, filename

    def __len__(self):
        return len(self.imgs)

def new_concat(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, dataloader):
    model.train()
    total_loss = 0
    for images, targets, _ in tqdm(dataloader, desc="Training", leave=True):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    return total_loss / len(dataloader)

def evaluate_loss(model, dataloader):
    model.train()
    total_loss = 0
    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Validation", leave=True):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    return total_loss / len(dataloader)

def run_test_evaluation(root_dir, checkpoints_dir):
    test_eval_script = os.path.join(os.getcwd(), "test_evaluate.py")
    subprocess.run([
        "python", test_eval_script,
        "--root_dir", root_dir,
        "--checkpoints_dir", checkpoints_dir
    ])

# مسیر داده‌ها
root_dir = r'E:\FASTRCNN\FASTRCNN\dataset\00_source_code_data_ADDING BEST 0.95 TO TRAIN AND VAL\step1_tts_2'

# بارگذاری داده‌ها
train_dataset = trDataset(root_dir, 'train')
val_dataset = trDataset(root_dir, 'val')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=new_concat)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=new_concat)

# اطمینان از وجود پوشه‌ی checkpoints
os.makedirs(os.path.join(root_dir, "checkpoints"), exist_ok=True)

for run_id in range(1, 11):
    print(f"Starting Training Run {run_id}")
    safe_run_id = f"run_{run_id}"
    checkpoints_dir = os.path.join(root_dir, "checkpoints", safe_run_id)
    os.makedirs(checkpoints_dir, exist_ok=True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 5)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(10):
        print("Start training...")
        train_loss = train_one_epoch(model, optimizer, train_loader)
        val_loss = evaluate_loss(model, val_loader)

        print(f"Epoch [{epoch}] Run {run_id}: \t Train Loss: {train_loss:.4f} \t Val Loss: {val_loss:.4f}")
        lr_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"best_model_{epoch}.pth"))
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best_model.pth"))

        # ✅ فقط شرط دوم: عدم بهبود در val loss در ۳ ایپاک اخیر
        no_improve = False
        if len(val_losses) >= 4:
            no_improve = all(val_losses[-i] >= val_losses[-i-1] for i in range(1, 4))

        if no_improve:
            print(f"🛑 Early stopping due to no improvement at epoch {epoch} in run {run_id}")
            run_test_evaluation(root_dir, checkpoints_dir)
            break

    else:
        run_test_evaluation(root_dir, checkpoints_dir)
