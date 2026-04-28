import subprocess
import sys
def install_packages():
    packages = [
        "albumentations==1.3.1",
        "lpips",
        "torch-fidelity",
        "torchinfo",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "pandas",
        "seaborn",
        "opencv-python-headless",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
install_packages()

import os
import gc
import cv2
import time
import json
import math
import random
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, ImageDraw, ImageFilter
from tqdm.auto import tqdm
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import torchvision.utils as vutils
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import lpips
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
warnings.filterwarnings("ignore")

class Config:
    COLOR_MODE = "GRAY"
    FORCE_RETRAIN = True
    TRAIN_IMG_DIR = "./data/train"
    BASE_OUTPUT = "./output"
    CHECKPOINT_DIR = os.path.join(BASE_OUTPUT, "checkpoints")
    RESULTS_DIR = os.path.join(BASE_OUTPUT, "results")
    FIGURES_DIR = os.path.join(BASE_OUTPUT, "figures")
    IMG_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 200
    LR = 2e-4
    SEED = 42
    MASK_CACHE_SIZE = 1024
    MIN_MASK_RATIO = 0.15
    MAX_MASK_RATIO = 0.50
    VAL_SPLIT = 0.15
    NUM_VIS_SAMPLES = 8
    @property
    def CHANNELS(self):
        return 3 if self.COLOR_MODE == "RGB" else 1

cfg = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(Config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class PolynomialFractureMaskGenerator:
    def __init__(self, img_size=256):
        self.img_size = img_size

    def _polynomial_curve(self, x_points, degree=3):
        if len(x_points) < degree + 1:
            degree = len(x_points) - 1
        t = np.linspace(0, 1, len(x_points))
        t_fine = np.linspace(0, 1, 200)
        x_coords = np.array([p[0] for p in x_points])
        y_coords = np.array([p[1] for p in x_points])
        coeffs_x = np.polyfit(t, x_coords, degree)
        coeffs_y = np.polyfit(t, y_coords, degree)
        x_smooth = np.polyval(coeffs_x, t_fine)
        y_smooth = np.polyval(coeffs_y, t_fine)
        return list(zip(x_smooth.astype(int), y_smooth.astype(int)))

    def _add_jagged_noise(self, points, amplitude=8):
        noisy = []
        for x, y in points:
            nx = x + random.randint(-amplitude, amplitude)
            ny = y + random.randint(-amplitude, amplitude)
            nx = np.clip(nx, 0, self.img_size - 1)
            ny = np.clip(ny, 0, self.img_size - 1)
            noisy.append((int(nx), int(ny)))
        return noisy

    def generate_corner_fracture(self, corner="top_left"):
        h = w = self.img_size
        mask = np.zeros((h, w), dtype=np.uint8)
        depth_x = random.randint(int(w * 0.28), int(w * 0.58))
        depth_y = random.randint(int(h * 0.28), int(h * 0.58))
        degree = random.choice([3, 4])

        if corner == "top_left":
            n_ctrl = random.randint(4, 6)
            ctrl_points = [(0, 0)]
            ctrl_points.append((depth_x + random.randint(-20, 20), random.randint(5, 30)))
            for _ in range(n_ctrl - 4):
                ctrl_points.append((
                    random.randint(int(depth_x * 0.3), int(depth_x * 0.8)),
                    random.randint(int(depth_y * 0.3), int(depth_y * 0.8))
                ))
            ctrl_points.append((random.randint(5, 30), depth_y + random.randint(-20, 20)))
            ctrl_points.append((0, 0))
            ctrl_points_inner = ctrl_points[1:-1]
            ctrl_points_inner.sort(key=lambda p: math.atan2(p[1], p[0]))
            boundary = [(depth_x, 0)]
            boundary.extend(ctrl_points_inner)
            boundary.append((0, depth_y))
            curve = self._polynomial_curve(boundary, degree=min(degree, len(boundary)-1))
            curve = self._add_jagged_noise(curve, amplitude=5)
            polygon = [(0, 0), (depth_x, 0)] + curve + [(0, depth_y), (0, 0)]
        elif corner == "top_right":
            depth_x_start = w - depth_x
            ctrl_points_inner = []
            for _ in range(random.randint(2, 4)):
                ctrl_points_inner.append((
                    random.randint(int(depth_x_start + depth_x * 0.2), w - 5),
                    random.randint(int(depth_y * 0.2), int(depth_y * 0.8))
                ))
            ctrl_points_inner.sort(key=lambda p: math.atan2(p[1], w - p[0]))
            boundary = [(depth_x_start, 0)] + ctrl_points_inner + [(w - 1, depth_y)]
            curve = self._polynomial_curve(boundary, degree=min(degree, len(boundary)-1))
            curve = self._add_jagged_noise(curve)
            polygon = [(w-1, 0), (depth_x_start, 0)] + curve + [(w-1, depth_y), (w-1, 0)]
        elif corner == "bottom_left":
            depth_y_start = h - depth_y
            ctrl_points_inner = []
            for _ in range(random.randint(2, 4)):
                ctrl_points_inner.append((
                    random.randint(int(depth_x * 0.2), int(depth_x * 0.8)),
                    random.randint(int(depth_y_start + depth_y * 0.2), h - 5)
                ))
            ctrl_points_inner.sort(key=lambda p: math.atan2(h - p[1], p[0]))
            boundary = [(0, depth_y_start)] + ctrl_points_inner + [(depth_x, h - 1)]
            curve = self._polynomial_curve(boundary, degree=min(degree, len(boundary)-1))
            curve = self._add_jagged_noise(curve)
            polygon = [(0, h-1), (0, depth_y_start)] + curve + [(depth_x, h-1), (0, h-1)]
        elif corner == "bottom_right":
            depth_x_start = w - depth_x
            depth_y_start = h - depth_y
            ctrl_points_inner = []
            for _ in range(random.randint(2, 4)):
                ctrl_points_inner.append((
                    random.randint(int(depth_x_start + depth_x * 0.2), w - 5),
                    random.randint(int(depth_y_start + depth_y * 0.2), h - 5)
                ))
            ctrl_points_inner.sort(key=lambda p: math.atan2(h - p[1], w - p[0]))
            boundary = [(w-1, depth_y_start)] + ctrl_points_inner + [(depth_x_start, h-1)]
            curve = self._polynomial_curve(boundary, degree=min(degree, len(boundary)-1))
            curve = self._add_jagged_noise(curve)
            polygon = [(w-1, h-1), (w-1, depth_y_start)] + curve + [(depth_x_start, h-1), (w-1, h-1)]
        else:
            return self.generate_edge_fracture()

        pts = np.array(polygon, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    def generate_edge_fracture(self):
        h = w = self.img_size
        mask = np.zeros((h, w), dtype=np.uint8)
        edge = random.choice(["top", "bottom", "left", "right"])
        depth = random.randint(int(h * 0.22), int(h * 0.42))
        n_points = random.randint(5, 10)

        if edge == "top":
            x_pts = sorted([random.randint(0, w-1) for _ in range(n_points)])
            y_pts = [random.randint(5, depth) for _ in range(n_points)]
            boundary = list(zip(x_pts, y_pts))
            curve = self._polynomial_curve(boundary, degree=min(3, len(boundary)-1))
            curve = self._add_jagged_noise(curve, amplitude=4)
            polygon = [(0, 0)] + curve + [(w-1, 0)]
        elif edge == "bottom":
            x_pts = sorted([random.randint(0, w-1) for _ in range(n_points)])
            y_pts = [h - 1 - random.randint(5, depth) for _ in range(n_points)]
            boundary = list(zip(x_pts, y_pts))
            curve = self._polynomial_curve(boundary, degree=min(3, len(boundary)-1))
            curve = self._add_jagged_noise(curve, amplitude=4)
            polygon = [(0, h-1)] + curve + [(w-1, h-1)]
        elif edge == "left":
            y_pts = sorted([random.randint(0, h-1) for _ in range(n_points)])
            x_pts = [random.randint(5, depth) for _ in range(n_points)]
            boundary = list(zip(x_pts, y_pts))
            curve = self._polynomial_curve(boundary, degree=min(3, len(boundary)-1))
            curve = self._add_jagged_noise(curve, amplitude=4)
            polygon = [(0, 0)] + curve + [(0, h-1)]
        elif edge == "right":
            y_pts = sorted([random.randint(0, h-1) for _ in range(n_points)])
            x_pts = [w - 1 - random.randint(5, depth) for _ in range(n_points)]
            boundary = list(zip(x_pts, y_pts))
            curve = self._polynomial_curve(boundary, degree=min(3, len(boundary)-1))
            curve = self._add_jagged_noise(curve, amplitude=4)
            polygon = [(w-1, 0)] + curve + [(w-1, h-1)]

        pts = np.array(polygon, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    def generate_multi_damage(self):
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        n_damages = random.randint(1, 3)
        corners = random.sample(["top_left", "top_right", "bottom_left", "bottom_right"], min(n_damages, 4))
        for corner in corners:
            single_mask = self.generate_corner_fracture(corner)
            mask = np.maximum(mask, single_mask)
        if random.random() > 0.5:
            edge_mask = self.generate_edge_fracture()
            mask = np.maximum(mask, edge_mask)
        return mask

    def generate(self):
        choice = random.random()
        if choice < 0.4:
            corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
            mask = self.generate_corner_fracture(corner)
        elif choice < 0.7:
            mask = self.generate_edge_fracture()
        else:
            mask = self.generate_multi_damage()

        for _ in range(random.randint(1, 3)):
            ksize = random.choice([3, 5])
            kernel = cv2.getStructuringElement(
                random.choice([cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]),
                (ksize, ksize)
            )
            if random.random() > 0.5:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)

        mask_float = mask.astype(np.float32)
        mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0)
        mask = (mask_float > 0.5).astype(np.uint8)

        ratio = mask.sum() / (self.img_size * self.img_size)
        if ratio < cfg.MIN_MASK_RATIO or ratio > cfg.MAX_MASK_RATIO:
            return self.generate()
        return mask

    def generate_tensor(self):
        mask = self.generate()
        return torch.from_numpy(mask).float().unsqueeze(0)

class HarappanSealDataset(Dataset):
    def __init__(self, img_dir, file_list=None, color_mode="GRAY", img_size=256,
                 mask_cache_size=1024, augment=False):
        self.img_dir = img_dir
        self.color_mode = color_mode
        self.img_size = img_size
        self.augment = augment
        self.channels = 3 if color_mode == "RGB" else 1

        if file_list is not None:
            self.image_files = file_list
        else:
            all_files = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.image_files = [f for f in all_files if '_aug' not in f.lower()]

        print(f"  Dataset: {len(self.image_files)} images (excluded _aug)")

        if self.channels == 3:
            self.norm_mean = [0.5, 0.5, 0.5]
            self.norm_std = [0.5, 0.5, 0.5]
        else:
            self.norm_mean = [0.5]
            self.norm_std = [0.5]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])
        self.aug_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1) if self.channels == 3 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])

        self.mask_gen = PolynomialFractureMaskGenerator(img_size)
        print(f"  Generating {mask_cache_size} polynomial fracture masks...")
        self.mask_cache = [self.mask_gen.generate_tensor() for _ in range(mask_cache_size)]
        print(f"  Mask cache ready.")

    def __len__(self):
        return len(self.image_files)
    

    def _colorize(self, gray_pil):
        g = np.array(gray_pil).astype(np.float32) / 255.0
        r = np.clip(g * 0.82 + 0.18, 0, 1)
        green = np.clip(g * 0.71 + 0.11, 0, 1)
        b = np.clip(g * 0.56 + 0.05, 0, 1)
        rgb = (np.stack([r, green, b], axis=-1) * 255).astype(np.uint8)
        return Image.fromarray(rgb)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        if self.color_mode == "RGB":
            image = self._colorize(image)

        if self.augment:
            gt_tensor = self.aug_transform(image)
        else:
            gt_tensor = self.transform(image)

        mask = random.choice(self.mask_cache)
        masked_image = gt_tensor * (1 - mask) + mask * 1.0
        return {
            'gt': gt_tensor,
            'masked': masked_image,
            'mask': mask,
            'name': self.image_files[idx]
        }

def prepare_data(color_mode="GRAY"):
    print(f"\nPreparing Data: {color_mode}")
    all_files = sorted([
        f for f in os.listdir(cfg.TRAIN_IMG_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_aug' not in f.lower()
    ])
    print(f"Total non-augmented images: {len(all_files)}")

    rng = random.Random(cfg.SEED)
    rng.shuffle(all_files)

    n = len(all_files)
    n_test = max(int(n * 0.15), 4)
    n_val = max(int(n * 0.10), 4)
    n_train = n - n_test - n_val

    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    print(f"Split: {n_train} train, {n_val} val, {n_test} test")

    train_ds = HarappanSealDataset(
        cfg.TRAIN_IMG_DIR, train_files, color_mode, cfg.IMG_SIZE,
        cfg.MASK_CACHE_SIZE, augment=True
    )
    val_ds = HarappanSealDataset(
        cfg.TRAIN_IMG_DIR, val_files, color_mode, cfg.IMG_SIZE,
        512, augment=False
    )
    test_ds = HarappanSealDataset(
        cfg.TRAIN_IMG_DIR, test_files, color_mode, cfg.IMG_SIZE,
        512, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    return train_loader, val_loader, test_loader, test_files

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, norm=True, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual, inplace=True)

class BaselineUNet(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        c = channels
        self.enc1 = nn.Sequential(ConvBNReLU(c, 64), ConvBNReLU(64, 64))
        self.enc2 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.enc3 = nn.Sequential(ConvBNReLU(128, 256), ConvBNReLU(256, 256))
        self.enc4 = nn.Sequential(ConvBNReLU(256, 512), ConvBNReLU(512, 512))
        self.bottleneck = nn.Sequential(ConvBNReLU(512, 1024), ConvBNReLU(1024, 1024))
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = nn.Sequential(ConvBNReLU(1024, 512), ConvBNReLU(512, 512))
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = nn.Sequential(ConvBNReLU(512, 256), ConvBNReLU(256, 256))
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = nn.Sequential(ConvBNReLU(256, 128), ConvBNReLU(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Sequential(ConvBNReLU(128, 64), ConvBNReLU(64, 64))
        self.final = nn.Conv2d(64, c, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.tanh(self.final(d1))

class DeepResUNet(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        c = channels
        self.enc1 = nn.Sequential(ConvBNReLU(c, 64), ResidualBlock(64, use_se=True))
        self.enc2 = nn.Sequential(ConvBNReLU(64, 128), ResidualBlock(128, use_se=True))
        self.enc3 = nn.Sequential(ConvBNReLU(128, 256), ResidualBlock(256, use_se=True))
        self.enc4 = nn.Sequential(ConvBNReLU(256, 512), ResidualBlock(512, use_se=True))
        self.bottleneck = nn.Sequential(
            ConvBNReLU(512, 1024),
            ResidualBlock(1024, use_se=True),
            ResidualBlock(1024, use_se=True)
        )
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = nn.Sequential(ConvBNReLU(1024, 512), ResidualBlock(512, use_se=True))
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = nn.Sequential(ConvBNReLU(512, 256), ResidualBlock(256, use_se=True))
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = nn.Sequential(ConvBNReLU(256, 128), ResidualBlock(128, use_se=True))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Sequential(ConvBNReLU(128, 64), ResidualBlock(64, use_se=True))
        self.final = nn.Conv2d(64, c, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.tanh(self.final(d1))

class ContextEncoder(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        c = channels
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, c, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Pix2PixGenerator(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        c = channels
        self.e1 = nn.Sequential(nn.Conv2d(c, 64, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True))
        self.e3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True))
        self.e4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        self.e5 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        self.e6 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        self.e7 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.d7 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.Dropout(0.5), nn.ReLU(True))
        self.d6 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.Dropout(0.5), nn.ReLU(True))
        self.d5 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.Dropout(0.5), nn.ReLU(True))
        self.d4 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.d3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.d2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.d1 = nn.Sequential(nn.ConvTranspose2d(128, c, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        d7 = self.d7(e7)
        d6 = self.d6(torch.cat([d7, e6], 1))
        d5 = self.d5(torch.cat([d6, e5], 1))
        d4 = self.d4(torch.cat([d5, e4], 1))
        d3 = self.d3(torch.cat([d4, e3], 1))
        d2 = self.d2(torch.cat([d3, e2], 1))
        d1 = self.d1(torch.cat([d2, e1], 1))
        return d1

class PatchDiscriminator(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        c = channels
        self.net = nn.Sequential(
            nn.Conv2d(c * 2, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 1, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, a, b):
        return self.net(torch.cat([a, b], 1))

class RFAModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, dilation=1), nn.ReLU(True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 2, dilation=2), nn.ReLU(True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 4, dilation=4), nn.ReLU(True)
        )
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.se = SEBlock(channels)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = self.fuse(torch.cat([b1, b2, b3], 1))
        return self.se(fused) + x

class RFANet(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        c = channels
        self.encoder = nn.Sequential(
            ConvBNReLU(c, 64), RFAModule(64),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128), RFAModule(128),
            nn.MaxPool2d(2),
            ConvBNReLU(128, 256), RFAModule(256),
            nn.MaxPool2d(2),
        )
        self.bottleneck = nn.Sequential(
            ConvBNReLU(256, 512), RFAModule(512), RFAModule(512)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            RFAModule(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            RFAModule(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            RFAModule(64),
        )
        self.final = nn.Sequential(nn.Conv2d(64, c, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        enc = self.encoder(x)
        bot = self.bottleneck(enc)
        dec = self.decoder(bot)
        return self.final(dec)

class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.weight.requires_grad = False
        self.mask_window_area = kernel_size * kernel_size * in_channels
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, mask):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        mask = torch.nan_to_num(mask, nan=0.0)

        output = self.conv(x * mask)
        with torch.no_grad():
            mask_output = self.mask_conv(mask)
            mask_ratio = self.mask_window_area / (mask_output + 1e-6)
            mask_ratio = mask_ratio * (mask_output > 0).float()
            mask_ratio = torch.clamp(mask_ratio, 0.0, self.mask_window_area)

        output = output * mask_ratio
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        new_mask = (mask_output > 0).float()
        return output, new_mask

class PConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, bn=True, act=True):
        super().__init__()
        self.pconv = PartialConv2d(in_c, out_c, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_c) if bn else None
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x, mask

class PConvUNet(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        c = channels
        self.enc1 = PConvBlock(c, 64, 7, 2, 3, bn=False)
        self.enc2 = PConvBlock(64, 128, 5, 2, 2)
        self.enc3 = PConvBlock(128, 256, 3, 2, 1)
        self.enc4 = PConvBlock(256, 512, 3, 2, 1)
        self.dec4 = PConvBlock(512 + 256, 256, 3, 1, 1)
        self.dec3 = PConvBlock(256 + 128, 128, 3, 1, 1)
        self.dec2 = PConvBlock(128 + 64, 64, 3, 1, 1)
        self.dec1 = PConvBlock(64 + c, 64, 3, 1, 1)
        self.final = nn.Conv2d(64, c, 1)

    def forward(self, x, mask):
        valid_mask = 1 - mask
        if valid_mask.shape[1] != x.shape[1]:
            valid_mask = valid_mask.expand_as(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        e1, m1 = self.enc1(x, valid_mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)

        d4 = F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        m4_up = F.interpolate(m4, size=m3.shape[2:], mode='nearest')
        d4, m4_d = self.dec4(torch.cat([d4, e3], 1), torch.cat([m4_up, m3], 1))

        d3 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=False)
        m3_up = F.interpolate(m4_d, size=m2.shape[2:], mode='nearest')
        d3, m3_d = self.dec3(torch.cat([d3, e2], 1), torch.cat([m3_up, m2], 1))

        d2 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        m2_up = F.interpolate(m3_d, size=m1.shape[2:], mode='nearest')
        d2, m2_d = self.dec2(torch.cat([d2, e1], 1), torch.cat([m2_up, m1], 1))

        d1 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False)
        m1_up = F.interpolate(m2_d, size=valid_mask.shape[2:], mode='nearest')
        d1, m1_d = self.dec1(torch.cat([d1, x], 1), torch.cat([m1_up, valid_mask], 1))

        out = torch.tanh(self.final(d1))
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class TimeConditionedBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.conv1 = ConvBNReLU(in_c, out_c)
        self.conv2 = ConvBNReLU(out_c, out_c)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_c))
        self.res_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb
        h = self.conv2(h)
        return h + self.res_conv(x)

class DiffusionUNet(nn.Module):
    def __init__(self, channels=1, time_dim=256):
        super().__init__()
        c = channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc1 = TimeConditionedBlock(c * 2, 64, time_dim) 
        self.enc2 = TimeConditionedBlock(64, 128, time_dim)
        self.enc3 = TimeConditionedBlock(128, 256, time_dim)
        self.bot = TimeConditionedBlock(256, 512, time_dim)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = TimeConditionedBlock(512, 256, time_dim)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = TimeConditionedBlock(256, 128, time_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = TimeConditionedBlock(128, 64, time_dim)
        self.final = nn.Conv2d(64, c, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        e1 = self.enc1(x, t)
        e2 = self.enc2(F.max_pool2d(e1, 2), t)
        e3 = self.enc3(F.max_pool2d(e2, 2), t)
        b = self.bot(F.max_pool2d(e3, 2), t)
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1), t)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1), t)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1), t)
        return self.final(d1)

class GaussianDiffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n):
        return torch.randint(1, self.timesteps, (n,), device=device)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def sample(self, model, gt, mask, n_steps=50):
        b = gt.shape[0]
        cond = gt * (1 - mask) + 1.0 * mask
        x = torch.randn_like(gt)
        x = gt * (1 - mask) + x * mask
        step_size = max(1, self.timesteps // n_steps)
        timesteps = list(range(self.timesteps - 1, 0, -step_size))
        for i, t_val in enumerate(timesteps):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)
            pred_x0 = model(torch.cat([x, cond], dim=1), t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            if t_val > step_size:
                t_prev = max(t_val - step_size, 0)
                alpha_hat_prev = self.alpha_hat[t_prev].view(1, 1, 1, 1).expand(b, 1, 1, 1)
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_hat_prev) * pred_x0 + torch.sqrt(1 - alpha_hat_prev) * noise
            else:
                x = pred_x0
            if i + 1 < len(timesteps):
                t_next = timesteps[i + 1]
                gt_noised, _ = self.noise_images(gt, torch.full((b,), t_next, device=device, dtype=torch.long))
            else:
                gt_noised = gt
            x = gt_noised * (1 - mask) + x * mask
        return x

class PerceptualLoss(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.channels = channels

    def forward(self, pred, target):
        if self.channels == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        pred = (pred * 0.5 + 0.5 - mean) / std
        target = (target * 0.5 + 0.5 - mean) / std
        return F.l1_loss(self.features(pred), self.features(target))

class CombinedLoss(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss(channels)
        self.channels = channels

    def forward(self, pred, target, mask):
        hole_loss = self.l1(pred * mask, target * mask)
        valid_loss = self.l1(pred * (1 - mask), target * (1 - mask))
        perc_loss = self.perceptual(pred, target)
        return valid_loss * 1.0 + hole_loss * 6.0 + perc_loss * 0.1

class MetricCalculator:
    def __init__(self):
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()

    @torch.no_grad()
    def compute(self, pred, target, mask=None):
        pred_01 = torch.clamp((pred + 1) / 2, 0, 1)
        target_01 = torch.clamp((target + 1) / 2, 0, 1)

        if pred.shape[1] == 1:
            lpips_pred = pred.repeat(1, 3, 1, 1)
            lpips_target = target.repeat(1, 3, 1, 1)
        else:
            lpips_pred = pred
            lpips_target = target

        lpips_val = self.lpips_fn(lpips_pred, lpips_target).mean().item()

        pred_np = pred_01.cpu().numpy()
        target_np = target_01.cpu().numpy()
        psnr_list = []
        ssim_list = []
        bs = pred.shape[0]

        for i in range(bs):
            p = pred_np[i].transpose(1, 2, 0).squeeze()
            t = target_np[i].transpose(1, 2, 0).squeeze()
            psnr_list.append(sk_psnr(t, p, data_range=1.0))
            if p.ndim == 2:
                ssim_list.append(sk_ssim(t, p, data_range=1.0))
            else:
                ssim_list.append(sk_ssim(t, p, data_range=1.0, channel_axis=2))

        return {
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'LPIPS': lpips_val
        }

MODEL_REGISTRY = OrderedDict([
     ("Baseline_UNet", BaselineUNet),
     ("DeepRes_UNet", DeepResUNet),
     ("Context_Encoder", ContextEncoder),
     ("Pix2Pix", Pix2PixGenerator),
     ("RFA_Net", RFANet),
     ("PConv_UNet", PConvUNet),
    ("Guided_DDPM", DiffusionUNet),
])

MODEL_COLOR_MAP = {
    "Baseline_UNet":    "#3498DB",
    "DeepRes_UNet":     "#2ECC71",
    "Context_Encoder":  "#E74C3C",
    "Pix2Pix":          "#F39C12",
    "RFA_Net":          "#9B59B6",
    "PConv_UNet":       "#1ABC9C",
    "Guided_DDPM":      "#E67E22",
}

MODEL_LR = {
    "Baseline_UNet": 2e-4,
    "DeepRes_UNet": 2e-4,
    "Context_Encoder": 2e-4,
    "Pix2Pix": 2e-4,
    "RFA_Net": 2e-4,
    "PConv_UNet": 5e-5,
    "Guided_DDPM": 2e-4,
}

def train_model(model_name, model_class, train_loader, val_loader, channels, mode_dir):
    print(f"\nTraining: {model_name} | Channels: {channels}")
    ckpt_dir = os.path.join(mode_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pth")

    if os.path.exists(ckpt_path) and not cfg.FORCE_RETRAIN:
        print(f"  Checkpoint exists. Skipping.")
        return

    is_diffusion = (model_name == "Guided_DDPM")
    is_pix2pix = (model_name == "Pix2Pix")
    is_pconv = (model_name == "PConv_UNet")

    model = model_class(channels=channels).to(device)
    print(f"  Parameters: {count_parameters(model):,}")
    model_lr = MODEL_LR.get(model_name, cfg.LR)
    opt_g = optim.AdamW(model.parameters(), lr=model_lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler_g = CosineAnnealingLR(opt_g, T_max=cfg.EPOCHS, eta_min=1e-6)
    if is_diffusion:
        criterion = nn.MSELoss()
        diffusion = GaussianDiffusion(timesteps=200)
    else:
        criterion = CombinedLoss(channels).to(device)
    disc = None
    opt_d = None
    scheduler_d = None
    if is_pix2pix:
        disc = PatchDiscriminator(channels=channels).to(device)
        opt_d = optim.AdamW(disc.parameters(), lr=model_lr, betas=(0.5, 0.999), weight_decay=1e-4)
        scheduler_d = CosineAnnealingLR(opt_d, T_max=cfg.EPOCHS, eta_min=1e-6)
        gan_criterion = nn.MSELoss()
    scaler = GradScaler("cuda", enabled=True)
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_lpips': []}
    best_val_psnr = 0
    patience = 30
    patience_counter = 0
    metrics_calc = MetricCalculator()
    consecutive_nan = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        if disc:
            disc.train()
        epoch_loss_g = 0
        valid_batches = 0

        loop = tqdm(train_loader, desc=f"  Ep {epoch+1}/{cfg.EPOCHS}", leave=False, ncols=100)
        for batch in loop:
            gt = batch['gt'].to(device, non_blocking=True)
            masked = batch['masked'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)

            with autocast("cuda", enabled=True):
                if is_diffusion:
                    t = diffusion.sample_timesteps(gt.shape[0])

                    x_t, noise = diffusion.noise_images(gt, t)
                    pred_x0 = model(torch.cat([x_t, masked], dim=1), t)
                    loss_g = criterion(pred_x0, gt)

                elif is_pconv:
                    pred = model(masked, mask)
                    loss_g = criterion(pred, gt, mask)
                else:
                    pred = model(masked)
                    loss_g = criterion(pred, gt, mask)

                if is_pix2pix and epoch >= 5:
                    fake_pred_d = disc(masked, pred)
                    loss_gan = gan_criterion(fake_pred_d, torch.ones_like(fake_pred_d))
                    loss_g = loss_g + loss_gan * 0.1
            if torch.isnan(loss_g) or torch.isinf(loss_g):
                opt_g.zero_grad(set_to_none=True)
                scaler.update()
                continue

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt_g)

            if is_pix2pix and disc and epoch >= 5:
                with autocast("cuda", enabled=True):
                    real_pred_d = disc(masked, gt)
                    fake_pred_d = disc(masked, pred.detach())
                    loss_d_real = gan_criterion(real_pred_d, torch.ones_like(real_pred_d))
                    loss_d_fake = gan_criterion(fake_pred_d, torch.zeros_like(fake_pred_d))
                    loss_d = (loss_d_real + loss_d_fake) * 0.5
                opt_d.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()
                scaler.step(opt_d)

            scaler.update()
            epoch_loss_g += loss_g.item()
            valid_batches += 1
            loop.set_postfix(loss=f"{loss_g.item():.4f}")

        scheduler_g.step()
        if scheduler_d:
            scheduler_d.step()

        avg_loss = epoch_loss_g / valid_batches if valid_batches > 0 else float('nan')
        history['train_loss'].append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == cfg.EPOCHS - 1:
            model.eval()
            val_metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}

            with torch.no_grad():
                for val_batch in val_loader:
                    v_gt = val_batch['gt'].to(device)
                    v_masked = val_batch['masked'].to(device)
                    v_mask = val_batch['mask'].to(device)

                    if is_diffusion:
                        v_pred = diffusion.sample(model, v_gt, v_mask, n_steps=50)
                    elif is_pconv:
                        v_pred = model(v_masked, v_mask)
                    else:
                        v_pred = model(v_masked)
                    if torch.isnan(v_pred).any():
                        continue

                    v_final = v_gt * (1 - v_mask) + v_pred * v_mask
                    m = metrics_calc.compute(v_final, v_gt)
                    val_metrics['PSNR'].append(m['PSNR'])
                    val_metrics['SSIM'].append(m['SSIM'])
                    val_metrics['LPIPS'].append(m['LPIPS'])

            if len(val_metrics['PSNR']) == 0:
                avg_psnr = avg_ssim = avg_lpips = float('nan')
                consecutive_nan += 1
            else:
                avg_psnr = np.mean(val_metrics['PSNR'])
                avg_ssim = np.mean(val_metrics['SSIM'])
                avg_lpips = np.mean(val_metrics['LPIPS'])
                consecutive_nan = 0

            history['val_psnr'].append(avg_psnr if not math.isnan(avg_psnr) else 0)
            history['val_ssim'].append(avg_ssim if not math.isnan(avg_ssim) else 0)
            history['val_lpips'].append(avg_lpips if not math.isnan(avg_lpips) else 1)

            print(f"  Ep {epoch+1}: Loss={avg_loss:.4f} | PSNR={avg_psnr:.2f} | SSIM={avg_ssim:.4f} | LPIPS={avg_lpips:.4f}")

            if not math.isnan(avg_psnr) and avg_psnr > best_val_psnr:
                best_val_psnr = avg_psnr
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_psnr': best_val_psnr,
                    'history': history,
                }, ckpt_path)
                patience_counter = 0
                print(f"    Best model saved (PSNR={best_val_psnr:.2f})")
            else:
                patience_counter += 1
            if consecutive_nan >= 30:
                print(f"  Early stopping at epoch {epoch+1} (persistent NaN)")
                break

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    save_training_curves(model_name, history, mode_dir)

    del model, opt_g, scaler, criterion
    if disc:
        del disc, opt_d
    torch.cuda.empty_cache()
    gc.collect()
    return history

def save_training_curves(model_name, history, mode_dir):
    fig_dir = os.path.join(mode_dir, "training_curves")
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history['train_loss'], 'b-', linewidth=1.5)
    axes[0].set_title(f'{model_name} Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    if history['val_psnr']:
        x_vals = list(range(4, len(history['val_psnr']) * 5, 5))[:len(history['val_psnr'])]
        if len(x_vals) != len(history['val_psnr']):
            x_vals = list(range(len(history['val_psnr'])))

        axes[1].plot(x_vals, history['val_psnr'], 'g-o', markersize=3, linewidth=1.5)
        axes[1].set_title('Validation PSNR (dB)')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x_vals, history['val_ssim'], 'r-o', markersize=3, linewidth=1.5)
        axes[2].set_title('Validation SSIM')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(x_vals, history['val_lpips'], 'm-o', markersize=3, linewidth=1.5)
        axes[3].set_title('Validation LPIPS (lower=better)')
        axes[3].set_xlabel('Epoch')
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{model_name}_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_all_models(test_loader, channels, mode_dir, color_mode_name):
    print(f"\nEvaluation: {color_mode_name}")

    ckpt_dir = os.path.join(mode_dir, "checkpoints")
    results_dir = os.path.join(mode_dir, "results")
    fig_dir = os.path.join(mode_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    metrics_calc = MetricCalculator()
    diffusion = GaussianDiffusion(timesteps=200)

    results = []
    all_visuals = {}

    all_test_data = []
    for batch in test_loader:
        all_test_data.append(batch)

    for model_name, model_class in MODEL_REGISTRY.items():
        print(f"\n  Evaluating: {model_name}")
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pth")
        is_diffusion = (model_name == "Guided_DDPM")
        is_pconv = (model_name == "PConv_UNet")

        model = model_class(channels=channels).to(device)

        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                    print(f"    Loaded from epoch {ckpt.get('epoch', '?')}, best PSNR={ckpt.get('best_psnr', '?'):.2f}")
                else:
                    model.load_state_dict(ckpt)
            except Exception as e:
                print(f"    Error loading checkpoint: {e}")
                print(f"    Using random weights.")
        else:
            print(f"    No checkpoint found. Using random weights.")

        model.eval()

        batch_metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}
        model_visuals = []
        total_time = 0
        total_images = 0

        with torch.no_grad():
            for batch in all_test_data:
                gt = batch['gt'].to(device)
                masked = batch['masked'].to(device)
                mask = batch['mask'].to(device)

                start_t = time.time()
                if is_diffusion:
                    pred = diffusion.sample(model, gt, mask, n_steps=50)
                elif is_pconv:
                    pred = model(masked, mask)
                else:
                    pred = model(masked)
                elapsed = time.time() - start_t
                total_time += elapsed
                total_images += gt.shape[0]
                if torch.isnan(pred).any():
                    pred = torch.nan_to_num(pred, nan=0.0)

                final = gt * (1 - mask) + pred * mask
                m = metrics_calc.compute(final, gt)
                batch_metrics['PSNR'].append(m['PSNR'])
                batch_metrics['SSIM'].append(m['SSIM'])
                batch_metrics['LPIPS'].append(m['LPIPS'])

                for j in range(gt.shape[0]):
                    model_visuals.append({
                        'gt': gt[j].cpu(),
                        'masked': masked[j].cpu(),
                        'pred': pred[j].cpu(),
                        'final': final[j].cpu(),
                        'mask': mask[j].cpu(),
                        'name': batch['name'][j]
                    })

        avg_psnr = np.mean(batch_metrics['PSNR'])
        avg_ssim = np.mean(batch_metrics['SSIM'])
        avg_lpips = np.mean(batch_metrics['LPIPS'])
        avg_time = total_time / max(total_images, 1)

        results.append({
            'Model': model_name,
            'PSNR (dB)': round(avg_psnr, 2),
            'SSIM': round(avg_ssim, 4),
            'LPIPS': round(avg_lpips, 4),
            'Params (M)': round(count_parameters(model) / 1e6, 2),
            'Inf. Time (s)': round(avg_time, 3),
        })
        all_visuals[model_name] = model_visuals
        print(f"    PSNR={avg_psnr:.2f} | SSIM={avg_ssim:.4f} | LPIPS={avg_lpips:.4f}")

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df = df.sort_values('PSNR (dB)', ascending=False).reset_index(drop=True)
    csv_path = os.path.join(results_dir, f"metrics_{color_mode_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Metrics saved to {csv_path}")
    print(df.to_string(index=False))
    generate_comparison_figure(all_visuals, df, channels, fig_dir, color_mode_name)
    generate_metrics_table_figure(df, fig_dir, color_mode_name)
    generate_mask_examples_figure(fig_dir)
    generate_per_sample_figure(all_visuals, channels, fig_dir, color_mode_name)
    generate_bar_chart(df, fig_dir, color_mode_name)

    return df, all_visuals

def tensor_to_image(t, channels):
    img = torch.clamp((t + 1) / 2, 0, 1)
    img = torch.nan_to_num(img, nan=0.5)
    if channels == 1:
        return (img.squeeze().numpy() * 255).astype(np.uint8)
    else:
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def generate_comparison_figure(all_visuals, df, channels, fig_dir, mode_name):
    print("  Generating comparison figure")
    model_names = list(MODEL_REGISTRY.keys())
    n_models = len(model_names)
    n_samples = min(cfg.NUM_VIS_SAMPLES, len(all_visuals[model_names[0]]))
    n_cols = n_models + 2
    n_rows = n_samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = ['Ground Truth', 'Damaged Input'] + [
        n.replace('_', '\n') for n in model_names
    ]

    for row in range(n_rows):
        vis = all_visuals[model_names[0]][row]
        gt_img = tensor_to_image(vis['gt'], channels)
        masked_img = tensor_to_image(vis['masked'], channels)
        cmap = 'gray' if channels == 1 else None

        axes[row, 0].imshow(gt_img, cmap=cmap)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title(col_headers[0], fontsize=9, fontweight='bold')

        axes[row, 1].imshow(masked_img, cmap=cmap)
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title(col_headers[1], fontsize=9, fontweight='bold')

        for col_idx, m_name in enumerate(model_names):
            vis = all_visuals[m_name][row]
            final_img = tensor_to_image(vis['final'], channels)
            ax = axes[row, col_idx + 2]
            ax.imshow(final_img, cmap=cmap)
            ax.axis('off')
            if row == 0:
                ax.set_title(col_headers[col_idx + 2], fontsize=8, fontweight='bold')

    plt.suptitle(f'Inpainting Comparison {mode_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"comparison_{mode_name}.png"),
        dpi=300, bbox_inches='tight', facecolor='white'
    )
    plt.savefig(
        os.path.join(fig_dir, f"comparison_{mode_name}.pdf"),
        bbox_inches='tight', facecolor='white'
    )
    plt.close()

def generate_per_sample_figure(all_visuals, channels, fig_dir, mode_name):
    print("  Generating per-sample detail figure")
    metrics_calc_local = MetricCalculator()
    best_model = list(MODEL_REGISTRY.keys())[0]
    best_psnr = 0

    for m_name in MODEL_REGISTRY.keys():
        if m_name in all_visuals and len(all_visuals[m_name]) > 0:
            vis = all_visuals[m_name][0]
            try:
                m = metrics_calc_local.compute(
                    vis['final'].unsqueeze(0).to(device),
                    vis['gt'].unsqueeze(0).to(device)
                )
                if m['PSNR'] > best_psnr:
                    best_psnr = m['PSNR']
                    best_model = m_name
            except Exception:
                pass

    print(f"    Best model for detail: {best_model}")
    n_samples = min(4, len(all_visuals[best_model]))

    fig, axes = plt.subplots(n_samples, 5, figsize=(15, n_samples * 3))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    headers = ['Ground Truth', 'Mask', 'Damaged', f'Restored\n({best_model})', 'Difference Map']
    cmap = 'gray' if channels == 1 else None

    for i in range(n_samples):
        vis = all_visuals[best_model][i]
        gt_img = tensor_to_image(vis['gt'], channels)
        mask_img = vis['mask'].squeeze().numpy()
        masked_img = tensor_to_image(vis['masked'], channels)
        final_img = tensor_to_image(vis['final'], channels)

        if channels == 1:
            diff = np.abs(gt_img.astype(float) - final_img.astype(float))
        else:
            diff = np.abs(gt_img.astype(float) - final_img.astype(float)).mean(axis=2)
        diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)

        axes[i, 0].imshow(gt_img, cmap=cmap)
        axes[i, 1].imshow(mask_img, cmap='gray')
        axes[i, 2].imshow(masked_img, cmap=cmap)
        axes[i, 3].imshow(final_img, cmap=cmap)
        axes[i, 4].imshow(diff, cmap='hot')

        for j in range(5):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(headers[j], fontsize=10, fontweight='bold')

    plt.suptitle(f'Detailed Restoration Results {best_model} ({mode_name})',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"detail_{mode_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, f"detail_{mode_name}.pdf"), bbox_inches='tight')
    plt.close()

def generate_metrics_table_figure(df, fig_dir, mode_name):
    print("  Generating metrics table figure")
    fig, ax = plt.subplots(figsize=(12, 3 + len(df) * 0.5))
    ax.axis('off')

    table_data = df.values.tolist()
    col_labels = df.columns.tolist()

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    psnr_col = col_labels.index('PSNR (dB)')
    ssim_col = col_labels.index('SSIM')
    lpips_col = col_labels.index('LPIPS')

    psnr_vals = [row[psnr_col] for row in table_data]
    ssim_vals = [row[ssim_col] for row in table_data]
    lpips_vals = [row[lpips_col] for row in table_data]

    best_psnr_idx = int(np.argmax(psnr_vals))
    best_ssim_idx = int(np.argmax(ssim_vals))
    best_lpips_idx = int(np.argmin(lpips_vals))

    table[best_psnr_idx + 1, psnr_col].set_facecolor('#C6EFCE')
    table[best_ssim_idx + 1, ssim_col].set_facecolor('#C6EFCE')
    table[best_lpips_idx + 1, lpips_col].set_facecolor('#C6EFCE')

    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            cell_color = table[i+1, j].get_facecolor()
                if not (abs(cell_color[0] - 0.776) < 0.01):
                    if i % 2 == 0:
                        table[i+1, j].set_facecolor('#F2F2F2')

    plt.title(f'Quantitative Results {mode_name}', fontsize=13, fontweight='bold', pad=20)
    plt.savefig(os.path.join(fig_dir, f"table_{mode_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, f"table_{mode_name}.pdf"), bbox_inches='tight')
    plt.close()

def generate_bar_chart(df, fig_dir, mode_name):
    print("  Generating bar charts")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = [MODEL_COLOR_MAP.get(m, "#95A5A6") for m in df['Model']]

    bars = axes[0].bar(range(len(df)), df['PSNR (dB)'], color=colors)
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR Comparison (higher=better)', fontweight='bold')
    for bar, val in zip(bars, df['PSNR (dB)']):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    bars = axes[1].bar(range(len(df)), df['SSIM'], color=colors)
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM Comparison (higher=better)', fontweight='bold')
    for bar, val in zip(bars, df['SSIM']):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    bars = axes[2].bar(range(len(df)), df['LPIPS'], color=colors)
    axes[2].set_xticks(range(len(df)))
    axes[2].set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel('LPIPS')
    axes[2].set_title('LPIPS Comparison (lower=better)', fontweight='bold')
    for bar, val in zip(bars, df['LPIPS']):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle(f'Model Performance Comparison {mode_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"barchart_{mode_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, f"barchart_{mode_name}.pdf"), bbox_inches='tight')
    plt.close()

def generate_mask_examples_figure(fig_dir):
    print("  Generating mask examples figure")
    mask_gen = PolynomialFractureMaskGenerator(cfg.IMG_SIZE)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        row = i // 5
        col = i % 5
        mask = mask_gen.generate()
        ratio = mask.sum() / (cfg.IMG_SIZE * cfg.IMG_SIZE) * 100
        axes[row, col].imshow(mask, cmap='gray')
        axes[row, col].set_title(f'Damage: {ratio:.1f}%', fontsize=9)
        axes[row, col].axis('off')
    plt.suptitle('Polynomial Fracture Mask Examples\n(Simulating Real Seal Breakage Patterns)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mask_examples.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, "mask_examples.pdf"), bbox_inches='tight')
    plt.close()

def colorize_results(gray_visuals, fig_dir):
    print("\n  Generating artificially colorized figures")

    def apply_warm_colormap(gray_img):
        img = gray_img.astype(np.uint8)
        colored = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return colored

    def apply_sepia(gray_img):
        img = gray_img.astype(np.float64)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = img @ sepia_filter.T
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return sepia_img

    model_names = list(MODEL_REGISTRY.keys())
    n_samples = min(4, len(gray_visuals[model_names[0]]))
    n_cols = len(model_names) + 2
    col_headers = ['Ground Truth', 'Damaged'] + [n.replace('_', '\n') for n in model_names]

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 2.5, n_samples * 2.5))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_samples):
        vis = gray_visuals[model_names[0]][row]
        gt_gray = tensor_to_image(vis['gt'], 1)
        masked_gray = tensor_to_image(vis['masked'], 1)
        axes[row, 0].imshow(apply_sepia(gt_gray))
        axes[row, 0].axis('off')
        axes[row, 1].imshow(apply_sepia(masked_gray))
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 0].set_title(col_headers[0], fontsize=9, fontweight='bold')
            axes[row, 1].set_title(col_headers[1], fontsize=9, fontweight='bold')
        for col_idx, m_name in enumerate(model_names):
            vis = gray_visuals[m_name][row]
            final_gray = tensor_to_image(vis['final'], 1)
            axes[row, col_idx + 2].imshow(apply_sepia(final_gray))
            axes[row, col_idx + 2].axis('off')
            if row == 0:
                axes[row, col_idx + 2].set_title(col_headers[col_idx + 2], fontsize=8, fontweight='bold')

    plt.suptitle('Artificially Colorized Results (Sepia Tone)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "comparison_COLORIZED_sepia.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, "comparison_COLORIZED_sepia.pdf"), bbox_inches='tight')
    plt.close()
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 2.5, n_samples * 2.5))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_samples):
        vis = gray_visuals[model_names[0]][row]
        gt_gray = tensor_to_image(vis['gt'], 1)
        masked_gray = tensor_to_image(vis['masked'], 1)
        axes[row, 0].imshow(apply_warm_colormap(gt_gray))
        axes[row, 0].axis('off')
        axes[row, 1].imshow(apply_warm_colormap(masked_gray))
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 0].set_title(col_headers[0], fontsize=9, fontweight='bold')
            axes[row, 1].set_title(col_headers[1], fontsize=9, fontweight='bold')
        for col_idx, m_name in enumerate(model_names):
            vis = gray_visuals[m_name][row]
            final_gray = tensor_to_image(vis['final'], 1)
            axes[row, col_idx + 2].imshow(apply_warm_colormap(final_gray))
            axes[row, col_idx + 2].axis('off')
            if row == 0:
                axes[row, col_idx + 2].set_title(col_headers[col_idx + 2], fontsize=8, fontweight='bold')

    plt.suptitle('Artificially Colorized Results (Warm Tone)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "comparison_COLORIZED_warm.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, "comparison_COLORIZED_warm.pdf"), bbox_inches='tight')
    plt.close()

def generate_combined_table(gray_df, rgb_df, fig_dir):
    print("\n  Generating combined comparison table")
    if gray_df is not None and rgb_df is not None:
        gray_df_renamed = gray_df.copy()
        rgb_df_renamed = rgb_df.copy()
        gray_df_renamed.columns = [c + ' (Gray)' if c != 'Model' else c for c in gray_df_renamed.columns]
        rgb_df_renamed.columns = [c + ' (RGB)' if c != 'Model' else c for c in rgb_df_renamed.columns]
        combined = gray_df_renamed.merge(rgb_df_renamed, on='Model', how='outer')
        csv_path = os.path.join(fig_dir, "combined_metrics.csv")
        combined.to_csv(csv_path, index=False)
        print(f"    Combined table saved to {csv_path}")
        print(combined.to_string(index=False))
        return combined
    elif gray_df is not None:
        csv_path = os.path.join(fig_dir, "metrics_GRAY.csv")
        gray_df.to_csv(csv_path, index=False)
        return gray_df
    return rgb_df

def main():
    print(f"Output: {cfg.BASE_OUTPUT}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    os.makedirs(cfg.BASE_OUTPUT, exist_ok=True)

    all_results = {}

    for color_mode in ["GRAY", "RGB"]:
        print(f"\nMODE: {color_mode}")

        cfg.COLOR_MODE = color_mode
        channels = 3 if color_mode == "RGB" else 1
        mode_dir = os.path.join(cfg.BASE_OUTPUT, color_mode)
        os.makedirs(mode_dir, exist_ok=True)

        train_loader, val_loader, test_loader, test_files = prepare_data(color_mode)
        print(f"\nTraining all 7 models: {color_mode}")
        for model_name, model_class in MODEL_REGISTRY.items():
            try:
                train_model(
                    model_name, model_class,
                    train_loader, val_loader,
                    channels, mode_dir
                )
            except Exception as e:
                print(f"  ERROR training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        try:
            df, visuals = evaluate_all_models(
                test_loader, channels, mode_dir, color_mode
            )
            all_results[color_mode] = {'df': df, 'visuals': visuals}

            if color_mode == "GRAY":
                colorize_results(visuals, os.path.join(mode_dir, "figures"))
        except Exception as e:
            print(f"  ERROR evaluating: {e}")
            import traceback
            traceback.print_exc()

        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
    gray_df = all_results.get("GRAY", {}).get('df', None)
    rgb_df = all_results.get("RGB", {}).get('df', None)
    combined_fig_dir = os.path.join(cfg.BASE_OUTPUT, "combined_figures")
    os.makedirs(combined_fig_dir, exist_ok=True)
    generate_combined_table(gray_df, rgb_df, combined_fig_dir)
    if gray_df is not None and rgb_df is not None:
        print("\n  Generating side-by-side mode comparison")

        merged = gray_df.merge(rgb_df, on='Model', suffixes=('_gray', '_rgb'))
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        x = np.arange(len(merged))
        w = 0.35

        axes[0].bar(x - w/2, merged['PSNR (dB)_gray'], w, label='Grayscale', color='#3498DB')
        axes[0].bar(x + w/2, merged['PSNR (dB)_rgb'],  w, label='RGB',       color='#E74C3C')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(merged['Model'], rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR: Grayscale vs RGB', fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(x - w/2, merged['SSIM_gray'],       w, label='Grayscale', color='#3498DB')
        axes[1].bar(x + w/2, merged['SSIM_rgb'],        w, label='RGB',       color='#E74C3C')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(merged['Model'], rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('SSIM: Grayscale vs RGB', fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.suptitle('Grayscale vs RGB Mode Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(combined_fig_dir, "gray_vs_rgb.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(combined_fig_dir, "gray_vs_rgb.pdf"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()