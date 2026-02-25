# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib


# ============================================================
#                ATTENTION U-NET (VISUALIZABLE)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        self.attention_map = None

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        self.attention_map = psi.detach()

        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=11, out_channels=1):
        super().__init__()

        self.return_attention = False

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        #out = torch.sigmoid(self.final(d1))
        out = self.final(d1)

        if self.return_attention:
            return out, {
                "att1": self.att1.attention_map,
                "att2": self.att2.attention_map,
                "att3": self.att3.attention_map,
                "att4": self.att4.attention_map,
            }

        return out


# ============================================================
#                RANDOM FOREST BASELINE
# ============================================================

class RandomForestBaseline:
    """
    Pixel-wise Random Forest regression for SWE prediction.
    """

    def __init__(self, n_estimators=100, max_depth=None, n_jobs=-1):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=42
        )

    def fit(self, dataloader, subsample=1):
        """
        subsample: keep every Nth pixel to reduce memory
        """
        X_all = []
        y_all = []

        for X, Y, _ in dataloader:

            B, C, H, W = X.shape

            X_np = X.numpy().transpose(0, 2, 3, 1).reshape(-1, C)
            y_np = Y.numpy().reshape(-1)

            if subsample > 1:
                X_np = X_np[::subsample]
                y_np = y_np[::subsample]

            X_all.append(X_np)
            y_all.append(y_np)

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        print("Training Random Forest...")
        self.model.fit(X_all, y_all)

    def predict(self, X_tensor):
        B, C, H, W = X_tensor.shape
        X_np = X_tensor.numpy().transpose(0, 2, 3, 1).reshape(-1, C)

        preds = self.model.predict(X_np)
        preds = preds.reshape(B, 1, H, W)

        return torch.from_numpy(preds).float()

    def feature_importance(self):
        return self.model.feature_importances_

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)