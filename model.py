import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics
from ptflops import get_model_complexity_info
from melbanks.melbanks import LogMelFilterBanks

class SpeechCNN(pl.LightningModule):
    def __init__(self, n_mels=40, groups=1):
        super().__init__()
        self.save_hyperparameters()

        self.mel_banks = LogMelFilterBanks(n_mels = n_mels)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, 32, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.flatten_size = 32 * 12
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        
        self.train_loss = torchmetrics.MeanMetric()
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_accuracy = torchmetrics.Accuracy(task="binary")

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.log("epoch_time", epoch_time, on_epoch=True, prog_bar=True)
        self.log("train_loss_epoch", self.train_loss.compute(), on_epoch=True, prog_bar=True)
        self.train_loss.reset()
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.mel_banks(x)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.train_loss(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True)

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_flops(self):
        # Prepare dummy input
        input_shape = (1, 16000)  # 1 second at 16kHz
        flops, _ = get_model_complexity_info(
            self, input_shape, as_strings=False,
            print_per_layer_stat=False
        )
        return flops