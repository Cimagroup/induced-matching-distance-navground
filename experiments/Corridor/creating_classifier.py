###################################
###################################
# EXPERIMENTS: PART 3
# CREATING THE CLASSIFIER
###################################
###################################

### Origin of the code for the network 
### https://github.com/PacktPublishing/Deep-Learning-for-Time-Series-Data-Cookbook/blob/main/Chapter_8/8.4_resnet_tsc.py

###################################
# 1: IMPORTING MODULES
###################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import seaborn as sns
import argparse
import pickle

matching_signals = np.load('files/matching.npy')
types = np.load('files/types.npy')

###################################
# 2: ADJUSTING PARAMETERS
###################################

parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test dataset')
parser.add_argument('--val_size', type=float, default=0.1, help='Size of the validation dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the training')
parser.add_argument('--epochs', type=int, default=1000, help='Maximum number of training epochs')
parser.add_argument('--patience', type=int, default=10, help='Epochs without validation loss improvement until stop')

args = parser.parse_args([
        '--test_size', '0.2',
        '--val_size', '0.1',
        '--batch_size', '32',
        '--learning_rate', '0.001',
        '--epochs', '100',
        '--patience', '50'
    ])

###################################
# 3: DEFINING THE RESNET
###################################

class TSCDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class TSCDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, batch_size=1):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size

        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(categories="auto", sparse_output=False)

        self.train = None
        self.validation = None
        self.test = None

    def setup(self, stage=None):
        y_train = self.encoder.fit_transform(
            self.train_df.iloc[:, 0].values.reshape(-1, 1))
        y_test = self.encoder.transform(
            self.test_df.iloc[:, 0].values.reshape(-1, 1))

        X_train = train.iloc[:, 1:]
        X_test = test.iloc[:, 1:]

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=args.val_size/(1-args.test_size), stratify=y_train
        )

        X_train, X_val, X_test = [
            torch.tensor(arr, dtype=torch.float).unsqueeze(1)
            for arr in [X_train, X_val, X_test]
        ]
        y_train, y_val, y_test = [
            torch.tensor(arr, dtype=torch.long) for arr in [y_train, y_val, y_test]
        ]

        self.train = TSCDataset(X_train, y_train)
        self.validation = TSCDataset(X_val, y_val)
        self.test = TSCDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(
            input, self.weight, self.bias, self.stride, self.dilation, self.groups
        )


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = ((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups,
    )

class ConvBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class ResNNBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                )
                for i in range(len(kernel_sizes))
            ]
        )

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(
                *[
                    Conv1dSamePadding(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.BatchNorm1d(num_features=out_channels),
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)

class ResidualNeuralNetworkModel(nn.Module):
    def __init__(
        self, in_channels: int, mid_channels: int = 64, num_pred_classes: int = 1
    ) -> None:
        super().__init__()

        self.input_args = {
            "in_channels": in_channels,
            "num_pred_classes": num_pred_classes,
        }

        self.layers = nn.Sequential(
            *[
                ResNNBlock(in_channels=in_channels, out_channels=mid_channels),
                ResNNBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
                ResNNBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),
            ]
        )

        self.fc = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x):
        x = self.layers(x)
        out = self.fc(x.mean(dim=-1))
        # out = torch.sigmoid(out)

        return out

class TSCResNet(pl.LightningModule):
    def __init__(self, output_dim):
        super().__init__()

        self.resnet = ResidualNeuralNetworkModel(
            in_channels=1, num_pred_classes=output_dim
        )
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []

    def forward(self, x):
        out = self.resnet.forward(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        y_pred = self.forward(x)

        loss = F.cross_entropy(y_pred, y)
        self.train_loss.append(loss.item())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.val_loss.append(loss.item())
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        acc = Accuracy(task="multiclass", num_classes=3)
        acc_score = acc(y_pred, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=args.learning_rate)

###################################
# 4: BUILDING THE RESNET
###################################

data = pd.DataFrame(matching_signals)
data.insert(0, "label", types)
train, test = train_test_split(data, test_size=args.test_size, stratify=data['label'], random_state=42)
datamodule = TSCDataModule(train_df=train, test_df=test, batch_size=args.batch_size)
model = TSCResNet(output_dim=3)

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=args.patience, verbose=False, mode="min"
)

logger = TensorBoardLogger("tb_logs", name="model_training")

trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="cpu",  # Use "gpu" if you have a compatible GPU
    logger=False,
    log_every_n_steps=10,
    enable_model_summary=False,
    callbacks=[early_stop_callback],
    enable_progress_bar=True,
)

trainer.fit(model, datamodule)

torch.save(model.state_dict(), "files/resnet_model.pth")

#Plotting the loss along the training
plt.figure(figsize=(10, 6))
plt.plot(model.train_loss, label='Training Loss')
plt.plot(model.val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("plots/loss_tracking.png", dpi='figure', bbox_inches='tight')

###################################
# 4: EVALUATING THE RESNET
###################################

#On train set
model.eval()
with torch.no_grad():
    true_labels = []
    predictions = []
    for batch in datamodule.train_dataloader():
        x, y = batch
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        true_labels.extend(torch.argmax(y, dim=1).tolist())
        predictions.extend(preds.tolist())
        
true_labels = np.array(true_labels)
predictions = np.array(predictions)
acc_train = accuracy_score(true_labels, predictions)
f1_train = f1_score(true_labels, predictions, average='macro')
cm_train = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=['HL', 'ORCA', 'SF'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Training Data")
plt.savefig("plots/cm_train.png", dpi='figure', bbox_inches='tight')

#On test set
model.eval()
with torch.no_grad():
    true_labels = []
    predictions = []
    for batch in datamodule.test_dataloader():
        x, y = batch
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        true_labels.extend(torch.argmax(y, dim=1).tolist())
        predictions.extend(preds.tolist())
        
true_labels = np.array(true_labels)
predictions = np.array(predictions)
acc_test = accuracy_score(true_labels, predictions)
f1_test = f1_score(true_labels, predictions, average='macro')
cm_test = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['HL', 'ORCA', 'SF'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Test Data")
plt.savefig("plots/cm_test.png", dpi='figure', bbox_inches='tight')

#Saving accuracy and f1-score
metrics = {
    "acc_train": acc_train,
    "f1_train": f1_train,
    "cm_train": cm_train,
    "acc_test": acc_test,
    "f1_test": f1_test,
    "cm_test": cm_test    
}

with open("files/metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)