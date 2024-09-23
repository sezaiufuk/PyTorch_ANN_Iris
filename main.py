import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./Iris.csv')
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
IDs = df_shuffled["Id"]
df_shuffled = df_shuffled.drop(columns=["Id"])

label_encoder = LabelEncoder()
X = df_shuffled.iloc[:, 1:-1]
y = label_encoder.fit_transform(df_shuffled.loc[:, "Species"])

train_size = 0.7
dev_size = 0.15
test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, shuffle=False, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=test_size/(test_size + dev_size), shuffle=False, random_state=42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))
X_dev = pd.DataFrame(scaler.transform(X_dev))

class ANN(nn.Module):
    def __init__(self, loss, optimizer):
        super().__init__()
        torch.manual_seed(42)
        self.losses = {
            "train":[],
            "dev":[]
        }
        self.loss_fn = loss
        self.optimizer = optimizer
        self.model_ref = None

    def forward(self, model, X):
        return model(X)

    def fit(self, X: pd.DataFrame, y: np.ndarray, epoch: int, batch_size: int=X.shape[0]):
        X_tensor = torch.Tensor(X.values)
        y_tensor = torch.LongTensor(y)
        tensor_dataset = TensorDataset(X_tensor, y_tensor)

        input_size = X.shape[1]
        class_size = len(np.unique(y))

        model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, class_size)
        )
        loss_fn = self.loss_fn()
        optimizer = self.optimizer(model.parameters(), lr=0.001, weight_decay=0.001)
        self.model_ref = model

        for i in range(epoch):
            model.train()
            dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                preds = self.forward(model, batch_X)
                loss = loss_fn(preds, batch_y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                
            self.losses["train"].append(total_loss / len(dataloader))
            
            model.eval()
            with torch.no_grad():
                dev_preds = self.forward(model, torch.Tensor(X_dev.values))
                dev_loss = loss_fn(dev_preds, torch.LongTensor(y_dev))
                self.losses["dev"].append(dev_loss.item())

    def predict(self, X):
        model_ref = self.model_ref
        model_ref.eval()
        X_tensor = torch.Tensor(X.values)
        with torch.no_grad():
            preds = self.forward(model_ref, X_tensor)
            return torch.argmax(preds, dim=1)

model = ANN(nn.CrossEntropyLoss, optim.Adam)
model.fit(X_train, y_train, epoch=850, batch_size=32)

y_pred = model.predict(X_test)

plt.plot(model.losses["train"], label=f"Train Loss: {round(model.losses['train'][-1],2)}")
plt.plot(model.losses["dev"], label=f"Dev Loss: {round(model.losses['dev'][-1],2)}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Mini-Batch Training and Dev Loss Over Epochs\nTest Accuracy: {accuracy_score(y_pred, y_test)}')
plt.legend()
plt.show()