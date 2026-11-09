"""PyTorch multi-output fully connected neural network, with a sklearn-like API."""

from copy import deepcopy
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class FullyConnectedNeuralNetwork:
    """Train on scaled data; return predictions in the original target units.

    Validation data is supplied to fit() separately so cross-validation callers can
    keep their outer test fold untouched and use only an inner training split for
    early stopping and model selection.
    """

    ACTIVATIONS = {
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
    }

    def __init__(self, hidden_layers=(32, 16), activation='relu',
                 learning_rate=0.001, batch_size=256, weight_decay=0.001,
                 epochs=110, patience=18, min_epochs=35, random_state=4360):
        if activation not in self.ACTIVATIONS:
            raise ValueError(f'Unknown activation function: {activation}')
        if not hidden_layers or any(int(n) < 1 for n in hidden_layers):
            raise ValueError('hidden_layers must be a tuple of positive integers')
        if learning_rate <= 0 or batch_size < 1 or weight_decay < 0:
            raise ValueError('Invalid learning rate, batch size, or weight decay')
        self.hidden_layers = tuple(int(n) for n in hidden_layers)
        self.activation = activation
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.random_state = int(random_state)
        self.history_ = []
        self.model_ = None

    def build_model(self, n_features, n_targets):
        layers = []
        n_in = n_features
        for n_out in self.hidden_layers:
            layers.append(nn.Linear(n_in, n_out))
            layers.append(self.ACTIVATIONS[self.activation]())
            n_in = n_out
        layers.append(nn.Linear(n_in, n_targets))  # Regression: no output activation
        return nn.Sequential(*layers)

    def fit(self, x_train, y_train, x_valid, y_valid):
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        x_valid = np.asarray(x_valid, dtype=np.float64)
        y_valid = np.asarray(y_valid, dtype=np.float64)
        self.x_scaler_ = StandardScaler().fit(x_train)
        self.y_scaler_ = StandardScaler().fit(y_train)
        x_t = torch.from_numpy(self.x_scaler_.transform(x_train).astype('float32'))
        y_t = torch.from_numpy(self.y_scaler_.transform(y_train).astype('float32'))
        x_v = torch.from_numpy(self.x_scaler_.transform(x_valid).astype('float32'))
        y_v = torch.from_numpy(self.y_scaler_.transform(y_valid).astype('float32'))

        torch.manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
            num_workers=0,
        )
        self.model_ = self.build_model(x_train.shape[1], y_train.shape[1])
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        best_loss = float('inf')
        best_weights = None
        best_epoch = 0
        wait = 0
        start = time.perf_counter()
        self.history_ = []
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_sum = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss; lower learning_rate')
                loss.backward()
                optimizer.step()
                train_sum += float(loss.item()) * len(xb)

            self.model_.eval()
            with torch.no_grad():
                valid_loss = float(criterion(self.model_(x_v), y_v).item())
            if not np.isfinite(valid_loss):
                raise FloatingPointError('Nonfinite validation loss')
            train_loss = train_sum / len(x_train)
            self.history_.append({
                'epoch': epoch, 'loss_train': train_loss,
                'loss_valid': valid_loss,
            })
            # Small tolerance avoids resetting patience for meaningless jitter.
            if valid_loss < best_loss - 1e-6:
                best_loss = valid_loss
                best_epoch = epoch
                best_weights = deepcopy(self.model_.state_dict())
                wait = 0
            else:
                wait += 1
            if epoch >= self.min_epochs and wait >= self.patience:
                break

        self.model_.load_state_dict(best_weights)
        self.model_.eval()
        self.best_epoch_ = best_epoch
        self.epochs_run_ = epoch
        self.converged_ = (epoch < self.epochs and best_epoch <= epoch - self.patience)
        self.best_valid_loss_ = best_loss
        self.elapsed_seconds_ = time.perf_counter() - start
        return self

    def predict(self, x):
        if self.model_ is None:
            raise RuntimeError('The neural network has not yet been fitted')
        x_scaled = self.x_scaler_.transform(np.asarray(x, dtype=np.float64))
        x_tensor = torch.from_numpy(x_scaled.astype('float32'))
        self.model_.eval()
        with torch.no_grad():
            prediction = self.model_(x_tensor).numpy()
        return self.y_scaler_.inverse_transform(prediction)
