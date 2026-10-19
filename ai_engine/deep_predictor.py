# ai_engine/deep_predictor.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockLSTMForecaster(nn.Module):
  """Neural network for predicting next-bar OHLCV based on historical sequences

  and technical indicators.
  """

  def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=5):
    super(StockLSTMForecaster, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    self.lstm = nn.LSTM(
        input_dim,
        hidden_dim,
        num_layers,
        batch_first=True,
        dropout=0.2 if num_layers > 1 else 0.0,
    )
    self.fc = nn.Sequential(
        nn.Linear(hidden_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),  # Open, High, Low, Close, Volume
    )

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out


class DeepPredictorEngine:

  def __init__(
      self,
      model_path="ai_engine/model_checkpoint.pt",
      input_dim=15,
      learning_rate=0.001,
  ):
    self.model_path = model_path
    self.input_dim = input_dim
    self.model = StockLSTMForecaster(input_dim=input_dim).to(DEVICE)
    self.criterion = nn.MSELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    if os.path.exists(self.model_path):
      try:
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=DEVICE)
        )
        print("Loaded AI model checkpoint from ai_engine/ successfully.")
      except Exception as e:
        print(f"Could not load checkpoint, starting fresh: {e}")

  def prepare_sequences(self, df, seq_length=30):
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "EMA_9",
        "EMA_50",
        "MACD_Line",
        "MACD_Signal",
        "MACD_Hist",
        "ADX",
        "RSI_14",
        "TSI",
        "VWAP",
        "ATR",
    ]
    available_cols = [col for col in feature_cols if col in df.columns]
    data = df[available_cols].values

    if len(data) <= seq_length + 1:
      return None, None

    X, y = [], []
    for i in range(seq_length, len(data) - 1):
      X.append(data[i - seq_length : i])
      y.append(data[i + 1, :5])

    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(
        np.array(y), dtype=torch.float32
    )

  def learn_from_reality(self, df, epochs=3, seq_length=30):
    """Continual Learning / Online Feedback Loop:

    Updates weights using prediction error against actual market outcomes.
    """
    self.model.train()
    X, y = self.prepare_sequences(df, seq_length=seq_length)

    if X is None or len(X) == 0:
      print("Not enough data points for neural network training loop.")
      return

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )

    print(f"Refining AI weights across {epochs} epochs on {DEVICE}...")

    for epoch in range(epochs):
      epoch_loss = 0.0
      for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

        self.optimizer.zero_grad()
        outputs = self.model(batch_X)
        loss = self.criterion(outputs, batch_y)
        loss.backward()
        self.optimizer.step()
        epoch_loss += loss.item()

      print(
          f"Epoch [{epoch+1}/{epochs}] | MSE Error:"
          f" {epoch_loss/len(loader):.6f}"
      )

    torch.save(self.model.state_dict(), self.model_path)
    print(f"Model checkpoint saved to '{self.model_path}'.")