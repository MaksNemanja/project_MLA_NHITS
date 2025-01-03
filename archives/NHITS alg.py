import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from math import ceil


### Model Components ###
class Block(nn.Module):
    def __init__(self, input_size: int, horizon: int, hidden_sizes: list, kernel_size: int, expressiveness_ratio: float):
        super(Block, self).__init__()
        self.L = input_size
        self.H = horizon
        self.k_l = kernel_size
        self.r_l = expressiveness_ratio
        self.theta_size = int(ceil(self.r_l * self.H))

        # MaxPool layer for multi-rate signal sampling
        self.maxpool = nn.MaxPool1d(kernel_size=self.k_l)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size // self.k_l, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        for hidden_size in hidden_sizes[1:]:
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())

        self.linear_f = nn.Linear(hidden_sizes[-1], self.theta_size)
        self.linear_b = nn.Linear(hidden_sizes[-1], self.theta_size)

    def forward(self, x: torch.Tensor) -> tuple:
        x_pooled = self.maxpool(x)
        h = x_pooled
        for layer in self.layers:
            h = layer(h)

        theta_f = self.linear_f(h).unsqueeze(1)
        theta_b = self.linear_b(h).unsqueeze(1)
        forecast = F.interpolate(theta_f, size=self.H, mode="linear").squeeze(1)
        backcast = F.interpolate(theta_b, size=self.L, mode="linear").squeeze(1)
        return forecast, backcast


class Stack(nn.Module):
    def __init__(self, nb_block: int, input_size: int, horizon: int, hidden_sizes: list, kernel_size: int, expressiveness_ratio: float):
        super(Stack, self).__init__()
        self.blocks = nn.ModuleList([Block(input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio) for _ in range(nb_block)])

    def forward(self, x: torch.Tensor) -> tuple:
        residual = x
        stack_forecast = 0
        for block in self.blocks:
            forecast, backcast = block(residual)
            residual -= backcast
            stack_forecast += forecast
        return stack_forecast, residual


class NHITS(nn.Module):
    def __init__(self, nb_stack: int, nb_block: int, input_size: int, horizon: int, hidden_sizes: list, kernel_size: int, expressiveness_ratio: float):
        super(NHITS, self).__init__()
        self.stacks = nn.ModuleList([Stack(nb_block, input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio) for _ in range(nb_stack)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        global_forecast = 0
        for stack in self.stacks:
            stack_forecast, residual = stack(residual)
            global_forecast += stack_forecast
        return global_forecast


### Data Loading and Preprocessing ###
def load_data(file_path: str) -> torch.Tensor:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file does not exist at the specified path: {file_path}")

    try:
        data = pd.read_csv(file_path)  # Assuming CSV file format
        print(f"File loaded successfully: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading file: {file_path}. Ensure it is a valid CSV file.\n{e}")

    
    data = data.drop(columns=["date"], errors="ignore")  # Ignore error if 'date' column doesn't exist
    # Convert all columns to numeric, forcing errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Handle missing values: Drop rows with NaN values or fill with a specific value
    data = data.dropna()  # Optionally, use fillna() instead of dropna()

    # Verify the cleaned data
    print(data.dtypes)  # Ensure all columns are now numeric
    print(data.info())
    # Convert the cleaned DataFrame to a PyTorch tensor
    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    return data_tensor


def preprocess_data(data_tensor: torch.Tensor, input_size: int, horizon: int):
    print(f"Data tensor shape before preprocessing: {data_tensor.shape}")
    X, Y = [], []
    for i in range(len(data_tensor) - input_size - horizon):
        X.append(data_tensor[i:i + input_size])
        Y.append(data_tensor[i + input_size:i + input_size + horizon])
    return torch.stack(X), torch.stack(Y)


def create_dataloader(X: torch.Tensor, Y: torch.Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


### Training and Evaluation ###
def train_model(model, dataloader, epochs: int, learning_rate: float):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def evaluate_model(model, dataloader) -> float:
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


### Main ###
if __name__ == "__main__":
    # File path to your dataset
    file_path = "ECL.csv"  # Update with the path to your CSV file

    # Load and preprocess data
    data_tensor = load_data(file_path)
    print(f"Data tensor shape: {data_tensor.shape}")
    input_size, horizon = 32, 10  # Define input sequence length and forecast horizon
    X, Y = preprocess_data(data_tensor, input_size, horizon)

    # Create DataLoader
    dataloader = create_dataloader(X, Y, batch_size=32)

    # Initialize NHITS model
    model = NHITS(nb_stack=3, nb_block=2, input_size=input_size, horizon=horizon,
                  hidden_sizes=[64, 32], kernel_size=2, expressiveness_ratio=0.5)

    # Train the model
    train_model(model, dataloader, epochs=10, learning_rate=0.001)

    # Evaluate the model
    eval_loss = evaluate_model(model, dataloader)
    print(f"Evaluation Loss: {eval_loss:.4f}")
