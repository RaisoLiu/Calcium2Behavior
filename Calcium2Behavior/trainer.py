# trainer.py
import torch
import torch.nn as nn
from .model import GRUModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
# from .focal_mse_loss import FocalMSELoss


def train_model(train_loader, data_specs, config, device):
    model = GRUModel(
        input_dim=data_specs['input_dim'],
        output_dim=data_specs['output_dim'],
        hidden_dim=config['training']['hidden_dim'],
        window_size=config['data']['left_window_size'] + config['data']['right_window_size'] + 1,
    ).to(device)

    criterion = torch.nn.MSELoss() if config['training']['task_type'] == 'regression' else torch.nn.CrossEntropyLoss()
    # criterion = FocalMSELoss(alpha=5.0, reduction='mean') if config['training']['task_type'] == 'regression' else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    epochs = config['training']['total_epochs']

    model.train()
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    return model, train_losses


def test_model(test_loader, model, data_specs, device, config):
    model.eval()

    predictions, ground_truth = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            predictions.append(preds.cpu().numpy())
            ground_truth.append(y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    if config['training']['task_type'] == 'regression':
        metric = r2_score(ground_truth, predictions)
    else:
        predictions = np.argmax(predictions, axis=1)
        metric = accuracy_score(ground_truth, predictions)

    return predictions, ground_truth, metric