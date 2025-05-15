import copy 

from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from utils import get_device

def get_predictions(model, loader, device=None, verbose=False):
    """ returns predictions for provided data loader """
    if device is None:
        device = get_device()

    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in tqdm(loader, disable=not verbose, total=len(loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            y_pred += predicted.tolist()
            y_true += labels.tolist()

    return y_pred, y_true


def evaluate_model(model, loader, metric=accuracy_score, device=None, verbose=False):
    """
    calculates provided metric
    """
    y_pred, y_true = get_predictions(model, loader, device, verbose)
    return metric(y_true, y_pred)


def train(model, train_loader, val_loader, optimizer = None, criterion = None, epochs = 20, score=accuracy_score, patience = None) -> dict:
    """
    Args:
        model: model to train
        train_loader: train data loader
        val_loader: validation data loader
        optimizer: default is Adam
        criterion: default is cross entropy
        epochs: number of epochs, default 20
        score: score function to calculate, should take y_true, y_pred as arguments in that order
        patience: number of epochs with bigger validation loss than the best validation loss before training stops early

    Returns:
        tuple[dict, model]: dict (with keys: train_loss, val_loss, train_score, val_score), best model checkpoint
    """

    # code chunks from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if patience is None:
        patience = epochs

    device = get_device()
    print(f"Device: {device}")
    model.to(device)

    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []

    best_val_loss = float("inf")
    patience_counter = 0

    best_model_state = copy.deepcopy(model)

    for epoch in range(epochs):  # loop over the dataset multiple times

        # training

        model.train()
        train_loss = 0.0
        y_pred = []
        y_true = []

        i = 0
        for inputs, labels in tqdm(train_loader, total=len(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize    
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            y_pred += predicted.tolist()
            y_true += labels.tolist()

            i += 1

        train_score = score(y_true, y_pred)
        train_loss = train_loss / len(train_loader)

        # evalutation

        model.eval()
        val_loss = 0.0
        y_pred = []
        y_true = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                y_pred += predicted.tolist()
                y_true += labels.tolist()

        val_score = score(y_true, y_pred)
        val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train score: {train_score:.2f}, Val score: {val_score:.2f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_scores.append(train_score)
        val_scores.append(val_score)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered")
            break

    results = {'train_loss': train_losses, 'val_loss': val_losses, 'train_score': train_scores, 'val_score': val_scores}
    return results, best_model_state