import os
import random

import torch
import numpy as np
from scipy.io import wavfile

def get_device():
    """ gets torch device """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed):
    """ sets seed for torch, numpy and python """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_model(model, filepath):
    """ saves the model to the specified filepath """
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    """ loads the model from the specified filepath """
    model.load_state_dict(torch.load(filepath))
    return model

def calculate_accuracy(y_true, y_pred):
    """ calculates the accuracy of predictions """
    return (y_true == y_pred).mean()
