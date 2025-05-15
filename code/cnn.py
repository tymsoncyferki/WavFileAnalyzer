from torch import nn
from config import Config

class CNN(nn.Module):
    """
    Baseline CNN for audio classification
    """

    def __init__(self, dropout: float = 0.5, n_classes: int = None, input_fdim: int = 64, input_tdim: int = 41):
        """
        Args:
            dropout (float): fraction of neurons in flat layer to drop during training
            n_classes (int): number of classes to predict
            input_fdim (int): number of bins for mel-spectrogram
            input_tdim (int): time dimension size
        """
        super().__init__()

        if n_classes is None:
            n_classes = len(Config.MAIN_CLASSES)

        self.conv_layers = nn.Sequential(
            # input: 1 x freqs x times (1 x H x W)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        new_dim = (input_fdim // 8) * (input_tdim // 8)  # divide by stride * stride * stride (2*2*2)

        self.layers = nn.Sequential(   
            nn.Flatten(start_dim=1),
            nn.Linear(128 * new_dim, 512),  # adjust the input size for based on num_bins_mel
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # adds channel dimension
        x = self.conv_layers(x)
        return self.layers(x)
