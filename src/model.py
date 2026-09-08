import torch
from torch import nn

class IntrusionModel(nn.Module):
    def __init__(self,input_dim: int):
        super().__init__()
        
        
        self.network = nn.Sequential(
            # layer 1 (input)
            nn.Linear(input_dim,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # we can later add dropout or L2 regularization if we see overfitting
            
            # layer 2
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # output layer
            nn.Linear(64,1)
            # no need for any activation function we need raw logits to use sigmoid later!            
        )
        
    def forward(self,X: torch.tensor) -> torch.tensor:
        return self.network(X)
        