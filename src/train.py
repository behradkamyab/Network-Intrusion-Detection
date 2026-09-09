import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import IntrusionModel


def train_model(epochs:100, lr = 0.01):
    
    # allocate GPU and CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Training Engine Initialized, accelerator: {device}")
    
    # data pipeline
    data_path = "../data/processed/clean_ddos_traffic.csv"
    train_loader, test_loader, input_dim = get_dataloaders(data_path, batch_size=256)
    
    # model loss optimizer
    model = IntrusionModel(input_dim=input_dim)
    loss = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters() , lr=lr)
    
    # training loop
    print("\n start the training loop...")
    for epoch in range(epochs):
        model.train()
        loss_value = 0.0