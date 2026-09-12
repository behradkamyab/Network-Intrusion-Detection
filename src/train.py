import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import get_dataloaders
from model import IntrusionModel


class EarlyStopping():
    def __init__(self, patience=5, delta=0.01, path="best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss , model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(),self.path)
        else:
            self.counter +=1
            print(f" Watchdog: patience {int(self.counter/self.patience)}, best loss: {self.best_loss:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
        

def train_model(epochs=100, lr = 0.01):
    
    # allocate GPU and CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Training Engine Initialized, accelerator: {device}")
    
    # data pipeline
    data_path = "../data/processed/clean_ddos_traffic.csv"
    train_loader, test_loader, input_dim = get_dataloaders(data_path, batch_size=256)
    
    # model loss optimizer
    model = IntrusionModel(input_dim=input_dim).to(device)
    loss = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters() , lr=lr)
    es = EarlyStopping(7,path="../data/processed/best_ddos_model.pth")
    
    # training loop
    print("\n start the training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx , (X_batch , y_batch) in enumerate(train_loader):
            # move the batch into the device (GPU or CPU) and reshape the output
            X_batch = X_batch.to(device)
            y_batch = y_batch.view(-1, 1).to(device)
            
            # Forward Propagation
            predictions = model(X_batch)
            losses = loss(predictions, y_batch)
            
            # Backpropagation & Optimization
            optim.zero_grad() 
            losses.backward()       
            optim.step()      
            
            running_loss += losses.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # eval phase
        model.eval() 
        test_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad(): 
            for X_test, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.view(-1, 1).to(device)
                
                test_preds = model(X_test)
                batch_loss = loss(test_preds, y_test)
                test_loss += batch_loss.item()
                
                probabilities = torch.sigmoid(test_preds)
                predicted_classes = (probabilities >= 0.5).float()
                
                correct_preds += (predicted_classes == y_test).sum().item()
                total_preds += y_test.size(0)
                
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = (correct_preds / total_preds) * 100
        
        es(avg_test_loss,model)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")

        if es.early_stop:
            print(f"\n Early stopping triggered at Epoch {epoch+1}")
            break
    # Load the best weights back into RAM
    model.load_state_dict(torch.load("../data/processed/best_ddos_model , but.pth"))
    return model
            
if __name__ == "__main__":
    train_model(epochs=50)