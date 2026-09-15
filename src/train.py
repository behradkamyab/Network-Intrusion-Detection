import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import get_dataloaders
from model import IntrusionModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score

from projects.Network.src import model


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after certain epochs.
    
    args:
        patience (int): How long to wait after last time validation loss improved.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        path (str): Path for the checkpoint to be saved to.
    """
    def __init__(self, patience=5, delta=0.0001, path="best_model.pth"):
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
            print(f" Watchdog: patience {self.counter}/{self.patience}, best loss: {self.best_loss:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
        

def train_model(epochs=100, lr = 0.01):
    """
    Function to train the IntrusionModel on the network intrusion dataset.
    It uses BCEWithLogitsLoss for binary classification and Adam optimizer.
    Early stopping is implemented to prevent overfitting.
    args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
    returns:
        model (IntrusionModel): The trained model with the best weights.
    """
    
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=2)
    
    # training loop
    print("\n start the training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx , (X_batch , y_batch) in enumerate(train_loader):
            # move the batch into the device (GPU or CPU) and reshape the output
            X_batch = X_batch.to(device)
            y_batch = y_batch.view(-1, 1).to(device)
            
            # forward Propagation
            predictions = model(X_batch)
            losses = loss(predictions, y_batch)
            
            # backpropagation & Optimization
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
        
        scheduler.step(avg_test_loss)
        es(avg_test_loss,model)
        
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")

        if es.early_stop:
            print(f"\n Early stopping triggered at Epoch {epoch+1}")
            break
    # load the best weights back into RAM
    model.load_state_dict(torch.load("../data/processed/best_ddos_model.pth"))
    
    
    print("\n Generating Final Evaluation Metrics...")
    model.eval()
    all_preds = []
    all_targets = []
    
    # run one final pass for metrics calculation
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.view(-1, 1).to(device)
            
            test_preds = model(X_test)
            probabilities = torch.sigmoid(test_preds)
            predicted_classes = (probabilities >= 0.5).float()
            
            # move tensors back to CPU and flatten them for Scikit-Learn
            all_preds.extend(predicted_classes.cpu().numpy().flatten())
            all_targets.extend(y_test.cpu().numpy().flatten())
            
    # calculate Metrics
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f2_score = fbeta_score(all_targets, all_preds, beta=2.0)
    
    print(f"Final Precision: {precision:.4f}")
    print(f"Final Recall:    {recall:.4f}")
    print(f"Final F2-Score:  {f2_score:.4f}")
    
    # generate the Confusion Matrix Heatmap
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Benign (0)", "DDoS (1)"],
                yticklabels=["Benign (0)", "DDoS (1)"])
    plt.title("Final Test Confusion Matrix")
    plt.xlabel("Model Prediction")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig("../data/processed/confusion_matrix.png", dpi=300, bbox_inches='tight')
    print("Confusion Matrix saved to ../data/processed/confusion_matrix.png")

    return model
            
if __name__ == "__main__":
    train_model(epochs=50)