import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NetworkDataset(Dataset):
    """
    A custom PyTorch Dataset class for network intrusion detection.
    
    """
    def __init__(self, features , labels):
        # features and labels are numpy arrays for now
        self.features = features
        self.labels = labels
        super().__init__()
        
    def __len__(self):
        # return the total number of rows
        return len(self.features)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y , dtype=torch.float32)
        
        return x_tensor , y_tensor
    
    
    
def get_dataloaders(data_path: str , batch_size : int = 256 , test_size: float = 0.2):
    """
    Function to create PyTorch DataLoaders for training and testing datasets.
    
    args:
        data_path (str): Path to the cleaned CSV dataset.
        batch_size (int): Number of samples per batch.
        test_size (float): Proportion of the dataset to include in the test split.
    """
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=test_size , random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_dataset = NetworkDataset(X_train_scaled,y_train)
    test_dataset = NetworkDataset(X_test_scaled,y_test)
    
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    
    input_dim = X_train.shape[1]
    return train_dataloader, test_dataloader , input_dim
    
if __name__ == "__main__":
    train_dl, test_dl, dim = get_dataloaders("../data/processed/clean_ddos_traffic.csv")
    
    X_batch, y_batch = next(iter(train_dl))
    print(f"\nBatch X shape: {X_batch.shape} (Batch Size x Features)")
    print(f"Batch y shape: {y_batch.shape} (Batch Size,)")
