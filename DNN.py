'''
**Author**: Benjamin Urben<br>
**Email**: burben@student.ethz.ch / benjamin.urben@hotmail.ch<br>
**Context**: Master Thesis on "Use of Machine Learning in the Design and Analysis for Steel Connections"<br>
**Institution**: ETH Zürich, Institute of Structural Engineering (IBK)
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.model = None
        self.num_features = None
        self.num_targets = None

        self.scaler_x = StandardScaler()
        self.criterion = nn.MSELoss()

        # Configuration of model and architecture parameters
        self.config = {
            "batch_size": 256,
            "epochs": 100,
            "learning_rate": 0.001,
            "test_size": 0.1,
            "val_size": 0.1, 
            "print_logs": True,
            "dropout_prob": 0.1, 

            "num_layers": 3,
            "neurons_per_layer": [64, 32, 16],
            "use_batch_norm": False,  # Batch normalization

            "seed": 42             # Seeding all random processes
            }
        
        # Information about the 
        self.data = {"data_reduction_factor": 0,
                     "data_file_name":None,
                     "data_indeces":None, # Save indeces and not full dataframe to save memory
                     "test_data":None, 
                     "val_data":None,
                     "train_data":None}

        # Training features
        self.features = ["A_x", "Iy_x", "Wely_x", "Wply_x", "fy_x", 
                        "A_y", "Iy_y", "Wely_y", "Wply_y", "fy_y", 
                        "Gamma", "Offset",
                        "h_wid", "b_wid", "d_wid", "t_fwid", "t_wwid", 
                        "t_stiffc",
                        "V_contribution", "M_contribution"] 
        
        # Output target
        self.target = ["target"]

        # Losses
        self.train_losses = {'total': []}
        self.val_losses = {'total': []}  
    
    def forward(self, x, apply_dropout=False):
        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d):
                # Skip BatchNorm if batch size is 1
                if x.size(0) == 1:
                    continue
            x = layer(x)
            if isinstance(layer, nn.Linear) and (self.training or apply_dropout): #old isinstance(layer, nn.ReLu)
                x = self.dropout(x)  
        return x

    def prepare_data(self, df):
        self.features = [feature for feature in self.features if feature in df.columns]
        self.x_data = df[self.features].to_numpy().astype(np.float32)
        self.y_data = df[self.target].to_numpy().astype(np.float32).reshape(-1, 1)

        x_train, x_temp, y_train, y_temp, train_idx, temp_idx = train_test_split(
            self.x_data, self.y_data, range(len(self.x_data)), 
            test_size=self.config['test_size'] + self.config['val_size'], 
            random_state=self.config['seed']
        )

        x_val, x_test, y_val, y_test, temp_val_idx, temp_test_idx = train_test_split(
            x_temp, y_temp, range(len(x_temp)), 
            test_size=self.config['test_size'] / (self.config['test_size'] + self.config['val_size']),  
            random_state=self.config['seed']
        )

        val_idx = [temp_idx[i] for i in temp_val_idx]
        test_idx = [temp_idx[i] for i in temp_test_idx]

        # Fit scaler only on training data
        self.scaler_x.fit(x_train)
        x_train_std = self.scaler_x.transform(x_train)
        x_val_std = self.scaler_x.transform(x_val)
        x_test_std = self.scaler_x.transform(x_test)

        self.x_train, self.x_val, self.x_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [x_train_std, x_val_std, x_test_std])
        self.y_train, self.y_val, self.y_test = map(lambda y: torch.tensor(y, dtype=torch.float32), [y_train, y_val, y_test])

        self.num_features = self.x_train.shape[1]
        self.num_targets = self.y_train.shape[1]

        self.data['train_data'] = train_idx
        self.data['val_data'] = val_idx
        self.data['test_data'] = test_idx

        self.data['data_indeces'] = df.index.tolist()

        if self.config['print_logs']:
            print("Number of training samples:", self.x_train.shape[0])
            print("Number of validation samples:", self.x_val.shape[0])
            print("Number of testing samples:", self.x_test.shape[0])


    def add_training_data(self, df_new):
        x_new_data = df_new[self.features].to_numpy().astype(np.float32)
        y_new_data = df_new[self.target].to_numpy().astype(np.float32)

        self.x_data = np.concatenate((self.x_data, x_new_data), axis=0)
        self.y_data = np.concatenate((self.y_data, y_new_data), axis=0)

        # Does the scaler need to be rescaled?
        x_train_std = np.concatenate((self.x_train.numpy(), self.scaler_x.transform(x_new_data)), axis=0)
        y_train = np.concatenate((self.y_train.numpy(), y_new_data), axis=0)

        # Update training data
        self.x_train = torch.tensor(x_train_std, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

        self.data['data_indeces'].extend(df_new.index.tolist())
        self.data['train_data'].extend(df_new.index.tolist())

    def build(self):
        if self.num_features is None:
            print('Prepare data first')
            return None
        
        num_layers = self.config.get("num_layers", 3)
        neurons_per_layer = self.config.get("neurons_per_layer", [64] * num_layers)
        use_batch_norm = self.config.get("use_batch_norm", False)

        layers = []
        input_size = self.num_features
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, neurons_per_layer[i]))
            # Add Batch normalization layer
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            # Add activation function
            layers.append(nn.ReLU())

            input_size = neurons_per_layer[i]

        layers.append(nn.Linear(input_size, self.num_targets))
        
        self.model = nn.Sequential(*layers)
        self.dropout = nn.Dropout(self.config.get("dropout_prob", 0.0))

    def train(self):
        if self.model is None:
            print('Build model first')
            return None
        
        # Seed all random processes
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        torch.cuda.manual_seed_all(self.config['seed'])

        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        train_dataset = TensorDataset(self.x_train, self.y_train)
        val_dataset = TensorDataset(self.x_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()  # Set model to training mode
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses['total'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()  # Set model to evaluation mode
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.val_losses['total'].append(avg_val_loss)
            
            if self.config['print_logs']:
                print(f"Epoch {epoch+1}/{self.config['epochs']}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    def fine_tune(self, df_new=None, epochs=None, learning_rate=None):
        """
        Fine-tune the model using a new set of training data.
        
        Parameters:
        - df_new (DataFrame): New training data.
        - epochs (int): Number of epochs to fine-tune. If None, use the default number from config.
        - learning_rate (float): Learning rate for fine-tuning. If None, use the default learning rate from config.
        """
        # Set fine-tuning parameters
        if epochs is None:
            epochs = self.config['epochs']
        if learning_rate is None:
            learning_rate = self.config['learning_rate']

        if df_new is not None:
            # Add new training data (this will update self.x_train and self.y_train)
            self.add_training_data(df_new)

        # Define optimizer (you can modify this to fine-tune specific layers)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Prepare the data loaders with the updated data
        train_dataset = TensorDataset(self.x_train, self.y_train)
        val_dataset = TensorDataset(self.x_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)

        # Fine-tuning phase
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses['total'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.val_losses['total'].append(avg_val_loss)
            
            if self.config['print_logs']:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    
    def evaluate(self):
        # Seed torch
        torch.manual_seed(self.config['seed'])
        
        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(self.x_test)
            test_loss = self.criterion(y_pred, self.y_test).item()
        
        return y_pred, test_loss
    
    def get_metrics(self,y_pred=None,y_true=None,dropout_prob=0.1):
        if y_true is None:
            y_true = self.y_test.numpy()

        if y_pred is None:
            y_pred = self.model(self.x_test).detach().numpy()

        mse = np.mean((y_pred - y_true)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_true))
        r2 = 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2)

        y_true = self.y_test.numpy()

        mcd_mean, mcd_std, _ = self.mc_dropout_uncertainty(dropout_prob=dropout_prob)

        MCDMS = mean_squared_error(np.zeros(len(mcd_std)), mcd_mean / y_true - 1)
        MCDUS = mean_squared_error(np.zeros(len(mcd_std)), mcd_std)

        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MCDMS': MCDMS, 'MCDUS': MCDUS}
    
    def predict(self, df):
        # See torch
        torch.manual_seed(self.config['seed'])

        # Set model to evaluation mode
        self.model.eval()

        x = df[self.features].to_numpy().astype(np.float32)
        y = df[self.target].to_numpy().astype(np.float32)
        x_std = self.scaler_x.transform(x)
        x_processed = torch.tensor(x_std, dtype=torch.float32)
        y_processed = torch.tensor(y, dtype=torch.float32)

        with torch.no_grad():
            y_pred = self.model(x_processed)
            test_loss = self.criterion(y_pred, y_processed).item()
        
        return y_pred, test_loss
    
    def mc_dropout_uncertainty(self, df_input = None, num_samples=100, dropout_prob=0.1):
        # Seed torch
        torch.manual_seed(self.config['seed'])

        if df_input is None:
            x_input = self.x_test
        else:
            x_input = df_input[self.features].to_numpy().astype(np.float32)
            x_input = self.scaler_x.transform(x_input)
            x_input = torch.tensor(x_input, dtype=torch.float32)
        
        # If for training no dropout was used, set it to the default value
        if self.config['dropout_prob'] == 0.0:
            self.dropout.p = dropout_prob

        with torch.no_grad():
            # Important: For MC Dropout torch has to be reseeded, else the deactivated neurons will be the same for each forward pass
            torch.manual_seed(torch.initial_seed() + torch.randint(0, 1000000, (1,)).item())  

            y_preds = torch.stack([self.forward(x_input, apply_dropout=True) for _ in range(num_samples)])

        # Reset dropout to the original value
        if self.config['dropout_prob'] == 0.0:
            self.dropout.p = 0.0

        # ATTENTION: PyTorch calculates the standard deviation using Bessel's correction (N-1), which is not the case in numpy.
        mean_pred = y_preds.mean(dim=0)
        std_pred = y_preds.std(dim=0, unbiased=True)

        return mean_pred.numpy(), std_pred.numpy(), y_preds.numpy()
    
    def get_MV_interaction_prediction(self,x_profile,n_predictions=40):
        # Set model to evaluation mode
        self.model.eval()

        x_profile_features = x_profile[self.features].to_numpy().astype(np.float32)
        x_profile_features = x_profile_features.reshape(1,-1)
        
        M_contribution = np.linspace(1, 0, n_predictions)
        V_contribution = np.linspace(0, 1, n_predictions)

        V_cont = V_contribution / np.sqrt(M_contribution**2 + V_contribution**2)
        M_cont = M_contribution / np.sqrt(M_contribution**2 + V_contribution**2)

        targets_predicted = np.zeros(n_predictions)

        V_cont_index = self.features.index('V_contribution')
        M_cont_index = self.features.index('M_contribution')

        for i in range(n_predictions):
            x_profile_features[0, V_cont_index] = V_cont[i]
            x_profile_features[0, M_cont_index] = M_cont[i]
            x_profile_std = self.scaler_x.transform(x_profile_features)
            x_profile_std = torch.tensor(x_profile_std, dtype=torch.float32)
            with torch.no_grad():
                targets_predicted[i] = self.model(x_profile_std).cpu().numpy()[0]

        Mpl_y = x_profile['Mpl_y'] 
        Vpl_y = x_profile['Vpl_y']

        M_Rd_pred = np.zeros(len(M_cont))
        V_Rd_pred = np.zeros(len(V_cont))

        for i in range(len(M_cont)):
            M_Rd_pred[i] = targets_predicted[i] * Mpl_y * M_cont[i]
            V_Rd_pred[i] = targets_predicted[i] * Vpl_y * V_cont[i]

        return M_Rd_pred, V_Rd_pred, targets_predicted
    
    def plot_training_loss(self):
        fig, ax = plt.subplots(figsize=(6,4))
        plt.plot(self.train_losses['total'], label='Training Loss')
        plt.plot(self.val_losses['total'], label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.grid(True, which='major', color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.xlim([0,None])
        plt.yscale('log')
        plt.legend()

        plt.show()

    def plot_test_predictions(self):
        y_pred, test_loss = self.evaluate()

        fig, ax = plt.subplots(figsize=(6,4))

        plt.scatter(self.y_test.numpy(), y_pred.numpy(), s=15, color="green", edgecolors="black")
        plt.grid(True, which='major', color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.ylabel("Target Predicted")
        plt.xlabel("Target Ground Truth")
        plt.show()

    @staticmethod
    def seed_everything(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save(self, path):
        torch.save(self, path)

        print("Model saved to: {}".format(path))
