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
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import copy
import pandas as pd


class Encoder(nn.Module):
    def __init__(self, input_dim, conditioning_dim, latent_dim, hidden_dims):
        super(Encoder, self).__init__()
            
        modules = []
        input_size = input_dim + conditioning_dim
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_size, h_dim),
                    #nn.BatchNorm1d(h_dim), CORRECTED: Can cause instability in encoder
                    nn.ReLU(),
                    #nn.LeakyReLU(0.2),
                    #nn.Dropout(0.2)
                )
            )
            input_size = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.z_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.z_log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        self.apply(self._init_weights)
                
    def _init_weights(self, m):
        '''
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        '''

        if isinstance(m, nn.Linear):
            # Kaiming/He initialization (good for ReLU)
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)
        
    def forward(self, x, c):
        x_combined = torch.cat((x, c), dim=1)
        
        x_encoded = self.encoder(x_combined)
        
        z_mean = self.z_mean(x_encoded)
        z_log_var = self.z_log_var(x_encoded)

        # Prevent numerical instability
        z_log_var = torch.clamp(z_log_var, min=-10, max=10)
        
        return z_mean, z_log_var
    
class Decoder(nn.Module):
    def __init__(self, input_dim, conditioning_dim, latent_dim, hidden_dims):
        super(Decoder, self).__init__()
            
        modules = []
        #input_size = latent_dim + conditioning_dim
        input_size = latent_dim  # CORRECTED: Start with latent_dim only, conditioning_dim added in loop
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_size + conditioning_dim, h_dim), # CORRECTED: Change to include conditioning_dim
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    #nn.LeakyReLU(0.2),
                    #nn.Dropout(0.1) 
                )
            )
            input_size = h_dim
            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, z, c):
        #z_conditioned = torch.cat((z, c), dim=1)

        # CORRECTED: Use a loop to condition z with c at each layer
        z_conditioned = z
        for layer in self.decoder:
            z_conditioned = layer(torch.cat([z_conditioned, c], dim=1))
        x_decoded = z_conditioned
        
        #x_decoded = self.decoder(z_conditioned)
        
        return self.final_layer(x_decoded)

class CVAE(nn.Module):
    def __init__(self, features, conditions):
        super(CVAE, self).__init__()
        self.encoder = None
        self.decoder = None

        self.input_dim = None
        self.conditioning_dim = None

        self.features = features

        self.conditions = conditions
        
        self.config = {
            "latent_dim": 8,
            "batch_size": 64,
            "epochs": 100,
            "learning_rate": 0.001,
            "hidden_dims_encoder": [32, 16],
            "hidden_dims_decoder": [16, 32, 64],
            "seed": 42 ,

            "print_logs": True,
            "loss_convergence_stopping":None,      # None = disabled, otherwise number of epochs to wait for loss convergence
            "posterior_collapse_stopping": False,  # False = disabled, otherwise stops training if KL divergence is too low
            "static_stopping":None,                # None = disabled, otherwise number of epochs to wait for static loss

            "beta":1.0,                            # Initial beta value for KL divergence loss scaling
            "beta_strategy": 'constant',           # Options: annealing, constant, loss_balance

            # Annealing parameters
            "annealing_final_beta": None,          # Final beta value for annealing strategy
            "annealing_start_at": 0,               # Epoch to start annealing
            "annealing_steps": None,               # Number of epochs to reach final beta value
            # Loss balance parameters
            "loss_balance_target": None,           # Target ratio for KL divergence to reconstruction loss
            "loss_balance_lr": None,               # Learning multiplier for loss balance strategy (loss_balance_lr * ratio, with loss_balance_lr < 1.0)

            "L2_factor": 0,                        # L2 regularization factor

            'test_size':0.1,
            'val_size':0.1,

            "freeze_copy_every_n_epochs": None,    # None = disabled, otherwise number of epochs to create a frozen copy of the model
            "frozen_copies":{},
            
            # Add device configuration
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

        self.data = {"data_reduction_factor": 0,
                     "data_file_name":None,
                     "data_indeces":None,
                     "test_data":None, 
                     "val_data":None,
                     "train_data":None}
        
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.train_losses = {
            'total': [],
            'reconstruction': [],
            'kl': [],
            'L2': []
        }
        self.val_losses = {
            'total': [],
            'reconstruction': [],
            'kl': [],
            'beta': [],
            'L2': []
        }

    '''
    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
    
    def forward(self, x, c):
        z_mean, z_log_var = self.encoder(x, c)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decoder(z, c), z_mean, z_log_var
        '''

    def reparameterize(self, mu, logvar): 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y): 
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, y)
        return x_hat, mu, logvar
    
    def build(self):
        if self.input_dim is None:
            print("Prepare data first")
            return None
        
        self.encoder = Encoder(
            self.input_dim, 
            self.conditioning_dim, 
            self.config["latent_dim"],
            self.config['hidden_dims_encoder']
        )
        
        self.decoder = Decoder(
            self.input_dim, 
            self.conditioning_dim, 
            self.config["latent_dim"],
            self.config['hidden_dims_decoder']
        )
        
        # Move models to GPU
        self.encoder.to(self.config["device"])
        self.decoder.to(self.config["device"])
        self.to(self.config["device"])
        
        return self.encoder, self.decoder

    def train_model(self):
        if self.encoder is None or self.decoder is None:
            print("Error: Build the model first")
            return None
        
        # ************************** Prepare data & model **************************

        # Seed all random processes
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        torch.cuda.manual_seed_all(self.config['seed'])

        # CORRECTED: Move data to CPU for DataLoader to avoid GPU memory accumulation
        x_train_cpu = self.x_train.cpu()
        y_train_cpu = self.y_train.cpu()
        x_val_cpu = self.x_val.cpu()
        y_val_cpu = self.y_val.cpu()

        # Create datasets and dataloaders
        train_dataset = TensorDataset(x_train_cpu, y_train_cpu)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        
        val_dataset = TensorDataset(x_val_cpu, y_val_cpu)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=False)

        # Define optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.config["learning_rate"], weight_decay=1e-5)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        beta = self.config['beta']
        
        # Display which device is being used
        print(f"Training on: {self.config['device']}")
        
        # ************************** Start training **************************
        for epoch in range(self.config['epochs']):
            self.train()  

            # CORRECTED: Disable frozen copies feature to prevent memory accumulation
            # This was storing entire model copies in memory which grows over time
            if self.config["freeze_copy_every_n_epochs"] is not None:
                print("Warning: Frozen copies feature disabled to prevent memory issues")
                # if (epoch) % self.config["freeze_copy_every_n_epochs"] == 0:
                #     # Make a CPU copy for freezing to save GPU memory
                #     cpu_copy = copy.deepcopy(self.to("cpu"))
                #     self.config["frozen_copies"][epoch] = cpu_copy
                #     # Move back to GPU for continued training
                #     self.to(self.config["device"])

            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            
            # ************************** TRAINING **************************
            train_total_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0
            train_L2_loss = 0
            
            for x_batch, y_batch in train_dataloader:
                # Move batch to GPU
                x_batch = x_batch.to(self.config["device"])
                y_batch = y_batch.to(self.config["device"])
                
                optimizer.zero_grad()
                
                batch_total_loss, batch_recon_loss, batch_kl_loss, L2_loss = self.calculate_loss(x_batch, y_batch, beta)
                
                # Prevent numerical instability
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Clip gradients

                batch_total_loss.backward()
                optimizer.step()
                
                # CORRECTED: Use .item() to extract scalar values and avoid tensor accumulation
                train_total_loss += batch_total_loss.item()
                train_recon_loss += batch_recon_loss.item()
                train_kl_loss += batch_kl_loss.item()
                train_L2_loss += L2_loss.item()
                
                # CORRECTED: Explicitly delete tensors to free GPU memory
                del x_batch, y_batch, batch_total_loss, batch_recon_loss, batch_kl_loss, L2_loss
                torch.cuda.empty_cache()  # Clear GPU cache
            
            # Calculate average losses
            train_total_loss /= len(train_dataloader)
            train_recon_loss /= len(train_dataloader)
            train_kl_loss /= len(train_dataloader)
            train_L2_loss /= len(train_dataloader)
            
            # CORRECTED: Store only scalar values, not tensors
            self.train_losses['total'].append(train_total_loss)
            self.train_losses['reconstruction'].append(train_recon_loss)
            self.train_losses['kl'].append(train_kl_loss)
            self.train_losses['L2'].append(train_L2_loss)  # CORRECTED: append instead of assign
            
            # ************************** VALIDATION **************************
            self.eval()  
            val_total_loss = 0
            val_recon_loss = 0
            val_kl_loss = 0
            val_L2_loss = 0
            
            with torch.no_grad():
                for x_batch, y_batch in val_dataloader:
                    # Move batch to GPU
                    x_batch = x_batch.to(self.config["device"])
                    y_batch = y_batch.to(self.config["device"])
                    
                    batch_total_loss, batch_recon_loss, batch_kl_loss, L2_loss = self.calculate_loss(x_batch, y_batch, beta)
                    
                    # CORRECTED: Use .item() to extract scalar values
                    val_total_loss += batch_total_loss.item()
                    val_recon_loss += batch_recon_loss.item()
                    val_kl_loss += batch_kl_loss.item()
                    val_L2_loss += L2_loss.item()
                    
                    # CORRECTED: Explicitly delete tensors to free GPU memory
                    del x_batch, y_batch, batch_total_loss, batch_recon_loss, batch_kl_loss, L2_loss
                
                # CORRECTED: Clear GPU cache after validation
                torch.cuda.empty_cache()
            
            # Calculate average validation losses
            val_total_loss /= len(val_dataloader)
            val_recon_loss /= len(val_dataloader)
            val_kl_loss /= len(val_dataloader)
            val_L2_loss /= len(val_dataloader)
            
            self.val_losses['total'].append(val_total_loss)
            self.val_losses['reconstruction'].append(val_recon_loss)
            self.val_losses['kl'].append(val_kl_loss)
            self.val_losses['L2'].append(val_L2_loss)
            
            # **************************  Update Beta  **************************
            '''
            kl_reconstrucion_ratio = beta*val_kl_loss / (val_recon_loss + 1e-12)
            ratio = np.min([np.max([self.config['loss_balance_target'] / (kl_reconstrucion_ratio + 1e-12),
                                   self.config['loss_balance_lr']]),
                                   1/self.config['loss_balance_lr']])
            beta = max(0.0, beta * ratio)
            '''
            kl_reconstruction_ratio = beta*val_kl_loss / (val_recon_loss + 1e-12)
            beta = self.update_beta(epoch, beta)

            self.val_losses['beta'].append(beta)
            
            # Print training progress
            print(f"  Train Loss: Total = {train_total_loss:.4f}, Recon = {train_recon_loss:.4f} ({train_recon_loss/(train_recon_loss+train_kl_loss*beta)*100:.2f}%), KL = {train_kl_loss:.4f} ({(train_kl_loss*beta)/(train_recon_loss+train_kl_loss*beta)*100:.2f}%), L2 = {train_L2_loss:.4f}")
            print(f"  Val Loss: Total = {val_total_loss:.4f}, Recon = {val_recon_loss:.4f}, KL = {val_kl_loss:.4f}, Beta = {beta:.4f}, KL/Reconstruction Ratio = {kl_reconstruction_ratio:.4f}, L2 = {val_L2_loss:.4f}")

            # ************************** Early stopping **************************
            if self.config['posterior_collapse_stopping']:
                if val_kl_loss / (val_recon_loss + 1e-12) < 1e-6:
                    print(f"Posterior collapse detected at epoch {epoch+1}, stopping training")
                    break

            if self.config['static_stopping'] is not None:
                if epoch > self.config['static_stopping']:
                    if np.abs(np.diff(self.val_losses['kl'][-self.config['static_stopping']:]).sum()) < 1e-3 and np.abs(np.diff(self.val_losses['reconstruction'][-self.config['static_stopping']:]).sum()) < 1e-3:
                        print(f"Static stopping triggered after {epoch+1} epochs")
                        break

            if self.config['loss_convergence_stopping'] is not None:
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['loss_convergence_stopping']:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break

    def update_beta(self, epoch, beta):
        if self.config['beta_strategy'] == 'annealing':

            if self.config['annealing_final_beta'] is None or self.config['annealing_steps'] is None:
                raise ValueError("annealing_final_beta and annealing_steps must be set for annealing strategy")
            
            if epoch >= self.config['annealing_steps']:
                beta = self.config['annealing_final_beta']
            elif epoch < self.config['annealing_start_at']:
                beta = self.config['beta']
            else:
                beta = (self.config['annealing_final_beta'] - self.config['beta']) * ((epoch-self.config['annealing_start_at']) / self.config['annealing_steps']) + self.config['beta']

            return beta
        elif self.config['beta_strategy'] == 'constant':
            if self.config['beta'] is None:
                raise ValueError("beta must be set for constant strategy")
            return beta
        elif self.config['beta_strategy'] == 'loss_balance':
            if self.config['loss_balance_target'] is None or self.config['loss_balance_lr'] is None:
                raise ValueError("loss_balance_target and loss_balance_lr must be set for loss balance strategy")
            kl_reconstruction_ratio = beta*self.val_losses['kl'][-1] / (self.val_losses['reconstruction'][-1] + 1e-12)
            ratio = np.min([np.max([self.config['loss_balance_target'] / (kl_reconstruction_ratio + 1e-12),
                                   self.config['loss_balance_lr']]),
                                   1/self.config['loss_balance_lr']])
            beta = max(0.0, beta * ratio)
            return beta
        else:
            raise ValueError("Invalid beta strategy")

    def calculate_loss(self, x_batch, y_batch, beta):
        x_reconstruction, z_mean, z_log_var = self(x_batch, y_batch)

        # Prevent numerical instability
        z_log_var = torch.clamp(z_log_var, min=-10, max=10)

        if torch.isnan(z_mean).any() or torch.isnan(z_log_var).any():
            raise ValueError("NaN detected in z_mean or z_log_var")
        
        # Use MSE for reconstruction loss without sigmoid activation
        reconstruction_loss = F.mse_loss(x_reconstruction, x_batch, reduction='mean')  # CORRECTED: Changed to sum from mean
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))

        # Tikhonov penalty (L2 regularization)
        if self.config['L2_factor'] != 0:
            tikhonov_penalty = sum(torch.sum(param ** 2) for param in self.parameters()) * self.config['L2_factor']
        else:
            tikhonov_penalty = torch.tensor(0.0, device=self.config["device"])

        total_loss = reconstruction_loss + beta * kl_loss + tikhonov_penalty

        return total_loss, reconstruction_loss, kl_loss, tikhonov_penalty

    def prepare_data(self, df):
        self.data['data_indeces'] = df.index

        self.features = [feature for feature in self.features if feature in df.columns]
        self.x_data = df[self.features].to_numpy().astype(np.float32)
        self.y_data = df[self.conditions].to_numpy().astype(np.float32)

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

        self.scaler_y.fit(y_train)
        y_train_std = self.scaler_y.transform(y_train)
        y_val_std = self.scaler_y.transform(y_val)
        y_test_std = self.scaler_y.transform(y_test)

        # CORRECTED: Keep data on CPU initially, move to GPU only when needed in DataLoader
        self.x_train = torch.tensor(x_train_std, dtype=torch.float32)
        self.x_val = torch.tensor(x_val_std, dtype=torch.float32)
        self.x_test = torch.tensor(x_test_std, dtype=torch.float32)
        
        self.y_train = torch.tensor(y_train_std, dtype=torch.float32)
        self.y_val = torch.tensor(y_val_std, dtype=torch.float32)
        self.y_test = torch.tensor(y_test_std, dtype=torch.float32)

        self.num_features = self.x_train.shape[1]
        self.num_targets = self.y_train.shape[1]

        self.data['train_data'] = train_idx
        self.data['val_data'] = val_idx
        self.data['test_data'] = test_idx

        self.input_dim = self.x_train.shape[1]
        self.conditioning_dim = self.y_train.shape[1]

        if self.config['print_logs']:
            print(f"Using device: {self.config['device']}")
            print("Number of training samples:", self.x_train.shape[0])
            print("Number of validation samples:", self.x_val.shape[0])
            print("Number of testing samples:", self.x_test.shape[0])
            print(f"Input dimension: {self.input_dim}")
            print(f"Conditioning dimension: {self.conditioning_dim}")

    def get_latent_space(self):
        # CORRECTED: Move to appropriate device and ensure proper tensor handling
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(self.x_train, dtype=torch.float32).to(self.config["device"])
            y_tensor = torch.tensor(self.y_train, dtype=torch.float32).to(self.config["device"])

            z_mean, z_log_var = self.encoder(x_tensor, y_tensor)

            epsilon = torch.randn_like(z_mean)
            z = z_mean + torch.exp(0.5 * z_log_var) * epsilon
            
            # CORRECTED: Move back to CPU before converting to numpy
            z = z.cpu().detach().numpy()
            z_mean = z_mean.cpu().detach().numpy()

        labels = self.x_data[:,0]

        return z, z_mean, labels

    '''
    def sample(self, profile, n_samples, sample_distance_factor=1.0):
        self.eval()

        profile_transformed = self.scaler_y.transform(profile[self.conditions].values.reshape(1, -1))
        profile_conditions = torch.tensor(profile_transformed, dtype=torch.float32).to(self.config["device"])

        with torch.no_grad():
            # CORRECTED: Create zero tensor on the correct device
            zero_input = torch.zeros(1, self.input_dim).to(self.config["device"])
            z_mean, z_log_var = self.encoder(zero_input, profile_conditions)

            std = torch.exp(0.5 * z_log_var) * sample_distance_factor
            eps = torch.randn(n_samples, z_mean.shape[-1]).to(self.config["device"])
            z_samples = z_mean + eps * std  # n_samples x latent_dim

            conditions_repeated = profile_conditions.repeat(n_samples, 1)

            # CORRECTED: Move to CPU before converting to numpy
            samples = self.decoder(z_samples, conditions_repeated).cpu().numpy()

        samples_transformed = self.scaler_x.inverse_transform(samples)
        df_samples_transformed = pd.DataFrame(samples_transformed, columns=self.features)

        return df_samples_transformed
        '''
    
    '''
    def sample(self, profile, n_samples, sample_distance_factor=1.0):
        self.eval()

        # Standardize condition
        profile_transformed = self.scaler_y.transform(profile[self.conditions].values.reshape(1, -1))
        profile_conditions = torch.tensor(profile_transformed, dtype=torch.float32).to(self.config["device"])

        with torch.no_grad():
            # Sample from standard normal (prior), scaled if needed
            z_samples = torch.randn(n_samples, self.config['latent_dim']).to(self.config["device"]) * sample_distance_factor

            # Repeat condition for each sample
            conditions_repeated = profile_conditions.repeat(n_samples, 1)

            # Decode
            samples = self.decoder(z_samples, conditions_repeated).cpu().numpy()

        samples_transformed = self.scaler_x.inverse_transform(samples)
        df_samples_transformed = pd.DataFrame(samples_transformed, columns=self.features)

        return df_samples_transformed
    '''
    

    def sample(self, profile, n_samples, sample_distance_factor=1.0, sample_from="prior"):
        self.eval()

        # Standardize input feature vector (x)
        profile_input = profile[self.features].values.reshape(1, -1)
        profile_input_transformed = self.scaler_x.transform(profile_input)
        profile_input_tensor = torch.tensor(profile_input_transformed, dtype=torch.float32).to(self.config["device"])

        # Standardize condition vector (y)
        profile_condition = profile[self.conditions].values.reshape(1, -1)
        profile_condition_transformed = self.scaler_y.transform(profile_condition)
        profile_condition_tensor = torch.tensor(profile_condition_transformed, dtype=torch.float32).to(self.config["device"])

        with torch.no_grad():
            if sample_from == "prior":
                # Sample from standard normal (prior), scaled if needed
                z_samples = torch.randn(n_samples, self.config['latent_dim']).to(self.config["device"]) * sample_distance_factor
            elif sample_from == "posterior":
                # Encode the input feature vector to get the posterior distribution
                mu, logvar = self.encoder(profile_input_tensor, profile_condition_tensor)
                std = torch.exp(0.5 * logvar)
                z_samples = mu + std * torch.randn(n_samples, self.config['latent_dim']).to(self.config["device"]) * sample_distance_factor
            else:
                raise ValueError("sample_from must be either 'prior' or 'posterior'")

            # Repeat condition for each sample
            conditions_repeated = profile_condition_tensor.repeat(n_samples, 1)

            # Decode
            samples = self.decoder(z_samples, conditions_repeated).cpu().numpy()

        samples_transformed = self.scaler_x.inverse_transform(samples)
        df_samples_transformed = pd.DataFrame(samples_transformed, columns=self.features)

        return df_samples_transformed   

    def save(self, path):
        # Move model to CPU before saving to ensure compatibility
        device_backup = self.config["device"]
        self.to("cpu")
        torch.save(self, path)
        # Move model back to original device
        self.to(device_backup)
        print("Model saved to: {}".format(path))