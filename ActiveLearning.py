'''
**Author**: Benjamin Urben<br>
**Email**: burben@student.ethz.ch / benjamin.urben@hotmail.ch<br>
**Context**: Master Thesis on "Use of Machine Learning in the Design and Analysis for Steel Connections"<br>
**Institution**: ETH Zürich, Institute of Structural Engineering (IBK)
'''

import copy
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import torch
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import qmc

class ActiveLearning:

    def __init__(self, model, method, df, init_strategy = 'random_sampling',
                 max_iterations=10, batch_size=100, n_init_samples=100):

        self.model = copy.deepcopy(model)
        self.method = method
        self.init_strategy = init_strategy
        self.df = df

        self.n_init_samples = n_init_samples
        self.batch_size = batch_size
        self.max_iterations = max_iterations

        self.config = {'stopping_criteria': None, #'MSE', 'RMSE', 'MAE', 'R2'
                       'stopping_threshold':0.95,
                       'seed': 42, 
                       'z_score_threshold': 2.0,
                       'selection_mode': 'dynamic_batch',
                       'training_mode':'fine_tuning'} #fine_tuning or retraining
        
        #Assures that the initial data is fixed and not randomly selected
        self.df_initial = self.select_initial_data()

        self.performance_template = {'data_ratio': [], 'data_indeces': [], 'MSE': [], 'RMSE': [], 'MAE': [], 'R2': [], 'MCDMS': [], 'MCDUS': []}
        self.methods_available = ['error_based', 
                                  'minimize_MCD_uncertainty',
                                  'minimize_MCD_uncertainty_and_error',  
                                  'random_sampling', 
                                  'cluster_based', 
                                  'latin_hypercube'
                                  ]
        
        self.initiation_methods_available = ['kmeans',
                                             'latin_hypercube', 
                                            'max_min_distance',   
                                            'random_sampling'
                                  ]
        
        # Seed numpy
        np.random.seed(self.config['seed'])
        
        print('** Available Methods **')
        for method_name in self.methods_available:
            print(method_name)

        print('** Available Initiation Methods **')
        for method_name in self.initiation_methods_available:
            print(method_name)

    def trigger_active_learning(self):
        method_name = f"method_{self.method}"
        method = getattr(self, method_name, lambda: "Invalid method")

        if method:
            return method()
        else:
            print(f"Method {method_name} not found")

    def _active_learning_loop(self, select):
        performance = copy.deepcopy(self.performance_template)

        df_train = self.df_initial.copy()
        df_remaining = self.df.drop(df_train.index)

        model = copy.deepcopy(self.model)
        model.config['print_logs'] = False
        model.prepare_data(df_train)
        model.build()
        model.train()

        # Initial metrics
        metrics = model.get_metrics()

        # Update log
        performance['data_ratio'].append(len(df_train) / len(self.df))
        performance['data_indeces'].append(df_train.index.tolist())  
        for metric in metrics:
            performance[metric].append(metrics[metric])

        print(f"Initial R² Score: {metrics['R2']}")

        for i in range(self.max_iterations+1):
            
            print(f"Iteration {i + 1}: Training with {len(df_train)} samples")

            df_new, df_remaining = select(model, df_remaining)

            # Add selected samples to training set.
            df_train = pd.concat([df_train, df_new])

            if len(df_remaining) < self.batch_size:
                print('No more samples to select')
                break
            
            if self.config['training_mode'] == 'fine_tuning':
                model.fine_tune(df_new)
            elif self.config['training_mode'] == 'retraining':
                model.add_training_data(df_new)
                model.train()
            else:
                print('Invalid training mode. Use fine_tuning or retraining.')
                break

            # Evaluate model performance.
            metrics = model.get_metrics()

            # Update log
            performance['data_ratio'].append(len(df_train) / len(self.df))
            performance['data_indeces'].append(df_train.index.tolist())  # Convert index to list
            for metric in metrics:
                performance[metric].append(metrics[metric])
            print(f"    R² Score: {metrics['R2']}")

            # Stopping criteria
            if self.config['stopping_criteria'] is not None:
                if self.config['stopping_criteria'] not in metrics:
                    print(f"Stopping metric {self.config['stopping_metric']} not found in metrics.")
                    continue
                if metrics[self.config['stopping_criteria']] > self.config['stopping_threshold']:
                    print(f"Stopping criteria reached: {self.config['stopping_criteria']} > {self.config['stopping_threshold']}")
                    break

        #self.performance = performance
        #self.df_initial = self.df.loc[df_train.index]

        return performance

    # **** INITIATION STRATEGIES ****

    def select_initial_data(self):
        if self.init_strategy == 'latin_hypercube':
            return self._latin_hypercube_sampling()
        elif self.init_strategy == 'kmeans':
            return self._kmeans_sampling()
        elif self.init_strategy == 'max_min_distance':
            return self._max_min_distance_sampling()
        elif self.init_strategy == 'random_sampling':
            return self._random_sampling()
        else:
            print('Invalid initial sampling strategy. Using random sampling.')
            return self.df.sample(n=self.n_init_samples, random_state=self.config['seed'])
        
    def _random_sampling(self):
        return self.df.sample(n=self.n_init_samples, random_state=self.config['seed'])
        
    def _latin_hypercube_sampling(self):
        df_features = self.df[self.model.features]

        sampler = qmc.LatinHypercube(d=len(self.model.features), seed=42)
        sample = sampler.random(n=self.n_init_samples)
        scaled_sample = qmc.scale(sample, df_features.min().values, df_features.max().values)
        distances = euclidean_distances(df_features.values, scaled_sample)

        selected_indices = np.argmin(distances, axis=0)

        return self.df.iloc[selected_indices]

    def _kmeans_sampling(self):
        df_features = self.df[self.model.features]

        kmeans = KMeans(n_clusters=self.n_init_samples, random_state=self.config['seed'], n_init=10)
        kmeans.fit(df_features)

        selected_indices = np.unique(kmeans.labels_, return_index=True)[1]
        
        return self.df.iloc[selected_indices]

    def _max_min_distance_sampling(self):
        feature_df = self.df[self.model.features]
        selected = [np.random.choice(feature_df.index)]
        for _ in range(self.n_init_samples - 1):
            dists = euclidean_distances(feature_df.loc[selected], feature_df).min(axis=0)
            next_index = feature_df.index[np.argmax(dists)]
            selected.append(next_index)
        
        return self.df.loc[selected]

    # **** ACTIVE LEARNING STRATEGIES ****

    def select_method(self, method, model, df_remaining):
        if method == 'error_based':
            return self.select_error_based(model, df_remaining)
        elif method == 'cluster_based':
            return self.select_cluster_based(model, df_remaining)
        elif method == 'latin_hypercube':
            return self.select_latin_hypercube(model, df_remaining)
        elif method == 'minimize_MCD_uncertainty':
            return self.select_minimize_MCD_uncertainty(model, df_remaining)
        elif method == 'minimize_MCD_uncertainty_and_error':
            return self.select_minimize_MCD_uncertainty_and_error(model, df_remaining)
        elif method == 'deviation_number':
            return self.select_deviation_number(model, df_remaining)
        elif method == 'random_sampling':
            return self.select_random_sampling(model, df_remaining)
        else:
            print('Invalid initial sampling strategy. Using random sampling.')
            return self.df.sample(n=self.n_init_samples, random_state=self.config['seed'])

    def method_error_based(self):
        def select(model, df_remaining):
            return self.select_error_based(model, df_remaining)
        return self._active_learning_loop(select)
    
    def select_error_based(self, model, df_remaining):
        y_pred, _ = model.predict(df_remaining)
        y_true = df_remaining[model.target].to_numpy()

        score = np.abs(y_pred - y_true)
        df_remaining['score'] = score

        # Select samples with highest error.
        if self.config['selection_mode'] == 'fixed_batch':
            df_new = df_remaining.nlargest(self.batch_size, 'score').drop(columns=['score'])
        elif self.config['selection_mode'] == 'dynamic_batch':
            mu = np.mean(score)
            std = np.std(score)
            threshold = mu + std * self.config['z_score_threshold']
            select_indices = np.where(score > threshold)[0]
            df_new = df_remaining.iloc[select_indices].drop(columns=['score'])
        else:
            raise ValueError("Invalid selection mode. Choose 'fixed_batch' or 'dynamic_batch'.")
        
        df_remaining = df_remaining.drop(df_new.index)

        return df_new, df_remaining

    def method_minimize_MCD_uncertainty(self):
        def select(model, df_remaining):
            return self.select_minimize_MCD_uncertainty(model, df_remaining)
        return self._active_learning_loop(select)
    
    def select_minimize_MCD_uncertainty(self, model, df_remaining):
        X_remaining = df_remaining[model.features].to_numpy()
        X_tensor = torch.tensor(X_remaining, dtype=torch.float32)

        mean_pred, std_pred, _ = model.mc_dropout_uncertainty(df_remaining,num_samples=100,dropout_prob=0.1)
        score = std_pred.flatten() / mean_pred.flatten()
        df_remaining['score'] = score

        if self.config['selection_mode'] == 'fixed_batch':
            df_new = df_remaining.nlargest(self.batch_size, 'score').drop(columns=['score'])
        elif self.config['selection_mode'] == 'dynamic_batch':
            mu = np.mean(score)
            std = np.std(score)
            threshold = mu + std * self.config['z_score_threshold']
            select_indices = np.where(score > threshold)[0]
            df_new = df_remaining.iloc[select_indices].drop(columns=['score'])
        else:
            raise ValueError("Invalid selection mode. Choose 'fixed_batch' or 'dynamic_batch'.")

        df_remaining = df_remaining.drop(df_new.index)
        
        return df_new, df_remaining
    
    def method_minimize_MCD_uncertainty_and_error(self):
        def select(model, df_remaining):
            return self.select_minimize_MCD_uncertainty_and_error(model,df_remaining)
        return self._active_learning_loop(select)
    
    def select_minimize_MCD_uncertainty_and_error(self, model, df_remaining):
        X_remaining = df_remaining[model.features].to_numpy()
        X_remaining_std = model.scaler_x.transform(X_remaining)
        X_tensor = torch.tensor(X_remaining_std, dtype=torch.float32)

        y_remaining = df_remaining[model.target].to_numpy()

        mean_pred, std_pred, _ = model.mc_dropout_uncertainty(df_remaining, num_samples=100,dropout_prob=0.1)

        # Calculate absolute error (not relative error)
        error = np.abs(mean_pred - y_remaining)

        # Important to consider Coefficient of variation (CV) to avoid bias towards high mean predictions
        cv_pred = std_pred / mean_pred

        ''''
        # Give each sample a rank based on how close it is to ideal prediction (1.0) (Rank 1 is best sample)
        sorted_indices_mean = np.argsort(error.flatten())
        ranks_mean = np.empty_like(sorted_indices_mean)
        ranks_mean[sorted_indices_mean] = np.arange(len(sorted_indices_mean))
        
        # Give each sample a rank based on how uncertain the model is (Rank 1 is best sample)
        sorted_indices_cv = np.argsort(cv_pred.flatten())
        ranks_cv = np.empty_like(sorted_indices_cv)
        ranks_cv[sorted_indices_cv] = np.arange(len(sorted_indices_cv))
        '''

        # For each sample add the ranks of the two metrics
        #score = alpha*ranks_mean + (1-alpha)*ranks_cv
        scaler = StandardScaler()
        score = scaler.fit_transform(error) + scaler.fit_transform(cv_pred)

        df_remaining['score'] = score

        #df_remaining['score'] = std_pred.flatten() + pred_truth_ratio.flatten()

        if self.config['selection_mode'] == 'fixed_batch':
            df_new = df_remaining.nlargest(self.batch_size, 'score').drop(columns=['score'])
        elif self.config['selection_mode'] == 'dynamic_batch':
            mu = np.mean(score)
            std = np.std(score)
            threshold = mu + std * self.config['z_score_threshold']
            select_indices = np.where(score > threshold)[0]
            df_new = df_remaining.iloc[select_indices].drop(columns=['score'])
        else:
            raise ValueError("Invalid selection mode. Choose 'fixed_batch' or 'dynamic_batch'.")
        
        df_remaining = df_remaining.drop(df_new.index)
        
        return df_new, df_remaining
    
    def method_latin_hypercube(self):
        def select(model, df_remaining):
            return self.select_latin_hypercube(model, df_remaining)
        return self._active_learning_loop(select)
    
    def select_latin_hypercube(self, model, df_remaining):
        n_samples = min(self.batch_size, len(df_remaining))

        sampler = qmc.LatinHypercube(d=len(model.features), seed=self.config['seed'])
        lhs_samples = sampler.random(n=n_samples)

        min_vals = df_remaining[model.features].min().values
        max_vals = df_remaining[model.features].max().values
        scaled_samples = lhs_samples * (max_vals - min_vals) + min_vals

        feature_array = df_remaining[model.features].values
        tree = KDTree(feature_array)

        _, nearest_indices = tree.query(scaled_samples)

        selected_indices = df_remaining.index[nearest_indices]
        df_new = df_remaining.loc[selected_indices]
        df_remaining = df_remaining.drop(selected_indices)

        return df_new, df_remaining

    def method_deviation_number(self):
        def select(model, df_remaining):
            return self.select_deviation_number(model, df_remaining)
        return self._active_learning_loop(select)
    
    def select_deviation_number(self, model, df_remaining):
        y_pred, _ = model.predict(df_remaining)
        median_pred = np.median(y_pred)

        score = np.abs(y_pred - median_pred)
        df_remaining['score'] = score
        
        if self.inverse:
            df_new = df_remaining.nsmallest(self.inverse_batch_size, 'score').drop(columns=['score'])
        else:
            if self.config['selection_mode'] == 'fixed_batch':
                df_new = df_remaining.nlargest(self.batch_size, 'score').drop(columns=['score'])
            elif self.config['selection_mode'] == 'dynamic_batch':
                mu = np.mean(score)
                std = np.std(score)
                threshold = mu + std * self.config['z_score_threshold']
                select_indices = np.where(score > threshold)[0]
                df_new = df_remaining.iloc[select_indices].drop(columns=['score'])

        df_remaining = df_remaining.drop(df_new.index)

        return df_new, df_remaining
    
    def method_random_sampling(self):
        def select(model, df_remaining):
            return self.select_random_sampling(model, df_remaining)

        return self._active_learning_loop(select)
    
    def select_random_sampling(self, model, df_remaining):
        df_new = df_remaining.sample(n=min(self.batch_size, len(df_remaining)), random_state=self.config['seed'])
        df_remaining = df_remaining.drop(df_new.index)
        return df_new, df_remaining

    def method_cluster_based(self):
        def select(model, df_remaining):
            return self.select_cluster_based(model, df_remaining)
        return self._active_learning_loop(select)
    
    def select_cluster_based(self, model, df_remaining):
        features_remaining = df_remaining[model.features].to_numpy()
        n_clusters = min(self.batch_size, len(features_remaining))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config['seed'], n_init='auto').fit(features_remaining)
        distances = kmeans.transform(features_remaining)[np.arange(len(features_remaining)), kmeans.labels_]
        
        selected_indices = np.argpartition(distances, self.batch_size)[:self.batch_size]
        df_new = df_remaining.iloc[selected_indices]
        df_remaining = df_remaining.drop(df_new.index)

        return df_new, df_remaining
