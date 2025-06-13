# %%
import os
import pandas as pd
from Settings import Directory

class DataReader:
    def __init__(self):
        directory = Directory()

        self.file_names = os.listdir("Data")
        self.loaded_file_name = None

        for file_name in self.file_names:
            print(file_name)

    def load(self, file_name):
        if file_name.endswith('.csv'):
            file = pd.read_csv(f"Data/{file_name}")
        elif file_name.endswith('.xlsx'):
            file = pd.read_excel(f"Data/{file_name}")
        else:
            print(f"File type unknown: {file_name}")
            return
        
        self.loaded_file_name = file_name
        print(f'{file_name}: {file.shape}')

        return file

    def load_all(self):
        self.files = {}
        for file_name in self.file_names:
            print('Reading', file_name)
            if file_name.endswith('.csv'):
                file = pd.read_csv(f"Data/{file_name}")
            elif file_name.endswith('.xlsx'):
                file = pd.read_excel(f"Data/{file_name}")
            else:
                print(f"File type unknown: {file_name}")
                continue

            self.files[file_name] = file

        for file_name in self.file_names:
            print(f'{file_name}: {self.files[file_name].shape}')

    def reduce(self, df, model, n_reduce=None, degree=1):
        features = model.features

        current_df = df.copy()
        removed_features = []  # To track the processed columns
        
        if n_reduce is None:
            n_reduce = model.data['data_reduction_factor']

        for _ in range(n_reduce):
            if current_df.shape[1] <= 1:  # Stop if only one column remains
                break

            print(f"Current size: {current_df.shape}")

            # Find the column with the highest occurrence of any single value
            most_frequent_col = None
            max_occurrences = 0
            value_to_keep = None
            
            # Iterate through columns that are in the features list and have not been processed
            for col in current_df.columns:
                if col in removed_features or col not in features:
                    continue  # Skip already processed columns or columns not in the features list

                # Drop NaNs before performing value_counts
                value_counts = current_df[col].dropna().value_counts()
                
                if not value_counts.empty:
                    most_common_value = value_counts.idxmax()
                    occurrences = value_counts.max()

                    # If this column has more occurrences of a value than previously found, update
                    if occurrences > max_occurrences:
                        most_frequent_col = col
                        max_occurrences = occurrences
                        value_to_keep = most_common_value

            if most_frequent_col is None:
                break  # Exit if no column is found (avoid infinite loop)

            # Keep only rows where the most frequent value in that column appears
            current_df = current_df[current_df[most_frequent_col] == value_to_keep]

            # Mark the column as processed and append to removed_features
            print("Processed column:", most_frequent_col)
            removed_features.append(most_frequent_col)

            reduced_features = [item for item in features if item not in removed_features]

            model.features = reduced_features
            
        # Save remaining indeces to the model
        model.data['data_file_name'] = self.loaded_file_name
        model.data['data_indeces'] = current_df.index.tolist()

        return current_df, model

