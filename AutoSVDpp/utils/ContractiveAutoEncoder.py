import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch
from torch import miopen_depthwise_convolution, nn, optim
from torch.utils.data import DataLoader, TensorDataset

class CAE():
    PATH_SMALL = 'datasets/smallsets/'
    PATH_SAMPLE = 'datasets/samplesets/'

    def __init__(self, path=PATH_SAMPLE):
        filename = "sample_business.json"
        df_business = pd.read_json(os.path.join(path, filename))
        df_business.to_csv(os.path.join(path, 'restaurant_features.csv'))
        self.df_business = df_business

    def train(self, path=PATH_SAMPLE, epochs=30):
        ## Step 1: Load and Preprocess Data
        # Load the dataset
        df = pd.read_csv(os.path.join(path, 'restaurant_features.csv'))
        
        # Assuming that the first two columns are 'business_id' and 'name'
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('business_id')
        categorical_cols.remove('name')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('Unnamed: 0')

        # One-hot encoding for categorical features
        onehot = OneHotEncoder(sparse=False)
        categorical_data = onehot.fit_transform(df[categorical_cols])

        # Normalize the numeric features
        scaler = MinMaxScaler()
        numeric_data = scaler.fit_transform(df[numeric_cols])

        # Combine categorical and numeric features
        combined_features = np.hstack((categorical_data, numeric_data))

        # Convert to PyTorch tensor
        features_tensor = torch.tensor(combined_features, dtype=torch.float)
        self.features_tensor = features_tensor

        ## Step 2: Build the Contractive Autoencoder
        class ContractiveAutoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim=10):
                super(ContractiveAutoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, encoding_dim),
                    nn.ReLU())
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, input_dim),
                    nn.Sigmoid())

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded

        # Instantiate the model
        input_dim = features_tensor.shape[1]
        model = ContractiveAutoencoder(input_dim)

        ## Step 3: Train the Autoencoder
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create DataLoader
        dataset = TensorDataset(features_tensor, features_tensor)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Train the model
        losses = []
        for epoch in range(epochs):
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                _, decoded = model(inputs)
                loss = criterion(decoded, targets)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        self.model = model
        return losses

    def extract_features(self, path=PATH_SAMPLE):
        self.model.eval()
        with torch.no_grad():
            encoded_features, _ = self.model(self.features_tensor)

        # Convert to DataFrame
        encoded_features_df = pd.DataFrame(encoded_features.numpy())
        print(encoded_features_df)

        # Save the encoded features to a CSV file
        encoded_features_df.to_csv(os.path.join(path, 'restaurant_features_encoded.csv'))

    def save_model(self, path=PATH_SAMPLE):
        torch.save(self.model.state_dict(), os.path.join(path, 'CAEModel.pth'))

    def load_model(self, path=PATH_SAMPLE):
        state_dict = torch.load(os.path.join(path, 'CAEModel.pth'))
        self.model.load_state_dict(state_dict)
