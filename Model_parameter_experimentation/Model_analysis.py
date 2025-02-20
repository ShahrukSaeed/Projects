import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from ydata_profiling import ProfileReport
import torch
import torch.nn as nn
import torch.nn.functional as F
import yellowbrick
import chardet
import random

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


# Creating master dataframe for customer and spotify datasets

with open("D:/GTech/Spring 2025/Machine Learning/Assignment_1/spotify-2023.csv", 'rb') as f:
    result = chardet.detect(f.read())

spotify_master_df = pd.read_csv("D:/GTech/Spring 2025/Machine Learning/Assignment_1/spotify-2023.csv", encoding=result['encoding'])
#profile = ProfileReport(spotify_master_df, title="Spotify Profiling Report")
#profile.to_file("spotify_report.html")

customer_pers_master_df = pd.read_csv("D:/GTech/Spring 2025/Machine Learning/Assignment_1/marketing_campaign.csv",sep='\t')
#profile = ProfileReport(customer_pers_master_df, title="customer Profiling Report")
#profile.to_file("customer_report.html")



# used to set seeds for neural net reproducibility
def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# initializing random state seeds
set_all_seeds(72)

# Defining neural network

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, output_dim=2):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of neurons in the hidden layer.
            output_dim (int): Number of classes (or regression outputs).
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Cleaning datasets

# setting working customer dataset
customer_df = customer_pers_master_df

# removing missing rows
customer_df.dropna(inplace=True) # this is due to the low number of missing rows(1.1%) which wont have much impact on training

# removing nonvaluable columns

customer_df.drop(["ID","Z_CostContact","Z_Revenue"], inplace = True) # removing ID, costcontact, revenue because they are either constant or completely unique values
                                                                           # constant values do not provide an information since the value is the same for all instances
                                                                           # completely unique values can cause a model to overfit or increase error

# removing names due to high dimensionality which could lead to overfitting


#
features_customer = ['Income', 'SpendingScore', 'Age']
target_customer = 'Segment'

# Convert features and target
X_customer = customer_df[features_customer].values.astype(np.float32)
y_customer = customer_df[target_customer].values  # Make sure this is numeric (or encode it)

# If the target is categorical (e.g., strings), consider encoding it:
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_customer = le.fit_transform(y_customer)

#############################################
# 3. Create a Skorch Pipeline for the Customer Dataset
#############################################
# Define the Skorch-wrapped neural network for the customer dataset.
net_customer = NeuralNetClassifier(
    module=SimpleNN,
    module__input_dim=X_customer.shape[1],  # Number of features
    module__hidden_dim=50,  # Hidden layer size (adjust as needed)
    module__output_dim=len(np.unique(y_customer)),  # Number of target classes
    max_epochs=20,
    lr=0.1,
    # Uncomment and set device if you have a GPU:
    # device='cuda' if torch.cuda.is_available() else 'cpu',
)

pipeline_customer = Pipeline([
    ('scaler', StandardScaler()),
    ('net', net_customer)
])

# Split the customer data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_customer, y_customer, test_size=0.2, random_state=42)
pipeline_customer.fit(X_train_c, y_train_c)
print("Customer Dataset Accuracy: {:.2f}%".format(pipeline_customer.score(X_test_c, y_test_c) * 100))

#############################################
# 4. Prepare the Top Spotify Songs 2023 Dataset
#############################################
# Replace 'top_spotify_songs_2023.csv' with your actual file path.
spotify_df = pd.read_csv("top_spotify_songs_2023.csv")

# Example cleaning (adjust as needed):
spotify_df.dropna(inplace=True)

# Inspect the dataset and decide on features and target.
# For demonstration, assume we use features like 'danceability', 'energy', and 'loudness'
# and that there is a target column 'genre' we wish to classify.
features_spotify = ['danceability', 'energy', 'loudness']
target_spotify = 'genre'

X_spotify = spotify_df[features_spotify].values.astype(np.float32)
y_spotify = spotify_df[target_spotify].values

# If necessary, encode the genre labels:
# le_spotify = LabelEncoder()
# y_spotify = le_spotify.fit_transform(y_spotify)

#############################################
# 5. Create a Skorch Pipeline for the Spotify Dataset
#############################################
net_spotify = NeuralNetClassifier(
    module=SimpleNN,
    module__input_dim=X_spotify.shape[1],
    module__hidden_dim=50,
    module__output_dim=len(np.unique(y_spotify)),
    max_epochs=20,
    lr=0.1,
    # device='cuda' if torch.cuda.is_available() else 'cpu',
)

pipeline_spotify = Pipeline([
    ('scaler', StandardScaler()),
    ('net', net_spotify)
])

# Split the Spotify data
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_spotify, y_spotify, test_size=0.2, random_state=72)
pipeline_spotify.fit(X_train_s, y_train_s)
print("Spotify Dataset Accuracy: {:.2f}%".format(pipeline_spotify.score(X_test_s, y_test_s) * 100))