README

The Project is an analysis of Neural Network, KNN, and Support Vector Machines algorithms on
2 classification problems one using an imbalanced target class dataset and the 
other using a balanced target class dataset. The goal is of the project is to explore different
parameters of each model on performance and describe why a model with a specific configuration
performs or underperforms.


### README: Package Installation

#### Requirements
Python 3.8 or higher.

#### Installation
Run the following command to install all required packages:

```
pip install pandas numpy matplotlib scikit-learn skorch ydata-profiling torch chardet
```

#### Packages Used
- `pandas`, `numpy`: Data manipulation and numerical operations.
- `matplotlib`: Data visualization.
- `scikit-learn`: Machine learning utilities (preprocessing, classification, evaluation).
- `skorch`: PyTorch wrapper for neural networks.
- `ydata-profiling`: Data analysis reports.
- `torch`: Deep learning framework.
- `chardet`: Character encoding detection.

#### Verification
Run the following to check installation:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyperch.neural.backprop_nn import BackpropModule
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from skorch import NeuralNetClassifier
from torch import nn, optim
import chardet
import random

print("Setup successful!")
```

#### Run the code

update the file paths for the spotify and customer datasets in line 32 and 35
in the code the files should be in csv format then run the code.

If no errors occur, the setup is complete.

overleaf link: https://www.overleaf.com/read/mwbhtyqwgyqp#3d34ea