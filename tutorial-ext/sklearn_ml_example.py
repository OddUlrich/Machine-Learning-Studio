import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

# Import the tools to perform cross-validation.
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Persist out model forr future use.
from sklearn.externals import joblib

# Load red wine data.
# read_csv() : we can load any CSV file, even from a remote URL!
dataset_dir = './winequality-red.csv'
data = pd.read_csv(dataset_dir, sep = ';')

# Output the first 5 rows of data.
print data.head()

print data.shape

# Print the summary statistics.
print data.describe()

# Split data into training and test sets.