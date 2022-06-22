import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('application_data.csv', engine='pyarrow').dropna()

features = list(df.columns)
features.remove('TARGET')

X = df[features]
y = df.TARGET

s = (X.dtypes == 'object')
object_cols = list(s[s].index)
label_X = X.copy()
label_X[object_cols] = OrdinalEncoder().fit_transform(X[object_cols])

X = label_X

model = KNeighborsClassifier(n_neighbors=9, metric='euclidean')
model.fit(X, y)