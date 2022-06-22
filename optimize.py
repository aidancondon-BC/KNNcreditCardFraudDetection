import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
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
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

y_vals = []
for K in range(1, 100):
    model = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
    model.fit(train_X, train_y)
    final_pred = model.predict(val_X)
    y_vals.append(accuracy_score(val_y, final_pred))
    print(accuracy_score(val_y, final_pred))

plt.plot(range(1, 100), y_vals)
plt.show()