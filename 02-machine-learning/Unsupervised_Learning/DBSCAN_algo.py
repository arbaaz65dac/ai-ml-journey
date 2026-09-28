# Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons # Generate a dataset  
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


X, y_true = make_moons(n_samples=500, noise=0.05, random_state=42)

df = pd.DataFrame(X,columns=['Feature_1', 'Feature_2'])
# print(df.head())

# Standard Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
# print(X_scaled)

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

df['dbscan_cluster'] = dbscan_labels

sns.scatterplot(x=df['Feature_1'],
                y=df['Feature_2'],
                hue=df['dbscan_cluster'],
                palette='tab10') 
plt.show()
