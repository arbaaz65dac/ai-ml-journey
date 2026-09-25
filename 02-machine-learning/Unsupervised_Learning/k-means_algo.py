import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs # Generate a dataset  
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=42)
# print(X)
df = pd.DataFrame(X,columns=['Feature_1', 'Feature_2'])
# print(df.head())

# Standard Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
# print(X_scaled)

# ==== Elbow Method Impl ======

inertia = []   #WCSS
K_range = range(1,11)

for k in K_range:
    Kmeans = KMeans(n_clusters=k, random_state=42)
    Kmeans.fit(X_scaled)
    inertia.append(Kmeans.inertia_)

# print("WCSS:",inertia)
# WCSS: [
#         999.9999999999999, 
#         297.8954141051722, 
#         11.575484723104978, 
#         9.752067977356841, 
#         8.257175272446283, 
#         6.9175773204167985, 
#         6.334755391595289, 
#         5.704177177901429, 
#         5.0602341335320755, 
#         4.762361898130396
#       ]
plt.plot(K_range, inertia, marker='o')
# plt.show()
#  Hence By examining all these we decide K = 3

Kmeans_final = KMeans(n_clusters=3, random_state=42)
cluster_labels = Kmeans_final.fit_predict(X_scaled)
# print(cluster_labels)
df['cluster'] = cluster_labels

sns.scatterplot(x=df['Feature_1'],
                y=df['Feature_2'],
                hue=df['cluster'],
                palette='viridis')  
# plt.show()
  

