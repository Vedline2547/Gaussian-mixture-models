# Import necessary libraries
from sklearn.mixture import GaussianMixture
import numpy as np
# Sample data(eg points in 2D space)
X = np.array([[1,2],[2,2],[2,3],[8,7],[8,8],[25,80]])
# Initialise and fit the model
gmm = GaussianMixture(n_components=3,random_state=42)
gmm.fit(X)
# Get cluster labels and probabilities
labels = gmm.predict(X)
probs = gmm.predict_proba(X)
print("Cluster Labels: ",labels)
print("Cluster Probabilities: ",probs)