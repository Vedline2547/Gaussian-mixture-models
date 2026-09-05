# Gaussian Mixture Model (GMM) Clustering

This project demonstrates **Gaussian Mixture Model (GMM) clustering** using Python and Scikit-learn.

GMM is an unsupervised machine learning algorithm that groups data points into clusters based on probability distributions. Unlike K-Means, GMM provides the **probability of each data point belonging to every cluster**, making it useful for soft or probabilistic clustering.

## 📌 Project Overview

The project uses a small 2D dataset containing six data points and applies a Gaussian Mixture Model to divide them into **three clusters**.

The model then produces:

* **Cluster labels** for each data point
* **Cluster probabilities** showing how likely each point is to belong to each cluster

## 🛠️ Technologies Used

* Python
* NumPy
* Scikit-learn
* Gaussian Mixture Model

## 📦 Installation

Install the required libraries using:

```bash
pip install numpy scikit-learn
```

## 📊 Dataset

The example dataset contains six points in a two-dimensional feature space:

```python
X = np.array([
    [1, 2],
    [2, 2],
    [2, 3],
    [8, 7],
    [8, 8],
    [25, 80]
])
```

The data contains three naturally separated groups:

* `[1,2]`, `[2,2]`, `[2,3]`
* `[8,7]`, `[8,8]`
* `[25,80]`

## 🚀 How It Works

### 1. Import the libraries

```python
from sklearn.mixture import GaussianMixture
import numpy as np
```

### 2. Create the dataset

```python
X = np.array([
    [1,2],
    [2,2],
    [2,3],
    [8,7],
    [8,8],
    [25,80]
])
```

### 3. Initialize the GMM model

```python
gmm = GaussianMixture(
    n_components=3,
    random_state=42
)
```

`n_components=3` tells the model to identify **three Gaussian distributions/clusters**.

### 4. Fit the model

```python
gmm.fit(X)
```

The model learns the statistical characteristics of the three clusters from the dataset.

### 5. Predict cluster labels

```python
labels = gmm.predict(X)
```

This assigns each data point to the cluster with the highest probability.

Example:

```text
Cluster Labels: [0 0 0 2 2 1]
```

The actual cluster numbers may be different because cluster labels are arbitrary.

### 6. Calculate cluster probabilities

```python
probs = gmm.predict_proba(X)
```

This returns the probability of each data point belonging to each cluster.

For example:

```text
[0.95, 0.03, 0.02]
```

means the point has:

* 95% probability of belonging to Cluster 0
* 3% probability of belonging to Cluster 1
* 2% probability of belonging to Cluster 2

## 🧠 Understanding GMM

Gaussian Mixture Model assumes that the data is generated from a mixture of several Gaussian probability distributions.

Each cluster is represented by a Gaussian distribution with parameters such as:

* Mean
* Variance
* Covariance
* Mixing probability

The model estimates these parameters and determines how likely each point is to belong to each cluster.

## 🔑 GMM vs K-Means

| Feature                | K-Means         | GMM                           |
| ---------------------- | --------------- | ----------------------------- |
| Type                   | Unsupervised    | Unsupervised                  |
| Clustering             | Hard clustering | Soft/probabilistic clustering |
| Output                 | Cluster label   | Cluster label + probabilities |
| Main concept           | Distance        | Probability                   |
| Cluster representation | Centroid        | Gaussian distribution         |
| `predict_proba()`      | ❌               | ✅                             |

One of the major advantages of GMM is that it provides **membership probabilities** rather than only assigning a point to one cluster.

## 📈 Output

The program prints:

```text
Cluster Labels: [0 0 0 2 2 1]

Cluster Probabilities:
[[... ... ...]
 [... ... ...]
 [... ... ...]
 [... ... ...]
 [... ... ...]
 [... ... ...]]
```

The labels show the assigned cluster, while the probability matrix shows the model's confidence for each cluster assignment.

## 📁 Project Structure

```text
Gaussian-Mixture-Model/
│
├── main.py
├── README.md
└── requirements.txt
```

## 📄 requirements.txt

```text
numpy
scikit-learn
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## 🎯 Learning Objectives

This project demonstrates how to:

* Implement Gaussian Mixture Model clustering
* Perform unsupervised learning with Scikit-learn
* Specify the number of clusters using `n_components`
* Generate cluster labels using `predict()`
* Generate membership probabilities using `predict_proba()`
* Understand the difference between hard and soft clustering
* Understand the basic difference between GMM and K-Means

## 🌍 Applications of GMM

Gaussian Mixture Models can be used in areas such as:

* Customer segmentation
* Anomaly detection
* Image segmentation
* Pattern recognition
* Speech recognition
* Density estimation
* Data analysis
* Clustering engineering or scientific measurements

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer | Future AI Engineer

---

⭐ If you found this project useful, consider giving the repository a star!
