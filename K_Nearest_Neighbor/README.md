# 📈 | K-Nearest Neighbor (KNN)

K-Nearest Neighbor (KNN) is a supervised machine learning algorithm used for classification and regression tasks. It operates on the principle of proximity, where it classifies a new data point based on the majority class of its k-nearest neighbors.

---

### 🧪 | Algorithm Overview

KNN works by calculating the distance between the input data point and all other data points in the training set. It then selects the k-nearest data points based on some distance metric, commonly Euclidean distance. The class or value of the new data point is determined by the majority class or average value of its k-nearest neighbors.

### 🧮 | Calculating Distance

The distance between two points \( \mathbf{p} \) and \( \mathbf{q} \) in an n-dimensional space can be calculated using various distance metrics, with Euclidean distance being the most commonly used:

$$
d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}
$$

### 🔋 | Parameter Tuning

Key parameters in KNN include the value of k (the number of neighbors to consider) and the choice of distance metric. The optimal values for these parameters can significantly impact the performance of the algorithm and may require tuning through techniques like cross-validation.

### 📚 | Resources

This directory contains an implementation of the K-Nearest Neighbor algorithm from scratch, along with a script for training and evaluation. Additionally, examples using the KNeighborsClassifier and KNeighborsRegressor models from Scikit-Learn are provided for reference.

To delve deeper into the topic, consider exploring the following resources:

* *Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer.*
* *Marsland, S. (2015). Machine Learning: An Algorithmic Perspective. CRC Press.*
* *Raschka, S., & Mirjalili, V. (2019). Python Machine Learning, 3rd Edition. Packt Publishing.*

These resources offer comprehensive insights into the theoretical foundations and practical applications of KNN and other machine learning algorithms.