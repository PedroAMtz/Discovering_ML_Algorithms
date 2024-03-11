# 📈 | Linear Regression 
Linear Regression is a supervised machine learning algorithm used for regression problems where the simplest interpretation describes a linear model that best fit´s corresponding data values from a dataset.

---

### 🧪 | Formulation 

¿Remmember how does the equation of a straight line looks like? It was something like this:

$$
y = mx + b
$$

 From this equation we can observe how x and y are related and how we can tell y in terms of x (for a given x) just by modifying m and b, then a prediction y will be a weighted sum of the input features plus the bias. 

$$
y_{\text{pred}=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n}
$$

In this equation:
*  \(y_{\text{pred}}\) is the predicted value.
* \(n\) is the number of features.
*  \({x_i}\) is the ith  feature value.
* \({\theta_j}\) is the jth model parameter that.describes weights and bias term.

### 🧮 | Calculating Error

If we want to optimize the model, we need to parametrize a function from which we can calculate the gradient. We can then obtain a new set of parameters by searching for the global minimization of the error/cost function. In this case, we aim to minimize it because a lesser error results in a better model.

For this purpose we use the Mean Square Error (MSE):

$$
MSE = \frac{1}{N} \sum_{i=1}^n (y_i - y_{\text{pred}})^2
$$

Mean Square Error is a cost function that calculates the difference (distance) between two points, in this case being y_true and y_predicted, raises to the power of two (square) and get´s the mean for the total number of samples present in the training data.

### 🔋 | Optimization 

From the MSE equation we update parameters first calculating the derivatives of the error (MSE) with respect to weights and| bias, which are the parameters to adjust/optimize. The following equations are the respective derivatives.

$$
\frac{df}{dw} = \frac{1}{N}\sum_{i=1}^n -2x_i(y_i - (w_ix_{i} + b_i))
$$

$$
\frac{df}{db} = \frac{1}{N}\sum_{i=1}^n -2(y_i - (w_ix_{i} + b_i))
$$

After computing the derivatives we then update weights and bias parameters with these final equations:

$$
w_{\text{new}} = w_{\text{current}} - \alpha \cdot \frac{df}{dw}
$$

$$
b_{\text{new}} = b_{\text{current}} - \alpha \cdot \frac{df}{db}
$$

Where alpha is the learning rate that describes the size of a step given an optimization algorithm such as Gradient Descent.

### 📚 | Resources

In this directory you will find a Linear Regression model implementation from scratch, along with a training script. I'm also working on a simple example using the LinearRegression model from Scikit-Learn. If you want to explore the topic more in-depth, I'll share my references with you.

* *Géron, A. (2023). Hands-on machine learning with scikit-learn, Keras, and tensorflow: Concepts, tools, and techniques to build Intelligent Systems. O’Reilly.*
* *Deisenroth, M. P., Ong, C. S., & Faisal, A. A. (2021). Mathematics for Machine Learning. Cambridge University Press.*
