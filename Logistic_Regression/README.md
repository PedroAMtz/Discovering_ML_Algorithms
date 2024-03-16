# 🎨 | Logistic Regression

Logistic Regression, like Linear Regression model, computes a weighted sum of the input features plus a bias, but instead of outputting the continuous value, it returns the logistic of this result using the sigmoid function.

---

### 🧪 | Formulation

We already said that logistic regression is like linear regression, for this particular model the task we are trying to solve for classification, so from the output of linear regression prediction equation we need to put does values in a way such that the final output is a probability ranging from 0 to 1 and then determine a treshold to return a final categorical output (0 or 1).

We achieve this by passing the linear predictions to a sigmoid function.

**Sigmoid Function**

Sigmoid function will output a value between 0 and 1, this is how the equation looks like:

$$
\sigma(t) = \frac{1}{1 + e^{-t}}
$$

### 📉 | Calculating error

Because Logistic Regression outputs a probability, we need to model a cost function such that penalizes for values far from the actual target (0 or 1) and optimize for probability values near the target. We achieve this using a log loss function also known as cross-entropy.

$$
H(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)
$$


### 📌 | Log Loss (Cross-Entropy Cost/Error Function)

Log loss, also known as cross-entropy loss, is a commonly used loss function in binary and multiclass classification problems. It measures the performance of a classification model where the prediction output is a probability value between 0 and 1.

#### Formula:

For a binary classification problem, the log loss is calculated using the following formula:

$$
\text{Log Loss} = - \frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right) 
$$

Where:
- $N$ is the number of samples.
- $y_i$ is the actual label of the $i$-th sample (0 or 1).
- $p_i$ is the predicted probability that the sample belongs to the positive class.

For multiclass classification, the formula is modified accordingly.

#### Interpretation:

- Log loss penalizes incorrect classifications more severely when the model is confident about the prediction.
- It is sensitive to the difference between the predicted probability and the actual label.

#### Application:

- Log loss is commonly used as the evaluation metric and optimization objective in logistic regression and neural networks for classification tasks.
- Minimizing log loss during training helps improve the model's predictive accuracy.

#### Properties:

- Log loss is non-negative.
- It approaches 0 when the predicted probabilities are close to the actual labels.

Certainly! Here's an explanation of the process of optimizing logistic regression using the log loss formula, formatted in Markdown:


### ⚡️ | Optimizing Logistic Regression using Log Loss

Logistic regression is a popular classification algorithm that uses a logistic function to model the probability of a binary outcome. The optimization process involves minimizing the log loss (also known as cross-entropy loss) to improve the model's predictive performance.

#### Optimization Process:

1. **Initialization:**
   - Initialize the model parameters (coefficients) with small random values or zeros.

2. **Forward Propagation:**
   - Compute the logits (linear combination of features and coefficients) for each sample:
$$z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n$$
   - Apply the logistic (sigmoid) function to the logits to obtain the predicted probabilities:
$$p = \frac{1}{1 + e^{-z}} $$

3. **Compute Log Loss:**
   - Calculate the log loss for the predicted probabilities using the actual labels:
$$\text{Log Loss} = - \frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right)$$

4. **Backpropagation:**
   - Compute the gradient of the log loss with respect to each model parameter using chain rule:
$$\frac{\partial \text{Log Loss}}{\partial \theta_j} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i) x_{ij} $$
   - Update the model parameters using a gradient descent optimization algorithm:
$$\theta_j := \theta_j - \alpha \frac{\partial \text{Log Loss}}{\partial \theta_j}$$
   - Where \( \alpha \) is the learning rate controlling the step size of parameter updates.

5. **Repeat:**
   - Iterate steps 2-4 until convergence criteria are met, such as reaching a predefined number of iterations or achieving a sufficiently low log loss.

#### Evaluation:

- After optimization, evaluate the trained logistic regression model on a separate validation or test dataset using appropriate performance metrics such as accuracy, precision, recall, and F1-score.
- Adjust hyperparameters (e.g., learning rate, regularization strength) as needed to improve model performance and prevent overfitting.

#### Conclusion:

Optimizing logistic regression using the log loss formula involves iteratively adjusting model parameters to minimize the discrepancy between predicted probabilities and actual labels. This process helps in building a robust classifier for binary classification tasks.

