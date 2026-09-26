\# Machine Learning



\## Supervised Learning



Supervised learning learns a mapping from input features to known target values. Classification predicts discrete labels, while regression predicts continuous values.



A training dataset is used to learn model parameters. A validation dataset can be used for model selection and hyperparameter tuning. A test dataset should be reserved for final evaluation.



\## Overfitting



Overfitting occurs when a model learns patterns that are too specific to the training data and performs poorly on unseen examples.



Regularization can reduce overfitting by discouraging overly complex models. Cross-validation provides another way to estimate how a model generalizes.



\## Linear Regression



Linear regression models a continuous target as a weighted combination of input features. The parameters can be estimated by minimizing a loss function such as mean squared error.



Gradient descent is an iterative optimization procedure that updates model parameters in the direction that reduces the objective function.



\## Logistic Regression



Logistic regression is commonly used for binary classification. It computes a weighted combination of features and applies a sigmoid function to obtain a value between zero and one.



A threshold can then be used to convert the predicted probability into a class label.



\## Decision Trees



A decision tree recursively splits data according to feature-based conditions. Internal nodes represent decisions, branches represent outcomes, and leaf nodes represent predictions.



Deep trees can overfit. Maximum depth, minimum sample constraints, and pruning strategies can control model complexity.



\## Neural Networks



A neural network contains layers of interconnected computational units. Each unit applies a weighted transformation followed by an activation function.



Training commonly uses backpropagation to calculate gradients and an optimization algorithm such as stochastic gradient descent to update parameters.



\## Evaluation



Common classification metrics include accuracy, precision, recall, and F1 score. Accuracy measures the fraction of correct predictions, while precision measures how many predicted positives are actually positive.



Recall measures how many relevant positive examples were identified. F1 combines precision and recall through their harmonic mean.

