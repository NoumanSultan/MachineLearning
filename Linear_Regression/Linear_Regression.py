"""Linear Regression"""

"""Importing Libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Loading train and test data files**"""

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

"""Converting dataframes to numpy arrays"""

# Convert the 'x' and 'y' columns of each DataFrame to numpy arrays
train_x = train_df['x'].to_numpy()
train_y = train_df['y'].to_numpy()

test_x = test_df['x'].to_numpy()
test_y = test_df['y'].to_numpy()

"""##Training the model
Use normal equation method to compute the optimal parameters for linear regression. Print the values of optimal parameters.

"""

# Initialize values for best theta and minimum cost
best_theta0, best_theta1 = None, None
min_cost = float('inf')

# Iterate over range of theta0 and theta1
for theta0 in np.arange(0.5, 5, 0.2):
    for theta1 in np.arange(0.5, 5, 0.2):
        # Predict values using hypothesis function
        predictions = theta0 + theta1 * train_x

        # Calculate the cost function
        cost = (1 / (2 * len(train_x))) * np.sum((predictions - train_y) ** 2)

        # Update the best theta values if the current cost is lower
        if cost < min_cost:
            min_cost = cost
            best_theta0, best_theta1 = theta0, theta1

print("Optimal theta0:", best_theta0)
print("Optimal theta1:", best_theta1)
print("Minimum cost:", min_cost)

"""##Testing the model
Predict the output (y`) for test dataset using the optimal parameters  computed at the previous step.
"""

#testing the model and predicting answers

# Using the optimal parameters to predict outputs for the test dataset
predicted_test_y = best_theta0 + best_theta1 * test_x
print(predicted_test_y)

"""Compute the error using mean squared error function"""

#compute and display the error

New_cost = 0

# Calculate the cost function for all points
for i in range(len(test_y)):

  New_cost = (1 / (2 * len(test_y))) * np.sum((predicted_test_y[i] - test_y[i]) ** 2)

# Displaying the values

print("Minimum cost:", New_cost)


"""##Optimization
Implementation of the gradient descent algorithm for optimizion of parameters 𝜃0 and 𝜃1
"""

import numpy as np

# Initialize parameters with random values
theta0 = np.random.rand()   # random initialization
theta1 = np.random.rand()   # random initialization
learning_rate = 0.05  # learning rate
num_iterations = 1000  # Number of iterations

# Normalize train_x (if necessary)
train_x = (train_x - np.min(train_x)) / (np.max(train_x) - np.min(train_x))

# Gradient Descent
for iteration in range(num_iterations):
    # Predictions using the current theta values
    predictions = theta0 + theta1 * train_x

    # Calculate the cost (mean squared error)
    cost2 = (1 / (2 * len(train_x))) * np.sum((predictions - train_y) ** 2)

    # Calculate gradients
    gradient_theta0 = (1 / len(train_x)) * np.sum(predictions - train_y)
    gradient_theta1 = (1 / len(train_x)) * np.sum((predictions - train_y) * train_x)

    # Update theta values
    theta0 -= learning_rate * gradient_theta0
    theta1 -= learning_rate * gradient_theta1


# Display the optimal values
print("Optimal theta0:", theta0)
print("Optimal theta1:", theta1)
print("Final cost:", cost2)

"""##Visualization
Plotting the actual and predicted answers after applying Gradient Decent
"""

# Predict outputs for the training dataset
training_predictions = theta0 + theta1 * train_x

# Plotting
plt.scatter(train_x, train_y, color='blue', label='Actual')
plt.plot(train_x, training_predictions, color='red', label='Predicted')
plt.xlabel("X (Input)")
plt.ylabel("Y (Output)")
plt.title("Linear Regression: Predicted vs Actual")
plt.legend()
plt.show()
