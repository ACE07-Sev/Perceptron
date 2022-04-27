import random

import matplotlib.pyplot as plt
import numpy as np

# initializing the random generator
rng = np.random.default_rng(seed=10)

# additional parameters - size, standard deviation
std = 0.78
# train-test split
num_train_points = 320
num_test_points = 80

# Generating the training data (normal distribution)
train_data = np.c_[
    np.r_[rng.normal(2, std, (num_train_points, 2)), rng.normal(4, std, (num_train_points, 2))],
    np.r_[np.zeros((num_train_points, 1)), np.ones((num_train_points, 1))],
]

X_train = train_data[:, :2]
y_train = train_data[:, 2]

# Plotting
fig1, ax = plt.subplots()
ax.set(title="Training Dataset", xlabel="Mary's Ratings", ylabel="John's Ratings")
scatter = ax.scatter(train_data[:, 0], train_data[:, 1], c=train_data[:, 2], edgecolors='black')
ax.legend(handles=scatter.legend_elements()[0], labels=["No", "Yes"])
plt.show()

# Generating the test data (normal distribution)
test_data = np.c_[
    np.r_[rng.normal(2, std, (num_test_points, 2)), rng.normal(4, std, (num_test_points, 2))],
    np.r_[np.zeros((num_test_points, 1)), np.ones((num_test_points, 1))],
]

X_test = test_data[:, :2]
y_test = test_data[:, 2]

# Plotting the dataset scatter plots
fig1, ax = plt.subplots()
ax.set(title="Test Dataset", xlabel="Mary's Ratings", ylabel="John's Ratings")
ax.scatter(test_data[:, 0], test_data[:, 1], c=test_data[:, 2], edgecolors='black')
ax.legend(handles=scatter.legend_elements()[0], labels=["No", "Yes"])
plt.show()


class Perceptron:
    # Constructor
    # properties : learning rate, theta, bias term, loss function history, accuracy history
    def __init__(self, lr=0.01, bias=-1, epoch=500):
        self.lr = lr
        self.theta = None
        self.bias = bias
        self.loss_list = []
        self.accuracy_list = []
        self.epoch = epoch

    # function to update the cost on each iteration
    def update_loss_list(self, y, y_pred):
        loss = np.sum((y - y_pred) ** 2)
        # truncate value to 5 decimal places
        loss = float(np.format_float_positional(loss, precision=5))
        self.loss_list.append(loss)
        return loss

    # function to update the accuracy of the model on each iteration
    def update_accuracy_list(self, y, y_pred):
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        equal_count = np.count_nonzero(y == y_pred)
        accuracy = equal_count * 100.00 / len(y)
        self.accuracy_list.append(accuracy)

    # function to update the weight on each iteration
    def update_weight_bias_SGD(self, x_value, y_value, pred_output):
        # update term for bias
        gradient_bias_comp = 2 * (pred_output - y_value) * pred_output * (1 - pred_output)
        # update term for weights
        gradient_weight_comp = x_value * gradient_bias_comp
        # updating the terms of the model on each iteration
        self.theta -= self.lr * gradient_weight_comp
        self.bias -= self.lr * gradient_bias_comp

    def update_weight_bias_BGD(self, x_vector, y_vector, pred_output):
        sum_w = 0
        sum_b = 0
        for i in range(len(x_vector)):
            # update term for bias
            gradient_bias_comp = 2 * (pred_output[i] - y_vector[i]) * pred_output[i] * (1 - pred_output[i])
            sum_b += gradient_bias_comp
            # update term for weights
            gradient_weight_comp = x_vector[i] * gradient_bias_comp
            sum_w += gradient_weight_comp
        self.theta -= self.lr * sum_w
        self.bias -= self.lr * sum_b
# update term for bias
        # gradient_bias_comp = np.sum(2 * (pred_output - y_vector) * pred_output * (1 - pred_output))
        # update term for weights
        # np.sum(((2 * (pred_output - y_vector) * pred_output * (1 - pred_output)).reshape(-1, 1) * x_vector), axis=0)
        # gradient_weight_comp = np.sum(gradient_bias_comp.reshape(-1, 1) * x_vector, axis=0)
        # gradient_weight_comp = np.sum(x_vector * gradient_bias_comp, axis=0)
        # updating the terms of the model on each iteration

    # BGD function
    def BGD(self, X,y, theta=None, batch_size=20):
        self.epoch = 500
        while self.epoch > 0:
            self.epoch -= 1
            # creating the shape of input dataset (mxn)
            n_samples, n_features = X.shape
            # instantiating theta to be zero before any iteration is run, else maintain the values from previous
            # iteration
            self.theta = theta if (theta is not None and theta.shape[0] == n_features) else np.zeros(n_features)
            # prediction value
            y_pred = self.predict(X)
            # loss value for initial weights
            loss_old = self.update_loss_list(y, y_pred)
            # accuracy for initial weights
            self.update_accuracy_list(y, y_pred)
            # counter to keep track of convergence
            count = 0
            # running the SGD until the loss doesn't change for 10 iterations
            while count < 10:
                # if the SGD hasn't run through the training point before
                # add the point to the index array
                v = np.dot(X, self.theta) + self.bias
                # checking for activation
                fv = self.activation(v)
                # updating the weights
                self.update_weight_bias_BGD(X, y, fv)
                # prediction value
                y_pred = self.predict(X)
                # add the error for current iteration
                loss_new = self.update_loss_list(y, y_pred)
                # add the accuracy for current iteration
                self.update_accuracy_list(y, y_pred)
                # update the convergence counter if the loss has not changed
                if loss_new == loss_old:
                    count += 1
                # if the loss changes, reset the counter
                else:
                    count = 0
                    loss_old = loss_new
            if count == 10:
                break

    # SGD function
    def SGD(self, X, y, theta=None):
        self.epoch = 500
        while self.epoch > 0:
            self.epoch -=1
            # creating the shape of input dataset (mxn)
            n_samples, n_features = X.shape
            # instantiating theta to be zero before any iteration is run, else maintain the values from previous
            # iteration
            self.theta = theta if (theta is not None and theta.shape[0] == n_features) else np.zeros(n_features)
            # prediction value
            y_pred = self.predict(X)
            # loss value for initial weights
            loss_old = self.update_loss_list(y, y_pred)
            # accuracy for initial weights
            self.update_accuracy_list(y, y_pred)
            # counter to keep track of convergence
            count = 0
            # list to keep track of indexes used for training
            index_arr = []
            # running the SGD until the loss doesn't change for 10 iterations
            while count < 10:
                # index array to keep track of the training points used
                if len(index_arr) == len(X):
                    index_arr = []
                # get a random example index (stochastic GD, hence we need to pick randomly)
                i = random.randint(0, len(X) - 1)
                # check if index has already been used for training or not
                if i in index_arr:
                    continue
                # if the SGD hasn't run through the training point before
                else:
                    # add the point to the index array
                    index_arr.append(i)
                    # Hypothesis
                    v = np.dot(X[i], self.theta) + self.bias
                    # checking for activation
                    fv = self.activation(v)
                    # updating the weights
                    self.update_weight_bias_SGD(X[i], y[i], fv)
                    # prediction value
                    y_pred = self.predict(X)
                    # add the error for current iteration
                    loss_new = self.update_loss_list(y, y_pred)
                    # add the accuracy for current iteration
                    self.update_accuracy_list(y, y_pred)
                    # update the convergence counter if the loss has not changed
                    if loss_new == loss_old:
                        count += 1
                    # if the loss changes, reset the counter
                    else:
                        count = 0
                        loss_old = loss_new
            if count == 10:
                break

    # function for predicting y using x
    def predict(self, X):
        output = np.dot(X, self.theta) + self.bias
        # y_pred = np.array([self.activation_func(v) for v in output])
        y_pred = self.activation(output)
        return y_pred

    # activation function
    def activation(self, v):
        return 1 / (1 + np.exp(-v))


wi = random.getstate()
# randomly initialize weights
theta = np.array([random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)])
# randomly initialize bias
bias = random.randint(-1, 1)

print("The initial weights were:\t\t", theta)
print("\nThe bias used:\t\t\t\t", bias)

# initialize the perceptron model
p = Perceptron(bias=bias)
# train the model
p.BGD(X_train, y_train, theta, batch_size=20)

print(p.theta)
print("bias:", p.bias)

# plot error on each iteration
x_axis = [i for i in range(len(p.loss_list))]

plt.title("Error")
plt.xlabel("Iterations")
plt.ylabel("Sum of Squares Error")
plt.plot(x_axis, p.loss_list)
plt.ylim(min(p.loss_list), max(p.loss_list))
plt.show()

# plot accuracy on each iteration
x_axis = [i for i in range(len(p.accuracy_list))]

plt.title("Accuracy")
plt.xlabel("Iterations")
plt.ylabel("% Accuracy")
plt.plot(x_axis, p.accuracy_list)
plt.ylim(0, 101)
plt.show()

# c and m holders for bias and theta terms to plot the model line
c = - p.bias / p.theta[1]
m = - p.theta[0] / p.theta[1]

x_range = np.array([0, 6])

# perceptron line
decision_line = m * x_range + c
# plotting...
fig1, ax = plt.subplots()
# plot of model
ax.set(title="2D Visuallization", xlabel="Mary's Ratings", ylabel="John's Ratings")
ax.scatter(train_data[:, 0], train_data[:, 1], c=train_data[:, 2])
plt.plot(x_range, decision_line)
ax.legend(handles=scatter.legend_elements()[0], labels=["No", "Yes"])
plt.show()
