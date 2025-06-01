import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax as softmax_scipy
from sklearn.datasets import make_classification

"""Optimizer classes"""


class Normal:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, gradient):
        update = self.learning_rate * gradient

        return update

    def reset(self):
        pass


class Momentum:
    def __init__(self, momentum=0.8, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.change = 0.0

    def update(self, gradient):
        self.change = self.learning_rate * gradient + self.momentum * self.change
        update = self.change

        return update

    def reset(self):
        self.change = 0.0


class RMSProp:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epsilon = 1e-6
        self.rho = 0.90
        self.r = 0.0

    def update(self, gradient):
        self.r = self.rho * self.r + (1 - self.rho) * (gradient**2)
        update = self.learning_rate / \
            (np.sqrt(self.epsilon + (self.r))) * gradient

        return update

    def reset(self):
        self.r = 0.0


class Adagrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epsilon = 1e-7
        self.r = 0.0

    def update(self, gradient):
        self.r += gradient**2
        update = self.learning_rate / \
            (self.epsilon + np.sqrt(self.r)) * gradient

        return update

    def reset(self):
        self.r = 0.0


class Adam:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.rho1 = 0.9
        self.rho2 = 0.999
        self.epsilon = 1e-8
        self.s = 0.0
        self.r = 0.0
        self.t = 0

    def update(self, gradient):
        self.t += 1
        self.s = self.rho1 * self.s + (1 - self.rho1) * gradient
        self.r = self.rho2 * self.r + (1 - self.rho2) * (gradient**2)
        s_hat = self.s / (1 - self.rho1**self.t)
        r_hat = self.r / (1 - self.rho2**self.t)
        update = s_hat * self.learning_rate / (np.sqrt(r_hat) + self.epsilon)

        return update

    def reset(self):
        self.s = 0.0
        self.r = 0.0
        self.t = 0


"""Activation functions"""


class ActivationFunction:
    def __init__(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input

        self.output = self.activation_function(self.input)

        return self.output

    def backward(self, output_gradient):
        return np.multiply(
            output_gradient, self.activation_function_derivative(self.input)
        )


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_derivative)

    @staticmethod
    def sigmoid(x):
        return sigmoid(x)

    @staticmethod
    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))


class Relu(ActivationFunction):
    def __init__(self):
        super().__init__(self.relu, self.relu_derivative)

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x < 0.0, 0.0, 1.0)


class Hyperbolic(ActivationFunction):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_derivative)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2


class Linear(ActivationFunction):
    def __init__(self):
        super().__init__(self.linear, self.linear_derivative)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)


class LeakyReLU(ActivationFunction):
    def __init__(self):
        super().__init__(self.leakyrelu, self.leakyrelu_derivative)

    @staticmethod
    def leakyrelu(x, alpha=0.01):
        return np.maximum(alpha * x, x)

    @staticmethod
    def leakyrelu_derivative(x, alpha=0.01):
        return np.where(x < 0, alpha, 1)


class ELU(ActivationFunction):
    def __init__(self):
        super().__init__(self.elu, self.elu_derivative)

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x <= 0, alpha * (np.exp(x) - 1), x)

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x < 0, alpha * np.exp(x), 1)


class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__(self.softmax, self.softmax_derivative)

    @staticmethod
    def softmax(x):
        return softmax_scipy(x)

    @staticmethod
    def softmax_derivative(x):
        e = np.exp(x - np.max(x))
        s = np.sum(e, axis=1, keepdims=True)
        return e / s


class Layer:
    """
    A single layer of a neural network containing weights and biases, and methods for forward and backward propagation.
    """

    def __init__(
        self,
        input_size,
        output_size,
        optimizer_weights=Normal(learning_rate=0.01),
        optimizer_biases=Normal(learning_rate=0.01),
        regularization=1.0,
        gradient_clipping=False,
    ):
        self.gradient_clipping = gradient_clipping
        self.regularization = regularization
        self.optimizer_weights = optimizer_weights
        self.optimizer_biases = optimizer_biases
        self.weights = np.random.randn(input_size, output_size)
        # 0.01 ensures all neurons have some output which can be backpropagated in the first training cycle.
        self.biases = np.zeros((1, output_size)) + 0.01

    def forward(self, input):
        """Forward pass through the layer."""
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases

        return self.output

    def backward(self, gradient):
        """Backward pass through the layer."""
        weights_gradient = np.dot(self.input.T, gradient)
        biases_gradient = np.mean(gradient, axis=0, keepdims=True)

        update_weights = self.optimizer_weights.update(
            weights_gradient * self.regularization
        )
        update_biases = self.optimizer_biases.update(biases_gradient)

        self.weights -= update_weights
        self.biases -= update_biases

        input_gradient = np.dot(gradient, self.weights.T)

        if self.gradient_clipping:
            input_gradient = np.clip(input_gradient, 1e-16, 1e16)

        return input_gradient


def add_layer_activation(nn, activation_function):
    """Adds an activation function to the neural network."""
    if activation_function == "sigmoid":
        nn.add_activation(Sigmoid())

    elif activation_function == "relu":
        nn.add_activation(Relu())

    elif activation_function == "leakyrelu":
        nn.add_activation(LeakyReLU())

    elif activation_function == "hyperbolic":
        nn.add_activation(Hyperbolic())

    elif activation_function == "linear":
        nn.add_activation(Linear())

    elif activation_function == "softmax":
        nn.add_activation(Softmax())

    elif activation_function == "elu":
        nn.add_activation(ELU())

    else:
        raise ValueError(
            "Activation function not supported. Use 'sigmoid', 'relu', 'leakyrelu', 'hyperbolic', 'linear', 'softmax', or 'elu'.")


def add_optimizer(optimizer, learning_rate):
    """Returns an optimizer based on the specified type."""
    if not optimizer:
        opt = Normal(learning_rate=learning_rate)

    elif optimizer == "momentum":
        opt = Momentum(learning_rate=learning_rate)

    elif optimizer == "adagrad":
        opt = Adagrad(learning_rate=learning_rate)

    elif optimizer == "rmsprop":
        opt = RMSProp(learning_rate=learning_rate)

    elif optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)

    else:
        raise ValueError(
            "Optimizer not supported. Use 'momentum', 'adagrad', 'rmsprop', 'adam', or None.")

    return opt


class NeuralNetwork:
    """A class representing a neural network, containing layers, activations, and methods for training and prediction."""

    def __init__(
        self,
        cost_function,
        cost_function_derivative,
    ):
        self.network = []
        self.activations = []
        self.cost_function = cost_function
        self.cost_function_derivative = cost_function_derivative

    def add_layer(self, layer):
        """Adds a layer to the neural network."""
        self.network.append(layer)

    def add_activation(self, activation):
        """Adds an activation function to the neural network."""
        self.activations.append(activation)

    def train(
        self,
        X,
        y,
        epochs,
        method="gd",
        minibatch_size=10,
        return_all_costs=False,
    ):
        """Trains the neural network on input data X and target y for a specified number of epochs using the specified method."""
        if return_all_costs:
            self.cost_list = []
        else:
            self.cost_list = None

        if method == "gd":
            for i in range(epochs):
                ypred = self.predict(X)

                gradient = self.cost_function_derivative(y, ypred, X)
                self.backpropagation(gradient)

        elif method == "sgd":
            n_minibatches = int(len(X) / minibatch_size)

            for i in range(epochs):
                for batch in range(n_minibatches):
                    k = minibatch_size * np.random.randint(n_minibatches)

                    X_batch = X[k: k + minibatch_size]
                    y_batch = y[k: k + minibatch_size]

                    ypred_batch = self.predict(X_batch)

                    if np.isnan(ypred_batch).any():
                        raise ValueError("The values are nan, e.i. overflow!")

                    if self.cost_list:
                        cost = self.cost_function(
                            y_batch, ypred_batch, X_batch)
                        self.cost_list.append(cost)

                    gradient = self.cost_function_derivative(
                        y_batch, ypred_batch, X_batch
                    )

                    self.backpropagation(gradient)

                for layer in self.network:
                    layer.optimizer_weights.reset()
                    layer.optimizer_biases.reset()
        else:
            raise ValueError("Method not supported. Use 'gd' or 'sgd'.")

    def backpropagation(self, gradient):
        for layer, activation_function in zip(
            reversed(self.network), reversed(self.activations)
        ):
            gradient = activation_function.backward(gradient)
            gradient = layer.backward(gradient)

    def predict(self, input):
        # self.best_cost = np.min(self.cost_list)
        output = input

        for layer, activation_function in zip(self.network, self.activations):
            output = layer.forward(output)
            output = activation_function.forward(output)

        return output

    def classify(self, input):
        output = input

        for layer, activation_function in zip(self.network, self.activations):
            output = layer.forward(output)
            output = activation_function.forward(output)

        return np.where(output >= 0.5, 1, 0)

    def get_cost_list(self):
        return self.cost_list


def build_neural_network(
    input_size,
    output_size,
    n_hidden_layer,
    n_hidden_nodes,
    cost_function,
    cost_function_derivative,
    activation_function="hyperbolic",
    optimizer=None,
    last_activation="linear",
    learning_rate=0.01,
    regularization=1.0,
    gradient_clipping=False,
):
    """
    Builds a neural network with the specified parameters.
    Parameters:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        n_hidden_layer (int): Number of hidden layers.
        n_hidden_nodes (int): Number of nodes in each hidden layer.
        cost_function (function): Function to compute the cost.
        cost_function_derivative (function): Function to compute the derivative of the cost.
        activation_function (str): Activation function for hidden layers.
        optimizer (str): Optimizer to use for weight updates.
        last_activation (str): Activation function for the output layer.
        learning_rate (float): Learning rate for the optimizer.
        regularization (float): Regularization strength.
        gradient_clipping (bool): Whether to apply gradient clipping.
    Returns:
        NeuralNetwork: An instance of the NeuralNetwork class.
    """
    if n_hidden_layer == 1:
        n_hidden_nodes = input_size
    elif n_hidden_layer < 1:
        raise ValueError("Number of hidden layers must be at least 1.")

    nn = NeuralNetwork(
        cost_function,
        cost_function_derivative,
    )

    if n_hidden_layer > 1:
        nn.add_layer(
            Layer(
                input_size,
                n_hidden_nodes,
                optimizer_weights=add_optimizer(optimizer, learning_rate),
                optimizer_biases=add_optimizer(optimizer, learning_rate),
                regularization=regularization,
                gradient_clipping=gradient_clipping,
            )
        )
        add_layer_activation(nn, activation_function)

    else:
        for i in range(n_hidden_layer - 2):
            nn.add_layer(
                Layer(
                    n_hidden_nodes,
                    n_hidden_nodes,
                    optimizer_weights=add_optimizer(optimizer, learning_rate),
                    optimizer_biases=add_optimizer(optimizer, learning_rate),
                    regularization=regularization,
                    gradient_clipping=gradient_clipping,
                )
            )
            add_layer_activation(nn, activation_function)

    nn.add_layer(
        Layer(
            n_hidden_nodes,
            output_size,
            optimizer_weights=add_optimizer(optimizer, learning_rate),
            optimizer_biases=add_optimizer(optimizer, learning_rate),
            regularization=regularization,
            gradient_clipping=gradient_clipping,
        )
    )

    add_layer_activation(nn, last_activation)

    return nn


def regression_example():
    # Using a simple mean squared error cost function for regression
    def mse(y, y_pred, X): 
        return np.mean((y - y_pred) ** 2)

    def mse_derivative(y, y_pred, X): 
        return 2 * (y_pred - y) / y.shape[0]

    # Example data generation
    X = np.random.rand(1000, 1)  # Features between 0 and 1
    y = 3 * X.squeeze() + np.random.randn(1000) * 2  # Add some noise
    y = y.reshape(-1, 1)  # Reshape to ensure y is a column vector

    # Example usage
    nn = build_neural_network(
        input_size=1,
        output_size=1,
        n_hidden_layer=1,
        n_hidden_nodes=20,
        cost_function=mse,
        cost_function_derivative=mse_derivative,
        activation_function="relu",
        optimizer="adam",
        last_activation="linear",
        learning_rate=0.1,
        regularization=1.0,
        gradient_clipping=True,
    )

    nn.train(X, y, epochs=100, method="gd", return_all_costs=True)
    predictions = nn.predict(X)

    plt.scatter(X, y, label='Data', alpha=0.5)
    plt.plot(X, predictions, color='red', label='Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def classification_example():
    # Using cross-entropy loss for binary classification
    def cross_entropy(y, y_pred, X):
        return -np.mean(y * np.log(y_pred + 1e-15))

    def cross_entropy_derivative(y, y_pred, X):
        return y_pred - y

    # Example data generation
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                               n_redundant=0, n_classes=2, random_state=42)
    y = y.reshape(-1, 1)

    # Example usage
    nn = build_neural_network(
        input_size=2,
        output_size=1,
        n_hidden_layer=2,
        n_hidden_nodes=20,
        cost_function=cross_entropy,
        cost_function_derivative=cross_entropy_derivative,
        activation_function="relu",
        optimizer="adam",
        last_activation="sigmoid",
        learning_rate=0.01,
        regularization=1.0,
        gradient_clipping=True,
    )

    nn.train(X, y, epochs=100, method="gd", return_all_costs=True)
    predictions = nn.classify(X)

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='coolwarm', alpha=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('True Labels')

    plt.subplot(1, 2, 2)

    plt.scatter(X[:, 0], X[:, 1], c=predictions.squeeze(),
                cmap='coolwarm', alpha=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Classification Results')
    plt.show()


if __name__ == "__main__":
    regression_example()
    classification_example()
