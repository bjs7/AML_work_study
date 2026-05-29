import numpy as np



from federated_learning.registry import register_reg_model


# So this should just be for parties. I should make one just for manager, where weights can be initialized
@register_reg_model('regression')
class LogReg:

    # right now we just have gradient descent, but what about stochastic?
    # though stochastic is in the overall data?

    # make a class/method for initial parameters for all algos?
    def __init__(self, eta = 0.01):
        pass
        self.current_w = None
        self.new_w = None
        self.eta = eta

    def initial_parameters(self):
        self.current_w = [1]*13

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
    
    def predict(self, X):
        z = X @ self.current_w
        return self.sigmoid(z)
    
    # this could be made a bass for all parties / models I suppose
    # though could also be kept outside of a class. So it is not defined over and over?
    # Not sure if the saved memory from this would be worth it?
    def predict_binary(self, X, threshold = 0.5):
        return (self.predict(X) >= threshold).astype(int)
    
    def f_loss(self, X, y):
        m = X.shape[0]
        y_hat = self.predict(X)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        loss = -1/m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    def q_values(self, X, y):
        m = X.shape[0]
        X_transposed = self.X.T
        self.Q_value = 1/m * X_transposed @ X
        self.q_value = -2/m * X_transposed @ y
    
    def calculate_gradient(self, X, y):
        #gradient = eta * (2 * self.Q_value @ self.current_w + self.q_value)
        m = X.shape[0]
        y_hat = self.predict(X)
        gradient = 1/m * X.T @ (y_hat - y) #
        return gradient

    def update_w(self, X, y):
        self.current_w = self.current_w - self.eta * self.calculate_gradient(X, y)

