import numpy as np

def logistic(z):
    '''
    This takes a value z and returns 1 / (1 + exp(z))
    This works on vectors and applies the operation to every value in the vector
    this is used for classifying with logistic regression
    '''
    return 1 / (1 + np.exp(z))

class log_regression:
    def __init__(self, num_iterations, l2_reg, learning_rate):
        '''
        This is the constructor, it sets the relevant constants for training
        '''
        self.num_iterations = num_iterations
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.weights = None
        
    def train(self, X, Y):
        '''
        This trains the weights according to some reatures X and classes Y
        '''
        # First the weights are initialized randomly
        self.weights = np.random.rand(X.shape[1])
        for _ in range(self.num_iterations):
            # At every step we calculate the gradient and update the weights
            p_1 = np.ones(X.shape[0]) - logistic(np.dot(X,self.weights))
            diff = Y - p_1
            grad = np.dot(X.T,diff) - (self.l2_reg * self.weights)
            self.weights += self.learning_rate * grad
            
    def predict(self,X):
        '''
        This takes some featueres X and predicts 1 or 0 depending on the weights
        we have learned from the training data
        '''
        return (np.exp(np.dot(X,self.weights)) >= 1).astype(int)