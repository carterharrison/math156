### Losses
import numpy as np
import math

class MSE:
    def loss(self, predictions, y):
        l = np.mean((y - predictions)**2)
        return l
    
    def loss_gradient(self, predictions, y):
        grad = -2*(y - predictions)/y.shape[1]
        return grad.T
    

class Cross_Entropy:
    def loss(self, predictions, y):
        # We add 1e-6 to the argument of the logarithm for numercal stability
        predictions_log = np.log(predictions + 1e-6)
        l = (-1 / (y.shape[1])) * np.sum(y*predictions_log)
       
        return l
        
    def loss_gradient(self, predictions, y):
        # We add 1e-6 for consistently with above
        grad = ((-1) * y/(predictions + 1e-6)) / (y.shape[1])

        return grad.T
