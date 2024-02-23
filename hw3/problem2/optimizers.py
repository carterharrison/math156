### Optimizers

import numpy as np
    
class SGDmomentum:
    """
    stochastic gradient descent with momentum
    """
    def __init__(self, beta = 0.9):
        self.beta = beta
        self.param_W = None
        self.param_b = None

    def update(self, grad_W, grad_b):
        if self.param_W is None:
            self.param_W = np.zeros_like(grad_W)
            self.param_b = np.zeros_like(grad_b)
            
        self.param_W = self.beta*self.param_W + (1-self.beta)*grad_W
        self.param_b = self.beta*self.param_b + (1-self.beta)*grad_b
    
class RMSprop:
    """RMSprop optimizer"""
    def __init__(self, beta = 0.90, eps = 1e-7):
        self.eps = eps
        self.beta = beta
        self.param_W = None
        self.param_b = None
        self.cache_W = None
        self.cache_b = None
        
    def update(self, grad_W, grad_b):
        if self.param_W is None:
            self.param_W = np.zeros_like(grad_W)
            self.param_b = np.zeros_like(grad_b)
            self.cache_W =  np.zeros_like(grad_W)
            self.cache_b =  np.zeros_like(grad_b)
            
        self.cache_W = (self.beta * self.cache_W) + ((1-self.beta) * (grad_W**2))
        self.cache_b = (self.beta * self.cache_b) + ((1-self.beta) * (grad_b**2))

        self.param_W = (1/(np.sqrt(self.cache_W) + self.eps)) * grad_W
        self.param_b = (1/(np.sqrt(self.cache_b) + self.eps)) * grad_b
        
        
class Adam:
    """
    Adam optimizer
    """
    
    def __init__(self, beta1 = 0.9, beta2 = 0.999, eps = 1e-7):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.cache_W = None
        self.cache_b = None
        self.cache_WW = None
        self.cache_bb = None
        self.param_W = None
        self.param_b = None
        
    def update(self, grad_W, grad_b):
        if self.cache_W is None:
            self.cache_W = np.zeros_like(grad_W)
            self.cache_WW = np.zeros_like(grad_W)
            self.cache_b = np.zeros_like(grad_b)
            self.cache_bb = np.zeros_like(grad_b)
            
        self.cache_W = self.beta1*self.cache_W + (1-self.beta1)*np.clip(grad_W, -5,5)
        self.cache_b = self.beta1*self.cache_b + (1-self.beta1)*np.clip(grad_b, -5,5)
                
        self.cache_WW = self.beta1*self.cache_WW + (1-self.beta1)*(np.clip(grad_W, -5,5)**2)
        self.cache_bb = self.beta1*self.cache_bb + (1-self.beta1)*(np.clip(grad_b, -5,5)**2)
        
        self.param_W = 1/(np.sqrt(self.cache_WW) + self.eps) * self.cache_W
        self.param_b = 1/(np.sqrt(self.cache_bb) + self.eps) * self.cache_b
        
        