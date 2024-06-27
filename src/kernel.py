import numpy as np

class kernel:
    def __init__(self, gamma = 1):
        self.gamma = gamma
    
    def expk(self, x, y):
        return np.exp(- self.gamma * (np.linalg.norm(x-y)**2))