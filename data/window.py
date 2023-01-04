import numpy as np

class SlidingWindow:
    def __init__(self, X: np.ndarray):
        self.X = X
        self.it = 0
        self.init_seq_size = X.shape[1]
        self.values = X[0]
        
    def next(self, pred):
        if self.it + 1 == len(self.X):
            return 0
        self.it += 1
        new_values = np.copy(self.X[self.it, :])
        if self.init_seq_size > 1:
            self.init_seq_size -= 1
            new_values[self.init_seq_size-1:-1, -1] = self.values[self.init_seq_size:, -1]
        else:
            new_values[:-1, -1] = self.values[1:, -1]
        new_values[-1, -1] = pred
        self.values = new_values
        return self.it
    
    def __str__(self):
        return str(("iterator: " + str(self.it), "init_seq_size: " + str(self.init_seq_size), self.values))
    
    def __repr__(self):
        return str(self)
