import numpy as np

class SlidingWindow:
    def __init__(self, X: np.ndarray, reset_cycle= 37):
        self.X = X
        self.it = 1
        self.init_seq_size = X.shape[1]
        self.values = X[0]
        self.reset_cycle = reset_cycle
        
    def next(self, pred):
        if self.it == len(self.X):
            return 0
        new_values = self.X[self.it, :].copy()
        if self.it % self.reset_cycle: # Not the start of the cycle
            if self.init_seq_size > 0:
                self.init_seq_size -= 1
                new_values[self.init_seq_size:-1, -1] = self.values[self.init_seq_size+1:, -1]
            else:
                new_values[:-1, -1] = self.values[1:, -1]
            new_values[-1, -1] = pred
            self.values = new_values
        else: # Start of the cycle
            # Reinit the values
            self.init_seq_size = self.X.shape[1]
            self.values = self.X[self.it]        
        self.it += 1
        return self.it
    
    def __str__(self):
        return str(("iterator: " + str(self.it-1), "init_seq_size: " + str(self.init_seq_size), self.values))
    
    def __repr__(self):
        return str(self)
