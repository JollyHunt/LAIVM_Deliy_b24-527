import numpy as np

class Qubit:
    def __init__(self, state=None):
        self.state = np.array([1, 0], dtype=complex) if state is None else state
    
    def measure(self):
        prob = np.abs(self.state[1])**2
        return 1 if np.random.random() < prob else 0
    
    def __str__(self):
        return f"Qubit state: {self.state}"
