import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def apply_gate(qubit, gate):
    qubit.state = np.dot(gate, qubit.state)

def cnot(control, target):
    if np.allclose(control.state, np.array([0, 1])):
        apply_gate(target, X)