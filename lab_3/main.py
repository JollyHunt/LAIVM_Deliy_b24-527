import numpy as np
from qubit import Qubit
from gates import X, Y, Z, apply_gate, cnot
from visualization import plot_bloch_sphere

def main():
    print("Для |1> -- [0  1]")
    for gate, name in [(X, "X"), (Y, "Y"), (Z, "Z")]:
        q = Qubit(state=np.array([0, 1], dtype=complex))
        apply_gate(q, gate)
        print(f"После гейта {name}:", q.state)
        plot_bloch_sphere(q, f"Гейт {name} для 1")

    print("\n\nДля |0> -- [1  0]")
    for gate, name in [(X, "X"), (Y, "Y"), (Z, "Z")]:
        q = Qubit(state=np.array([1, 0], dtype=complex))
        print(q.state)
        apply_gate(q, gate)
        print(f"После гейта {name}:", q.state)
        plot_bloch_sphere(q, f"Гейт {name} для 0")

    print("\n\nДля |0> -- [1/2  1/2]")
    for gate, name in [(X, "X"), (Y, "Y"), (Z, "Z")]:
        q = Qubit(state=np.array([0.5, 0.5], dtype=complex))
        print(q.state)
        apply_gate(q, gate)
        print(f"После гейта {name}:", q.state)
        plot_bloch_sphere(q, f"Гейт {name} для 0_5")
    
    control = Qubit(np.array([0, 1], dtype=complex))
    target = Qubit()
    print("\nДо CNOT: Target =", target.state)
    cnot(control, target)
    print("После CNOT: Target =", target.state)

    control = Qubit(np.array([1, 0], dtype=complex))
    target = Qubit()
    print("\nДо CNOT: Target =", target.state)
    cnot(control, target)
    print("После CNOT: Target =", target.state)

if __name__ == "__main__":
    main()
