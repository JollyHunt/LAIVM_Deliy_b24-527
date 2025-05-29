import numpy as np
import matplotlib.pyplot as plt

def plot_bloch_sphere(qubit, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = 2 * np.real(qubit.state[0] * np.conj(qubit.state[1]))
    y = 2 * np.imag(qubit.state[1] * np.conj(qubit.state[0]))
    z = np.abs(qubit.state[0])**2 - np.abs(qubit.state[1])**2
    
    ax.quiver(0, 0, 0, x, y, z, color='r', arrow_length_ratio=0.1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"{filename}")
    plt.close()