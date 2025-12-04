import numpy as np
import matplotlib.pyplot as plt

def brownian_motion_2D(number_steps: int, delta: float) -> tuple:
    """
    Random motion of a particle in a surface

    Arguments
        delta: step size
    """
    # Array to store the motion
    x = np.zeros(number_steps)
    y = np.zeros(number_steps)

    # Simulation
    for i in range(1, number_steps):
        dx = np.random.normal(0, delta)
        dy = np.random.normal(0, delta)
        # next step position based on the previous one
        x[i] = x[i - 1] + dx
        y[i] = y[i - 1] + dy
    return x, y

def n_particle_motion(number_of_particles: int, number_steps: int, 
                      delta: float)-> None:
    """
    different particles in a surface with a random motion, 
    the interaction between them is not considered.
    All the particles start at the coordinates (0.0, 0.0)

    Arguments
        delta: step size
    """
    # List of colors
    cmap = plt.get_cmap('viridis', number_of_particles)

    plt.figure(figsize=(8, 8))
    for i in range(number_of_particles):
        x, y = brownian_motion_2D(number_steps, delta)
        trajectory_color = cmap(i)

        plt.plot(x, y, color = trajectory_color, linewidth = 1, 
                 label = f'Particle{i + 1}')
        plt.scatter(x[0], y[0], color = 'green')
        plt.scatter(x[-1], y[-1], color = 'blue')
    # This section inside of the loop creates n-different
    # visualization files
    plt.title("Brownian motion in 2D")
    plt.xlabel("position in X")
    plt.ylabel("position in Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def main():
    # Parameters
    number_steps = 100
    delta = 0.1
    number_of_particles = 5

    # plot
    n_particle_motion(number_of_particles, number_steps, delta)

#----------------------------------------------
if __name__ == "__main__":
    main()
