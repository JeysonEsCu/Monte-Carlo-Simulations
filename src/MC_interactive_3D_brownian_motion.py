import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", ensures interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def brownian_motion_3D(number_steps: int, delta: float) -> tuple:
    x = np.zeros(number_steps)
    y = np.zeros(number_steps)
    z = np.zeros(number_steps)
    
    #next steps
    for i in range(1, number_steps):
        x[i] = x[i-1] + np.random.normal(0, delta)
        y[i] = y[i-1] + np.random.normal(0, delta)
        z[i] = z[i-1] + np.random.normal(0, delta)
        
    return x, y, z

def draw_particles(ax: plt.Axes, cmap: plt.cm.ScalarMappable, number_particles: int, 
                   number_steps: int, delta: float) -> None:
    """
    Draws or refreshes particle trajectories
    """
    # Clear previous plots
    ax.cla()
    
    for i in range(number_particles):
        x, y, z = brownian_motion_3D(number_steps, delta)
        # Assign color based on particle index from the colormap list
        color = cmap(i)
        
        ax.plot(x, y, z, color=color, linewidth=1, label=f'Particle {i+1}')
        ax.scatter(x[0], y[0], z[0], color='green', s=50)  # start
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=50)  # end

    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Brownian Motion 3D")
    ax.legend()
    ax.figure.canvas.draw_idle()  # force update

def main():
    number_steps = 100
    delta = 0.1
    number_particles = 5
    # Array of colors
    cmap = plt.get_cmap('viridis', number_particles)

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the first time
    draw_particles(ax, cmap, number_particles, number_steps, delta)

    # Refresh button
    ax_button = plt.axes([0.2, 0.80, 0.15, 0.05]) # x, y, width, height
    button = Button(ax_button, 'Refresh')
    button.on_clicked(lambda event: draw_particles(ax, cmap, number_particles, number_steps, delta))

    plt.show()

if __name__ == "__main__":
    main()
