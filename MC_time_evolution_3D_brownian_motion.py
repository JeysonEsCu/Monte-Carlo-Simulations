import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def brownian_motion_3D(number_steps: int, delta: float) -> tuple:
    """
    Generates 3D random steps of Brownian Motion
    """
    x = np.zeros(number_steps)
    y = np.zeros(number_steps)
    z = np.zeros(number_steps)
    for i in range(1, number_steps):
        x[i] = x[i-1] + np.random.normal(0, delta)
        y[i] = y[i-1] + np.random.normal(0, delta)
        z[i] = z[i-1] + np.random.normal(0, delta)
    return x, y, z

def animate_trajectory(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> FuncAnimation:
    """
    Creates an animation of the trajectory step by step, stopping at the end
    """
    # line for the trajectory
    line = ax.plot([], [], [], 'b-', lw=1)[0]
    # Starting point (0,0,0) in green
    start_point = ax.scatter(x[0], y[0], z[0], color='green', s=50)
    # Coordinates of the ending point will be updated in the animation
    end_point = ax.scatter([], [], [], color='red', s=50)

    def update(frame: int) -> tuple:
        # Updates the line from the start up to the current frame (current step)
        line.set_data(x[ : frame + 1], y[ : frame + 1])
        # Updates the z data for 3D line
        line.set_3d_properties(z[ : frame + 1])
        # Updates the end point
        end_point._offsets3d = (x[frame : frame + 1], y[frame : frame + 1], z[frame : frame + 1])
        return line, end_point

    ani = FuncAnimation(
        fig, update, frames=len(x), interval=50, blit=False, repeat=False
    )
    return ani

# Parameters
number_steps = 200
delta = 0.1

# Generate trajectory
x, y, z = brownian_motion_3D(number_steps, delta)

# Create figure and 3D axis
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Adjust limits to show the entire trajectory
margin = 0.1
ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
ax.set_ylim(np.min(y) - margin, np.max(y) + margin)
ax.set_zlim(np.min(z) - margin, np.max(z) + margin)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Brownian Motion 3D - Trajectory Animation")

# Animation
ani = animate_trajectory(ax, x, y, z)

plt.show()

