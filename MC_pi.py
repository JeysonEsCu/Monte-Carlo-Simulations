import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def generate_points(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random points within the unit square.
    - param n_points: Number of random points to generate
    - return: Tuple of arrays (x, y) with random points
    """
    x = np.random.rand(n_points)
    y = np.random.rand(n_points)
    return x, y

def classify_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return boolean arrays of points inside and outside the circle.
    - param x: Array of x coordinates
    - param y: Array of y coordinates
    - return: Tuple of boolean arrays (inside, outside)
    """
    inside = x**2 + y**2 <= 1
    outside = np.logical_not(inside)
    return inside, outside

def calculate_pi(inside: np.ndarray) -> float:
    """
    Calculate the estimation of pi using Monte Carlo method.
    - param inside: Boolean array indicating points inside the circle
    """
    return 4 * np.sum(inside) / len(inside)

def relative_error(estimated_pi: float) -> float:
    """
    - param estimated_pi: Estimated value of pi
    - return: Relative error compared to numpy's pi in %
    """
    return abs(np.pi - estimated_pi) / np.pi * 100

def update_plot(value: float, ax: plt.Axes) -> None:
    """
    Function that updates the plot according to the slider value.
    - param value: Number of random points to use for the estimation
    - param ax: Matplotlib Axes object to update the plot on
    """
    n_points = int(value)
    x, y = generate_points(n_points)
    inside, outside = classify_points(x, y)
    pi_estimate = calculate_pi(inside)
    error = relative_error(pi_estimate)

    ax.clear()
    # plot points (x,y) of position where the array inside is True
    ax.scatter(x[inside], y[inside], color='blue', s=1, label='Inside')
    # plot points (x,y) of position where the array outside is True
    ax.scatter(x[outside], y[outside], color='red', s=1, label='Outside')
    
    ax.set_title(f'Estimate of pi: {pi_estimate:.5f} with {n_points} points'
                 f'\nRelative error: {error:.2f}%')
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend()
    # position
    ax.legend(loc='upper right')
    plt.draw()

def main():
    """Main configuration for the plot and slider."""
    fig, ax = plt.subplots(figsize=(5,5))
    plt.subplots_adjust(bottom=0.25)
    # Position of the slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Points', 10, 50000, valinit=10, valstep=100)
    
    slider.on_changed(lambda val: update_plot(val, ax))
    update_plot(slider.val, ax)
    
    plt.show()

if __name__ == "__main__":
    main()

