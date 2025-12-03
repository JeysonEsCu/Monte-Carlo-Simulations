import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Callable

# ---------------------------------------------------------
# Objective function
# ---------------------------------------------------------
def function_to_minimize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(x) + 0.1 * x**2 + np.cos(y) + 0.1 * y**2


def create_figure() -> Tuple:
    fig = plt.figure(figsize=(13, 6))
    plt.subplots_adjust(bottom=0.25, left=0.07, right=0.95)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.2, 1.0])
    ax2d = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    return fig, ax2d, ax3d

# ---------------------------------------------------------
# Create 3D surface
# ---------------------------------------------------------
def plot_3d_surface(ax3d: plt.Axes, func: Callable, grid_range: Tuple = (-10, 10), 
                    grid_points: int = 120) -> Tuple:
    # range to be adjusted as needed
    x = np.linspace(grid_range[0], grid_range[1], grid_points)
    y = np.linspace(grid_range[0], grid_range[1], grid_points)
    
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    # Smooth surface
    ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85,
                      linewidth=0, edgecolor='none', antialiased=True)

    ax3d.set_title("3D Surface of f(x,y)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("f(x,y)")
    ax3d.view_init(elev=35, azim=45)
    ax3d.grid(True)
    return X, Y, Z

# ---------------------------------------------------------
# Setup 2D panel with heatmap and dynamic elements
# ---------------------------------------------------------
def setup_2d_panel(ax2d: plt.Axes, Z: np.ndarray) -> Tuple:
    heatmap = ax2d.imshow(
        Z, extent=[-10, 10, -10, 10], origin="lower",
        cmap="viridis", aspect="auto"
    )

    # Colorbar
    cax = ax2d.figure.add_axes([0.51, 0.25, 0.015, 0.60])
    cbar = ax2d.figure.colorbar(heatmap, cax=cax)
    cbar.set_label("f(x,y)")

    # Dynamic elements
    samples_scatter = ax2d.scatter([], [], s=20, c="white", alpha=0.6)
    best_point = ax2d.scatter([], [], s=80, color="red")
    best_text = ax2d.text(0, 0, "", color="red", fontsize=12)

    ax2d.legend(["Samples", "Best point"], loc="lower left")
    ax2d.set_title("Monte Carlo Minimization (2D)")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")

    return samples_scatter, best_point, best_text

# ---------------------------------------------------------
# Plot real minimum lines in 3D
# ---------------------------------------------------------
def plot_3d_min_lines(ax3d: plt.Axes, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    x_min, y_min, z_min = X[min_idx], Y[min_idx], Z[min_idx]
    # Guide lines from the true minimum to planes
    #ax3d.plot([x_min, x_min], [y_min, y_min], [0, z_min], 'r--')
    #ax3d.plot([x_min, x_min], [0, y_min], [z_min, z_min], 'r--')
    #ax3d.plot([0, x_min], [y_min, y_min], [z_min, z_min], 'r--')

# ---------------------------------------------------------
# Slider with dynamic 3D marker for current minimum
# ---------------------------------------------------------
def create_slider_with_3d_marker(fig: plt.Figure, ax2d: plt.Axes, ax3d: plt.Axes, func: Callable, 
                                 X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                                 samples_scatter: plt.collections.PathCollection,
                                 best_point: plt.collections.PathCollection, best_text: plt.Text
                                 ) -> Slider:
    # Slider
    slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.05])
    slider = Slider(slider_ax, "Samples", 100, 10000, valinit=100, valstep=100)

    # Marker for the current minimum on 3D surface
    min_marker = ax3d.plot([0], [0], [0], 'ro', markersize=6)[0] # first element of the list

    def update(val: float) -> None:
        N = int(slider.val)
        xs = np.random.uniform(-10, 10, N)
        ys = np.random.uniform(-10, 10, N)
        zs = func(xs, ys)

        # Update 2D scatter
        # np.c_[] combine xs and ys (column by column) into Nx2 array
        samples_scatter.set_offsets(np.c_[xs, ys])

        # Current minimum
        idx = np.argmin(zs)
        bx, by, bz = xs[idx], ys[idx], zs[idx]
        best_point.set_offsets([[bx, by]])
        best_text.set_position((bx, by + 0.4))
        best_text.set_text(f"min = {bz:.3f}")

        # Update 3D marker
        min_marker.set_data([bx], [by])
        min_marker.set_3d_properties([bz])

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(None)
    return slider

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    fig, ax2d, ax3d = create_figure()
    X, Y, Z = plot_3d_surface(ax3d, function_to_minimize)
    plot_3d_min_lines(ax3d, X, Y, Z)  # true minimum guide lines
    samples_scatter, best_point, best_text = setup_2d_panel(ax2d, Z)
    slider = create_slider_with_3d_marker(fig, ax2d, ax3d, function_to_minimize, X, Y, Z,
                                          samples_scatter, best_point, best_text)
    plt.show()

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
