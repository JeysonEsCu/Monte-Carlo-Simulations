import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Slider, RadioButtons

# -----------------------------
# Monte Carlo Integration Function
# -----------------------------
def function1(x):
    return np.cos(np.log(x)) / (x**2 + 1)
# result = (np.pi / 4) * (1 / np.cosh(np.pi / 2))

def f2(x):
    return np.exp(-x**2)

# Dictionary of functions for easy access, Latex labels, integration limits
functions_dict = {
    "f(x) = cos(ln(x)) / (x^2 + 1)": { "func": function1, 
        "label_eq": r"$f(x) = \frac{\cos(\ln(x))}{x^2 + 1}$", 
        "a": 0.01, "b": 1.0, "exact": (np.pi / 4) * (1 / np.cosh(np.pi / 2))},
    "f(x) = exp(-x^2)": { "func": f2, 
        "label_eq": r"$f(x) = e^{-x^2}$", "a": 0.0, "b": 2.0, 
        "exact": 0.8820813907624217}  # approx value of integral from 0 to 2
}


def monte_carlo_integral_function(f: callable, N: int, a: float, b: float, ax: plt.Axes, label_eq: str, exact=None) -> None:
    """
    Approximate the integral of a given function f(x) over [a,b] using Monte Carlo.

    Parameters:
    - f: callable function f(x)
    - N: number of random points
    - a, b: integration interval
    - ax: matplotlib Axes to plot
    """
    # Generate random points
    x_random = np.random.uniform(a, b, N)
    y_random = f(x_random)

    # Monte Carlo estimate
    integral_estimate = (b - a) * np.mean(y_random)

    # Plot
    ax.clear()
    xs = np.linspace(a, b, 500)
    ax.plot(xs, f(xs), 'r', label=label_eq)
    ax.scatter(x_random, y_random, color='blue', s=5, label='Random points')
    
    title = f"Monte Carlo estimate: {integral_estimate:.5f} (N={N})"
    if exact is not None:
        error_pct = abs(integral_estimate - exact)/abs(exact) * 100
        title += f"\nExact value: {exact:.5f}, Error: {error_pct:.2f}%"
    ax.set_title(title)
    ax.legend()
    plt.draw()

# Slider update function

def update(val):
    N = int(slider_N.val)
    key = radio.value_selected
    func_info = functions_dict[key]
    monte_carlo_integral_function(func_info["func"], N=N, a=func_info["a"], b=func_info["b"], 
                    ax=ax, label_eq=func_info["label_eq"], exact=func_info["exact"])

# -----------------------------
# Main program
# -----------------------------

def main():
    global fig, ax, slider_N, radio

    fig, ax = plt.subplots(figsize=(7,5))
    plt.subplots_adjust(bottom=0.3, left=0.25)  # space for slider and radio

    # Slider for number of points N
    ax_slider_N = plt.axes([0.25, 0.2, 0.5, 0.03])
    slider_N = Slider(ax_slider_N, "Points N", 100, 5000, valinit=500, valstep=100)
    slider_N.on_changed(update)

    # RadioButtons for selecting function
    ax_radio = plt.axes([0.25, 0.05, 0.5, 0.15])
    radio = RadioButtons(ax_radio, list(functions_dict.keys()))
    radio.on_clicked(update)

    # Initial plot
    update(None)
    plt.show()

if __name__ == "__main__":
    main()
