import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
#  Simulation parameters
# -----------------------------
# For large half-life, decays are rare, so we can simulate longer times
# For short half-life, decays are frequent, so they behave as a continuous decay
# affecting the accuracy of the simulation. To improve accuracy, we can
# make a dependence of the time step dt with respect to the half-life 
# in the slider update function.

# -----------------------------
# Simulation functions
# -----------------------------

def simulate_decay(N0: int, half_time: float, Time: float, dt: float) -> list:
    """
    Simulate radioactive decay using Monte Carlo.

    Parameters:
    - N0: initial number of nuclei
    - half_time: half-life of the substance 
    - decay_constant: decay constant lambda
    - T: total time
    - dt: time step

    Returns:
    - list of remaining nuclei at each time step
    """
    nuclei = np.ones(N0, dtype=bool)  # True = nucleus active
    decay_constant = np.log(2) / half_time
    # List to store number of remaining nuclei at each time step
    N_t = []
    timesteps = int(Time/dt)

    for _ in range(timesteps):
        # Probability of survival at time dt: N_t/N0
        p = np.exp(-decay_constant*dt)
        # Random numbers, simulate the probability of decay
        r = np.random.rand(N0)
        # probability of decay is larger than probability of survival
        # and there exists nucleus
        decay = (r > p) & nuclei
        # Nuclei disappear (False) at the position in the array 
        # where there a nucleus and decay at that position
        nuclei[decay] = False
        # Store the number of remaining nuclei
        N_t.append(np.sum(nuclei))

    return N_t

def plot_decay(N_t: list, N0: int, half_time: float, dt: float, ax):
    """
    Plot Monte Carlo simulation and theoretical decay curve.
    """
    T = dt * len(N_t) # Total time
    time = np.arange(0, T, dt)
    decay_constant = np.log(2) / half_time
    theoretical = N0 * np.exp(-decay_constant*time)

    ax.clear()
    ax.plot(time, N_t, label="Monte Carlo")
    ax.plot(time, theoretical, '--', label="Theoretical")
    ax.set_xlabel("Time")
    ax.set_ylabel("Remaining nuclei")
    ax.set_title(f"Decay simulation: N0={N0}, half-life={half_time}")
    ax.legend()
    plt.draw()

# -----------------------------
# Slider update function
# -----------------------------

def update_decay(val):
    N0 = int(slider_nuclei.val)
    half_time = slider_half_time.val
    total_time = slider_total_time.val
    N_t = simulate_decay(N0, half_time, total_time, dt)
    plot_decay(N_t, N0, half_time, dt, ax)
    
# -----------------------------
# Main program
# -----------------------------

def main():
    global fig, ax, slider_nuclei, slider_half_time, slider_total_time, dt

    # Plot setup
    fig, ax = plt.subplots(figsize=(6,5))
    plt.subplots_adjust(bottom=0.25)  # space for sliders

    # Sliders
    ax_slider_nuclei = plt.axes([0.25, 0.15, 0.5, 0.03])
    slider_nuclei = Slider(ax_slider_nuclei, "Nuclei (N0)", 1000, 500000, valinit=1000, valstep=1000)

    ax_half_time = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider_half_time = Slider(ax_half_time, "Half-life", 1.0, 50.0, valinit=1.0, valstep=1.0)
    
    ax_total_time = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_total_time = Slider(ax_total_time, "Total Time", 20, 200, valinit=20, valstep=2)
    
    # time step based on half-life
    dt = 10.0/slider_half_time.val
    
    # Connect sliders to update function
    slider_nuclei.on_changed(update_decay)
    slider_half_time.on_changed(update_decay)
    slider_total_time.on_changed(update_decay)
    # Initial plot
    update_decay(None)

    plt.show()

if __name__ == "__main__":
    main()
