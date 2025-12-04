# Monte Carlo Simulations – Beginner Projects

This repository contains beginner-friendly projects that demonstrate Monte Carlo simulations applied to simple problems.

## Projects

### 1. Estimating the value of pi

- Description: Uses Monte Carlo simulation to estimate the value of pi by randomly generating points in a square and counting how many fall inside a quarter circle.
- Script: `src/MC_pi.py`
- Result: Scatter plot showing the points inside and outside the quarter circle.

### 2. Radioactive Decay Simulation

- Description: Simulates the stochastic decay of radioactive particles over time using Monte Carlo methods and compares it with the theoretical exponential decay.
- Script: `src/MC_radioactive_decay.py`
- Result: Plot of simulated decay vs theoretical curve.

### 3. Computing the value of one variable integrals

- Description: Calculates the integral of a function in a defined interval, calculating the relative error.
- Script: `src/MC_integrals.py`
- Result: Plot of simulated and exact curve.

### 4. Seeking the minimum value of a multi-variable function

- Description: Search for the minimum of a 2-variable function in a 2D projection and look the position in a surface.
- Script: `src/MC_minimization.py`
- Result: Plot of a surface and its projection in a XY-plane searching for the minimum of the function.

### 5. Brownian motion in a flat surface

- Description: Plotting the Brownian motion of multiple particles without considering interaction between them.
- Script: `src/MC_"2D_brownian_motion".py`
- Result: Plot of the brownian motion in a XY-plane for 5-particles.

### 6. Brownian motion in a 3D surface

- Description: Plotting the Brownian motion of 5 particles without considering interaction between them. Allows to check for other paths based on pseudo random numbers by clicking on the button Refresh
- Script: `src/MC_interactive_3D_brownian_motion".py`
- Result: Plot of the brownian motion for 5-particles with a Refresh button to see other possible paths.

### 7. Animation of the Brownian motion in a 3D surface

- Description: Animation of the Brownian motion of a particles.
- Script: `src/MC_time_evolution_3D_brownian_motion".py`
- Result: Motion of a particle in a volume.

### Future Projects

- More Monte Carlo simulations in science and mathematics
- Applications in physics, chemistry, and data analysis

## Technologies

- Python 3.14
- NumPy
- Matplotlib

## How to Run

Clone the repository and run the scripts:

git clone https://github.com/JeysonEsCu/Monte-Carlo-Simulations.git
cd monte-carlo-simulations
python src/monte_carlo_pi.py
python src/radioactive_decay.py
