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

### 3.Computing the value of one variable integrals

- Description: Calculates the integral of a function in a defined interval, calculating the relative error.
- Script: `src/MC_integrals.py`
- Result: Plot of simulated and exact curve.

### 4.Seeking the minimum value of a multi-variable function

- Description: Search for the minimum of a 2-variable function in a 2D projection and look the position in a surface
- Script: `src/MC_minimization.py`
- Result: Plot of a surface and its projection in a XY-plane searching for the minimum of the function

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
