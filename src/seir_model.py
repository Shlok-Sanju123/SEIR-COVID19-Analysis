
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from pathlib import Path

Path("results").mkdir(exist_ok=True)

# Define SEIR model
def seir_model(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Predict daily reported cases
def predict_cases(params, t_data, N):
    beta, sigma, gamma, rho, log_E0, log_I0 = params

    E0 = 10 ** log_E0
    I0 = 10 ** log_I0
    R0 = 0
    S0 = N - E0 - I0 - R0
    initial_conditions = [S0, E0, I0, R0]

    solution = solve_ivp(
        seir_model,
        (t_data[0], t_data[-1]),
        initial_conditions,
        t_eval=t_data,
        args=(beta, sigma, gamma, N)
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    S, I = solution.y[0], solution.y[2]
    return rho * beta * S * I / N

# Calculate residuals
def residuals(params, t_data, observed_cases, N):
    predicted = predict_cases(params, t_data, N)
    return predicted - observed_cases

# Load and prepare Maharashtra data
data = pd.read_csv("data/processed/maharashtra_covid.csv")
data["Date"] = pd.to_datetime(data["Date"])

first_wave = data[
    (data["Date"] >= "2020-03-09") &
    (data["Date"] <= "2020-12-31")
].reset_index(drop=True)

observed_cases = first_wave["New_Cases"].to_numpy()
time_data = np.arange(len(observed_cases))

# Define total population
N = 112_000_000

# Parameter bounds
lower_bounds = [0.05, 0.10, 0.05, 0.01, 0, 0]
upper_bounds = [1.0, 1.0, 1.0, 1.0, 6, 6]

# Multiple initial guesses
initial_guesses = [
    [0.3, 0.2, 0.1, 0.2, 2, 1],
    [0.5, 0.3, 0.2, 0.3, 3, 2],
    [0.7, 0.5, 0.3, 0.5, 4, 3],
    [0.2, 0.2, 0.1, 0.5, 3, 2],
    [0.8, 0.4, 0.2, 0.2, 4, 2]
]

# Run optimisation from multiple starting points
results = []

for guess in initial_guesses:
    result = least_squares(
        residuals,
        x0=guess,
        bounds=(lower_bounds, upper_bounds),
        args=(time_data, observed_cases, N),
        x_scale="jac",
        max_nfev=3000
    )

    if result.success:
        results.append(result)

if not results:
    raise RuntimeError("All optimisation attempts failed.")

# Select the solution with the lowest residual error
result = min(results, key=lambda x: np.sum(x.fun ** 2))

beta_est, sigma_est, gamma_est, rho_est, log_E0, log_I0 = result.x

E0 = 10 ** log_E0
I0 = 10 ** log_I0
R0_initial = 0
S0 = N - E0 - I0 - R0_initial

initial_conditions = [S0, E0, I0, R0_initial]

predicted_cases = predict_cases(result.x, time_data, N)

# Calculate model evaluation metrics
reproduction_number = beta_est / gamma_est
rmse = np.sqrt(np.mean((predicted_cases - observed_cases) ** 2))

ss_res = np.sum((observed_cases - predicted_cases) ** 2)
ss_tot = np.sum((observed_cases - np.mean(observed_cases)) ** 2)
r2 = 1 - ss_res / ss_tot

# Display results
print("\nMaharashtra first-wave analysis")
print("Observations:", len(observed_cases))
print("Period:", first_wave["Date"].min().date(), "to", first_wave["Date"].max().date())

print("\nEstimated SEIR parameters")
print(f"Beta: {beta_est:.5f}")
print(f"Sigma: {sigma_est:.5f}")
print(f"Gamma: {gamma_est:.5f}")
print(f"Rho: {rho_est:.5f}")

print("\nEstimated initial populations")
print(f"Susceptible: {S0:.0f}")
print(f"Exposed: {E0:.0f}")
print(f"Infectious: {I0:.0f}")
print(f"Recovered: {R0_initial:.0f}")

print("\nModel evaluation")
print(f"R0: {reproduction_number:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Optimisation successful: {result.success}")
print(f"Optimisation attempts: {len(initial_guesses)}")

# Check whether parameters reached their bounds
parameter_names = ["Beta", "Sigma", "Gamma", "Rho", "log_E0", "log_I0"]

print("\nParameter bound check")
for name, value, lower, upper in zip(
    parameter_names, result.x, lower_bounds, upper_bounds
):
    at_bound = np.isclose(value, lower, atol=1e-4) or np.isclose(value, upper, atol=1e-4)
    print(f"{name}: {'At bound' if at_bound else 'Within bounds'}")

# Plot actual versus predicted cases
plt.figure(figsize=(12, 6))
plt.plot(time_data, observed_cases, label="Actual reported cases")
plt.plot(time_data, predicted_cases, label="SEIR predicted cases")
plt.xlabel("Time (days)")
plt.ylabel("Daily confirmed cases")
plt.title("Actual vs Predicted COVID-19 Cases - Maharashtra")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png", dpi=300)
plt.show()
