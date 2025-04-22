import numpy as np
import pandas as pd
from causality.inference.search import IC
from causality.estimation.adjustments import AdjustForDirectCauses
from dowhy import CausalModel

# Define the structural equations
np.random.seed(42)
n_samples = 1000
X = np.random.normal(size=n_samples)
U_M = np.random.normal(size=n_samples)
M = 0.5 * X + U_M
U_Y = np.random.normal(size=n_samples)
Y = 0.3 * M + 0.2 * X + U_Y

# Create a DataFrame
data = pd.DataFrame({'X': X, 'M': M, 'Y': Y})

# Define a CausalModel using DoWhy
model = CausalModel(
    data=data,
    treatment='X',
    outcome='Y',
    common_causes=['M']
)

# Identify the causal effect
identified_estimand = model.identify_effect()

# Estimate the causal effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print(f"Estimated causal effect of X on Y: {estimate.value:.4f}")