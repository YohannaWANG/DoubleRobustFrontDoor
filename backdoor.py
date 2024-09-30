# This is a sample Python script.


import numpy as np
import pandas as pd
# import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare


def simulate_data(n):
    # Covariates (X)
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)

    # Treatment (A)
    A = np.random.binomial(1, 0.5, n)

    # Outcome (Y)
    beta0 = 1
    beta1 = 2  # True causal effect
    beta2 = 1.5
    beta3 = -1
    epsilon = np.random.normal(0, 1, n)
    Y = beta0 + beta1 * A + beta2 * X1 + beta3 * X2 + epsilon

    return pd.DataFrame({'A': A, 'X1': X1, 'X2': X2, 'Y': Y})

def estimate_propensity_scores(data_first_half):
    propensity_model = LogisticRegression()
    propensity_model.fit(data_first_half[['X1', 'X2']], data_first_half['A'])
    return propensity_model

def estimate_outcome_model(data_second_half):
    outcome_model = LinearRegression()
    outcome_model.fit(data_second_half[['A', 'X1', 'X2']], data_second_half['Y'])
    return outcome_model

def true_outcome_model(X1, X2, A):
    # True coefficients
    beta0 = 1
    beta1 = 2
    beta2 = 1.5
    beta3 = -1
    return beta0 + beta1 * A + beta2 * X1 + beta3 * X2

def double_robust_estimator(data, propensity_model, outcome_model):
    half = len(data) // 2
    data_second_half = data[half:].copy()

    # Estimate propensity scores
    data_second_half['propensity'] = propensity_model.predict_proba(data_second_half[['X1', 'X2']])[:, 1]

    # Estimate outcome predictions
    data_second_half['outcome_pred'] = outcome_model.predict(data_second_half[['A', 'X1', 'X2']])
    data_second_half['outcome_true'] = true_outcome_model(data_second_half['X1'], data_second_half['X2'], data_second_half['A'])

    # Calculate the outcome predictions for both treated and untreated
    data_second_half['outcome_pred_A1'] = outcome_model.predict(pd.DataFrame({'A': np.ones(half), 'X1': data_second_half['X1'], 'X2': data_second_half['X2']}))
    data_second_half['outcome_pred_A0'] = outcome_model.predict(pd.DataFrame({'A': np.zeros(half), 'X1': data_second_half['X1'], 'X2': data_second_half['X2']}))

    # Calculate the doubly robust estimate
    data_second_half['dr_estimator'] = (
        (data_second_half['A'] / data_second_half['propensity'] - (1 - data_second_half['A']) / (1 - data_second_half['propensity'])) *
        (data_second_half['Y'] - data_second_half['outcome_pred']) +
        data_second_half['outcome_pred_A1'] -
        data_second_half['outcome_pred_A0']
    )

    # Calculate the average treatment effect
    return data_second_half['dr_estimator'].mean(), data_second_half['outcome_pred'], data_second_half['outcome_true']

def propensity_score_weighting(data, propensity_model):
    half = len(data) // 2
    data_second_half = data[half:].copy()
    data_second_half['propensity'] = propensity_model.predict_proba(data_second_half[['X1', 'X2']])[:, 1]
    weights = data_second_half['A'] / data_second_half['propensity'] - (1 - data_second_half['A']) / (1 - data_second_half['propensity'])
    estimate = np.mean(weights * data_second_half['Y'])
    return estimate, data_second_half['Y'] * weights

def outcome_regression(data, outcome_model):
    half = len(data) // 2
    data_second_half = data[half:].copy()
    data_second_half['outcome_pred_A1'] = outcome_model.predict(pd.DataFrame({'A': np.ones(half), 'X1': data_second_half['X1'], 'X2': data_second_half['X2']}))
    data_second_half['outcome_pred_A0'] = outcome_model.predict(pd.DataFrame({'A': np.zeros(half), 'X1': data_second_half['X1'], 'X2': data_second_half['X2']}))
    estimate = np.mean(data_second_half['outcome_pred_A1'] - data_second_half['outcome_pred_A0'])
    return estimate, data_second_half['outcome_pred_A1'] * data_second_half['A'] + data_second_half['outcome_pred_A0'] * (1 - data_second_half['A'])

def calculate_chi_square_distance(observed, expected):
    observed_freq, expected_freq = np.histogram(observed, bins=10), np.histogram(expected, bins=10)
    return chisquare(observed_freq[0], expected_freq[0])[0]

def chi_square_distance_gaussians(mu1, sigma1, mu2, sigma2, num_samples=10000, num_bins=50):
    # Generate samples from both Gaussian distributions
    samples1 = np.random.normal(mu1, sigma1, num_samples)
    samples2 = np.random.normal(mu2, sigma2, num_samples)

    # Define the bin edges
    bin_edges = np.linspace(min(samples1.min(), samples2.min()), max(samples1.max(), samples2.max()), num_bins + 1)

    # Calculate the histogram frequencies for both sets of samples
    hist1, _ = np.histogram(samples1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(samples2, bins=bin_edges, density=True)

    # Normalize the histograms to get probabilities
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # Calculate the Chi-square distance
    chi_square_dist = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))  # Add a small value to avoid division by zero

    return chi_square_dist

def run_experiment(sample_size, num_experiments):
    results_dr = []
    results_ps = []
    results_or = []
    chi_square_dr = []
    chi_square_ps = []
    chi_square_or = []
    mse_yyhat_dr = []

    for _ in range(num_experiments):
        data = simulate_data(sample_size)

        half = sample_size // 2
        data_first_half = data[:half]
        data_second_half = data[half:]

        propensity_model = estimate_propensity_scores(data_first_half)
        outcome_model = estimate_outcome_model(data_second_half)

        dr_estimate, dr_outcome_pred, dr_outcome_true = double_robust_estimator(data, propensity_model, outcome_model)
        ps_estimate, ps_outcome_pred = propensity_score_weighting(data, propensity_model)
        or_estimate, or_outcome_pred = outcome_regression(data, outcome_model)

        results_dr.append(dr_estimate)
        results_ps.append(ps_estimate)
        results_or.append(or_estimate)

        true_outcome = data_second_half['Y']
        sigma1 = 1
        sigma2 = 1

        chi_square_dr.append(chi_square_distance_gaussians(np.mean(true_outcome), sigma1, np.mean(dr_outcome_pred), sigma2, num_samples=sample_size, num_bins=50))
        chi_square_ps.append(chi_square_distance_gaussians(np.mean(true_outcome), sigma1, np.mean(ps_outcome_pred), sigma2, num_samples=sample_size, num_bins=50))
        chi_square_or.append(chi_square_distance_gaussians(np.mean(true_outcome), sigma1, np.mean(or_outcome_pred), sigma2, num_samples=sample_size, num_bins=50))

        mse_yyhat_dr.append(np.mean((dr_outcome_true - dr_outcome_pred) ** 2))
        #chi_square_dr.append(calculate_chi_square_distance(true_outcome, dr_outcome_pred))
        #chi_square_ps.append(calculate_chi_square_distance(true_outcome, ps_outcome_pred))
        #chi_square_or.append(calculate_chi_square_distance(true_outcome, or_outcome_pred))

    return results_dr, results_ps, results_or, chi_square_dr, chi_square_ps, chi_square_or, mse_yyhat_dr

sample_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
num_experiments = 100

results_dr_all = []
results_ps_all = []
results_or_all = []
chi_square_dr_all = []
chi_square_ps_all = []
chi_square_or_all = []
mse_yyhat_dr_all = []

for size in sample_sizes:
    results_dr, results_ps, results_or, chi_square_dr, chi_square_ps, chi_square_or, mse_yyhat_dr = run_experiment(size, num_experiments)
    results_dr_all.append(results_dr)
    results_ps_all.append(results_ps)
    results_or_all.append(results_or)
    chi_square_dr_all.append(chi_square_dr)
    chi_square_ps_all.append(chi_square_ps)
    chi_square_or_all.append(chi_square_or)
    mse_yyhat_dr_all.append(mse_yyhat_dr)
#print('mse', mse_yyhat_dr_all)
#print('chi square', chi_square_dr_all)

# Create a DataFrame for the chi-square results
chi_square_df = pd.DataFrame({
    'Sample Size': np.repeat(sample_sizes, num_experiments),
    'Double Robust': np.concatenate(chi_square_dr_all),
    'Propensity Score': np.concatenate(chi_square_ps_all),
    'Outcome Regression': np.concatenate(chi_square_or_all),
    'MSE error': np.concatenate(mse_yyhat_dr_all)
})

# Melt the DataFrame for plotting
chi_square_melted = chi_square_df.melt(id_vars=['Sample Size'], value_vars=['Double Robust', 'Propensity Score', 'Outcome Regression'],
                                       var_name='Method', value_name='Chi-Square Distance')

# Plot the results
plt.figure(figsize=(12, 6))
sns.boxplot(x='Sample Size', y='Chi-Square Distance', hue='Method', data=chi_square_melted)
plt.title('Comparison of Chi-Square Distance for Causal Effect Estimation Methods')
plt.legend()

# Calculate mean Chi-Square distance for each sample size and method
# mean_chi_square = chi_square_melted.groupby(['Sample Size', 'Method'])['Chi-Square Distance'].mean().reset_index()

# Group by 'Sample Size' and calculate the mean
mean_values = chi_square_df.groupby('Sample Size').mean().reset_index()

# Plot the data
plt.figure(figsize=(10, 6))
sns.lineplot(data=mean_values, x='Double Robust', y='MSE error', hue='Sample Size', marker='o')
plt.title('Mean MSE error vs Mean Double Robust')
plt.xlabel('Mean Double Robust')
plt.ylabel('Mean MSE error')
plt.grid(True)
plt.show()

# Plot the data
plt.figure(figsize=(10, 6))
sns.scatterplot(data=mean_values, x='Double Robust', y='MSE error', hue='Sample Size', style='Sample Size', s=100, markers=['o', 's', 'D'])
sns.lineplot(data=mean_values, x='Double Robust', y='MSE error', hue='Sample Size', markers=True, dashes=False)
sns.regplot(data=mean_values, x='Double Robust', y='MSE error', scatter=False, color='grey', line_kws={"linestyle":"--"})

plt.title('Mean MSE error vs Mean Double Robust')
plt.xlabel('Mean Double Robust')
plt.ylabel('Mean MSE error')
plt.grid(True)
plt.show()