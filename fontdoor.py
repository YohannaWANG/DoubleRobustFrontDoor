import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from autograd import jacobian

from autograd import numpy as anp
from autograd import grad

import matplotlib.pyplot as plt

# expit is already provided by scipy.special (sigmoid function), so no need to redefine it

######## GENERATE DATASET ###########
# Y and M     A     C1    C2    C3
# 1.000000  1.0  0.0  0.0
# 0.999999  1.0  0.0  0.0  0.0  1.0
# 0.999998  1.0  0.0  0.0  0.0  1.0

"""
Ground Truth Parameters: 
"""
p_c1, p_c3 = 0.6, 0.3
alpha = np.array([1, 0.5])
omega = np.array([0.5, 0.2, 0.4, 0.5, 0.2])
beta = np.array([1, 1, -2, 2, 8, 0])
theta = np.array([1, 2, 2, -8, 3, 1, 1, 1])
sigma_m, sigma_y = 2.0, 1.0
"""
Elements in truth_est
 Mean y: `-22.81326`
 PSI   : `-18.20763`
 PIIE-diff  : `-4.605624`
 PIIE-exp   : `-4.605624`
"""
truth_est = np.array([-22.81326, -18.20763, -4.605624, -4.605624])
def sigmoid(x):
    return 1 / (1 + anp.exp(-x))
def gen_med_data_continuous(n, p_c1, p_c3, alpha, omega, beta, theta, sigma_m, sigma_y):
    # Generate first confounder (c1)
    c1 = np.random.binomial(1, p_c1, size=n)

    # Generate second confounder (c2) given c1
    p_c2 = expit(np.c_[np.ones(n), c1] @ alpha) * (1 - int(alpha[0] == 0 and alpha[1] == 0))
    c2 = np.random.binomial(1, p_c2)

    # Generate a confounder of A & Y (independent of other confounders)
    c3 = np.random.binomial(1, p_c3, size=n)

    # Generate binary exposure (a) given c1, c2, c1 * c2, and c3
    p_a = expit(np.c_[np.ones(n), c1, c2, c1 * c2, c3] @ omega)
    a = np.random.binomial(1, p_a)

    # Generate continuous mediator (m) given a, c1, c2, and all combinations
    mean_m = np.c_[np.ones(n), a, c1, c2, c1 * c2, c3] @ beta
    m = np.random.normal(mean_m, sigma_m)

    # Generate continuous outcome Y (y) given a, m, c1, c2, c3, and all combinations
    mean_y = np.c_[np.ones(n), a, m, a * m, c1, c2, c1 * c2, c3] @ theta
    y = np.random.normal(mean_y, sigma_y)

    # Combine into a dataframe
    sim_data = pd.DataFrame(np.c_[y, a, m, c1, c2, c3], columns=['y', 'a', 'm', 'c1', 'c2', 'c3'])

    return sim_data


###### FUNCTION THAT CALCULATES PSI MLE ######
def psi_mle_function_cont(cov_vals_all, exposure_data, beta_hat, theta_hat, astar, interaction):
    mean_cov_all = np.mean(cov_vals_all, axis=0)
    mean_exposure = np.mean(exposure_data)
    mean_cov_exposure = np.mean(exposure_data[:, None] * cov_vals_all, axis=0)

    if interaction == 1:
        psi = (theta_hat[0] + theta_hat[2] * beta_hat[0] + theta_hat[2] * beta_hat[1] * astar
               + (theta_hat[1] + theta_hat[3] * beta_hat[0] + theta_hat[3] * beta_hat[1] * astar) * mean_exposure
               + (theta_hat[2] * beta_hat[2:] + theta_hat[4:]) @ mean_cov_all.T
               + theta_hat[3] * beta_hat[2:] @ mean_cov_exposure.T)
    else:
        psi = (theta_hat[0] + theta_hat[2] * beta_hat[0] + theta_hat[2] * beta_hat[1] * astar
               + theta_hat[1] * mean_exposure
               + (theta_hat[2] * beta_hat[2:] + theta_hat[3:]) @ mean_cov_all.T)

    return psi


###### FUNCTION THAT CALCULATES PIIE MLE AND VARIANCE ######
def piie_mle_variance_function_cont(cov_vals_all, exposure_data, theta_hat, beta_hat, alpha_hat, astar, interaction,
                                    fit_z, fit_y):
    n = len(exposure_data)

    mean_cov_all = np.mean(cov_vals_all, axis=0)
    mean_exposure = np.mean(exposure_data)
    mean_cov_exposure = np.mean(exposure_data[:, None] * cov_vals_all, axis=0)

    if interaction == 1:
        psi = (theta_hat[0] + theta_hat[2] * beta_hat[0] + theta_hat[2] * beta_hat[1] * astar
               + (theta_hat[1] + theta_hat[3] * beta_hat[0] + theta_hat[3] * beta_hat[1] * astar) * mean_exposure
               + (theta_hat[2] * beta_hat[2:] + theta_hat[4:]) @ mean_cov_all.T
               + theta_hat[3] * beta_hat[2:] @ mean_cov_exposure.T)
    else:
        psi = (theta_hat[0] + theta_hat[2] * beta_hat[0] + theta_hat[2] * beta_hat[1] * astar
               + theta_hat[1] * mean_exposure
               + (theta_hat[2] * beta_hat[2:] + theta_hat[3:]) @ mean_cov_all.T)

    if interaction == 1:
        est_piie = theta_hat[2] * beta_hat[1] * mean_exposure + theta_hat[3] * beta_hat[1] * mean_exposure
    else:
        est_piie = theta_hat[2] * beta_hat[1] * mean_exposure

    if interaction == 1:
        var_piie = ((np.var(exposure_data) / n) * (beta_hat[1] ** 2) * (theta_hat[2] + theta_hat[3]) ** 2
                    + (mean_exposure ** 2) * ((theta_hat[2] + theta_hat[3]) ** 2) * fit_z.cov_params().iloc[1, 1]
                    + mean_exposure * beta_hat[1] * (mean_exposure * beta_hat[1] * fit_y.cov_params().iloc[2, 2]
                                                     + mean_exposure * beta_hat[1] * fit_y.cov_params().iloc[2, 3])
                    + mean_exposure * beta_hat[1] * (mean_exposure * beta_hat[1] * fit_y.cov_params().iloc[2, 3]
                                                     + mean_exposure * beta_hat[1] * fit_y.cov_params().iloc[3, 3]))
    else:
        var_piie = (((mean_exposure * theta_hat[2]) ** 2) * fit_z.cov_params().iloc[1, 1]
                    + ((mean_exposure * beta_hat[1]) ** 2) * fit_y.cov_params().iloc[2, 2]
                    + ((beta_hat[1] * theta_hat[2]) ** 2) * np.var(exposure_data) / n)

    output = np.array([psi, est_piie, var_piie]).reshape(1, -1)
    return pd.DataFrame(output, columns=["PSI", "PIIE", "Var PIIE"])


def piie_sp_1_variance_function_cont(intermediate, outcome, fit_z, astar, interaction):
    n = len(outcome)
    sigma = np.std(fit_z.resid)  # Equivalent to summary(fit.z)$sigma in R

    model_matrix_z = fit_z.model.exog  # Model matrix from the fit_z
    model_matrix_z_astar = model_matrix_z.copy()
    model_matrix_z_astar[:, 1] = 0  # Replace second column with 0s (like in R code)

    beta_hat = fit_z.params  # Estimated coefficients from the model
    z_mean_astar = np.dot(model_matrix_z_astar, beta_hat)
    z_mean_ind = np.dot(model_matrix_z, beta_hat)

    # Calculate psi.sp.ind
    psi_sp_ind = outcome * (norm.pdf(intermediate, z_mean_astar, sigma) / norm.pdf(intermediate, z_mean_ind, sigma))

    piie_sp_ind = outcome - psi_sp_ind
    piie_sp = np.mean(outcome) - np.mean(psi_sp_ind)

    # Score.sp matrix
    score_sp = np.hstack([model_matrix_z * (intermediate - z_mean_ind).reshape(-1, 1),
                          (piie_sp_ind - piie_sp).reshape(-1, 1)])

    estimates = beta_hat
    len_z = len(beta_hat)

    # Calculate the Jacobian using autograd's jacobian function
    deriv_sp = jacobian(U_sp_1)(np.hstack([estimates, piie_sp]), model_matrix_z, outcome, intermediate, len_z, n, sigma, interaction)

    # Variance matrix
    var_sp = np.linalg.inv(deriv_sp) @ score_sp.T @ score_sp @ np.linalg.inv(deriv_sp).T

    # Variance for our estimator
    piie_var_sp = var_sp[-1, -1]  # The variance for the PIIE


    piie_var_sp_value = piie_var_sp[0]
    output = np.array([np.mean(psi_sp_ind), piie_sp, piie_var_sp_value])
    output_labels = ["PSI.1", "PIIE.1", "Var PIIE.1"]

    return dict(zip(output_labels, output))

# Define the score function U_sp_1 (similar to R function U.sp.1)
def U_sp_1(estimates, model_matrix_z, data_y, data_z, len_z, n, sigma, interaction):
    beta_hat = estimates[:len_z]
    piie_est = estimates[len(estimates)-1]

    model_matrix_z_astar = model_matrix_z.copy()
    model_matrix_z_astar[:, 1] = 0  # Set the second column to 0 (like in R code)

    z_mean_astar = anp.array(model_matrix_z_astar) @ beta_hat

    #z_mean_ind = np.dot(model_matrix_z, beta_hat)
    z_mean_ind = anp.array(model_matrix_z) @ beta_hat
    #print('check', sigma.size, data_z.size, z_mean_ind.shape, z_mean_astar.size, beta_hat.shape, model_matrix_z.shape, model_matrix_z_astar.shape)

    # pdf_values = norm.pdf(data_z, loc=z_mean_astar, scale = sigma)
    # print("norm.pdf(data_z, z_mean_astar, sigma)", pdf_values)

    # Calculate psi.sp.ind
    # psi_sp_ind = data_y * (norm.pdf(data_z, z_mean_astar, sigma) / norm.pdf(data_z, z_mean_ind, sigma))
    psi_sp_ind = data_y * (anp.exp(-(data_z - z_mean_astar)**2 / (2 * sigma**2)) /
                            anp.sqrt(2 * anp.pi * sigma**2) /
                            (anp.exp(-(data_z - z_mean_ind)**2 / (2 * sigma**2)) /
                             anp.sqrt(2 * anp.pi * sigma**2)))

    piie_sp_ind = data_y - psi_sp_ind

    piie_sp = anp.mean(data_y) - anp.mean(piie_sp_ind)

    # Score.sp matrix
    differences = data_z - z_mean_ind
    score_sp = anp.column_stack((model_matrix_z * differences[:, anp.newaxis],
                          (piie_sp_ind - piie_est)))

    deriv = anp.ones((1, n)) @ score_sp  # Compute the Jacobian correctly
    #deriv = np.sum(score_sp, axis=0).reshape(1, -1)

    return deriv



'''
###### FUNCTION THAT CALCULATES PSI SP 2 (MODEL FOR A,Y) ######
'''
def piie_sp_2_variance_function_cont(cov_vals_all, exposure, outcome, i_y, fit_a, fit_y, astar, interaction):

    n = len(exposure)

    # Model matrices from the fitted models
    model_matrix_a = fit_a.model.exog  # Equivalent to model.matrix(fit.a)
    model_matrix_y = fit_y.model.exog  # Equivalent to model.matrix(fit.y)

    # Coefficients from the models
    alpha_hat = fit_a.params  # Equivalent to summary(fit.a)$coefficients[,1]
    theta_hat = fit_y.params  # Equivalent to summary(fit.y)$coefficients[,1]

    # Select the relevant covariates for Y
    cov_vals_y = cov_vals_all[:, np.where(i_y == 1)[0]]

    # Calculate the means for a and y
    a_mean = expit(np.dot(model_matrix_a, alpha_hat))  # expit is the sigmoid function (logistic)
    y_mean = np.dot(model_matrix_y, theta_hat)

    # Calculate sum.a based on interaction condition
    if interaction == 1:
        sum_a = np.dot(np.column_stack((np.ones(n), a_mean, model_matrix_y[:, 2], a_mean * model_matrix_y[:, 2], cov_vals_y)), theta_hat)
    else:
        sum_a = np.dot(np.column_stack((np.ones(n), a_mean, model_matrix_y[:, 2], cov_vals_y)), theta_hat)

    # Calculate psi.sp.ind
    psi_sp_ind = ((1 - model_matrix_y[:, 1]) / (1 - a_mean)) * sum_a

    # Calculate piie.sp.ind
    piie_sp_ind = outcome - psi_sp_ind

    # Calculate piie.sp
    piie_sp = np.mean(outcome) - np.mean(psi_sp_ind)

    # Create the score.sp matrix
    score_sp = np.hstack([model_matrix_a * (exposure - a_mean).reshape(-1, 1),
                          model_matrix_y * (outcome - y_mean).reshape(-1, 1),
                          (piie_sp_ind - piie_sp).reshape(-1, 1)])

    # Combine the estimates for alpha and theta
    estimates = np.concatenate([alpha_hat, theta_hat])
    len_a = len(alpha_hat)

    # Calculate the Jacobian using autograd's jacobian function
    deriv_sp = jacobian(U_sp_2)(np.concatenate([estimates, [piie_sp]]), model_matrix_a, model_matrix_y, outcome, exposure, i_y, cov_vals_all, len_a, n, interaction)

    # Calculate the variance matrix
    var_sp = np.linalg.inv(deriv_sp) @ score_sp.T @ score_sp @ np.linalg.inv(deriv_sp).T

    # Variance for the estimator
    piie_var_sp = var_sp[-1, -1]  # The variance for the PIIE

    # Prepare the output
    output = np.array([np.mean(psi_sp_ind), piie_sp, piie_var_sp[0]])
    output_labels = ["PSI.2", "PIIE.2", "Var PIIE.2"]

    return dict(zip(output_labels, output))

def U_sp_2(estimates, model_matrix_a, model_matrix_y, data_y, data_a, i_y, cov_vals_all, len_a, n, interaction):
    # Extract estimates
    alpha_hat = estimates[:len_a]
    theta_hat = estimates[len_a:(len(estimates) - 1)]
    piie_est = estimates[-1]

    # Select relevant covariates for Y
    cov_vals_y = cov_vals_all[:, anp.where(i_y == 1)[0]]

    # Calculate mean for a and y
    a_mean = sigmoid(anp.dot(model_matrix_a, alpha_hat))  # expit is the sigmoid function
    y_mean = anp.dot(model_matrix_y, theta_hat)

    # Calculate sum.a based on whether interaction is 1
    if interaction == 1:
        sum_a = anp.dot(anp.column_stack((anp.ones(n), a_mean, model_matrix_y[:, 2], a_mean * model_matrix_y[:, 2], cov_vals_y)), theta_hat)
    else:
        sum_a = anp.dot(anp.column_stack((anp.ones(n), a_mean, model_matrix_y[:, 2], cov_vals_y)), theta_hat)

    # Calculate psi.sp.ind
    psi_sp_ind = ((1 - model_matrix_y[:, 1]) / (1 - a_mean)) * sum_a

    # Calculate piie.sp.ind
    piie_sp_ind = data_y - psi_sp_ind

    # Calculate piie.sp
    piie_sp = anp.mean(data_y) - anp.mean(psi_sp_ind)

    # Calculate score.sp
    diff_a_mean = (data_a - a_mean)
    diff_y_mean = (data_y - y_mean)
    diff_piie_est = (piie_sp_ind - piie_est)
    score_sp = anp.column_stack((model_matrix_a * diff_a_mean[:, anp.newaxis], #(data_a - a_mean),
                                 model_matrix_y * diff_y_mean[:, anp.newaxis], #(data_y - y_mean),
                                 diff_piie_est[:, anp.newaxis] ))

    # Calculate the derivative (Jacobian)
    deriv = anp.ones((1, n)) @ score_sp
    value = anp.array([d._value for d in deriv])
    return deriv


###### FUNCTION THAT CALCULATES PSI SP (MODEL FOR A, Z, Y) ######

def piie_sp_variance_function_cont(cov_vals_all, exposure, intermediate, outcome, i_y, i_z, i_a, fit_a, fit_z, fit_y, astar, interaction):
    n = len(exposure)

    # Sigma from the fitted model for z
    sigma = fit_z.scale  # This assumes fit_z is a fitted model object with scale attribute
    print('sigma', sigma)

    # Model matrices from the fitted models
    model_matrix_a = fit_a.model.exog  # Equivalent to model.matrix(fit.a)
    model_matrix_z = fit_z.model.exog  # Equivalent to model.matrix(fit.z)
    model_matrix_y = fit_y.model.exog  # Equivalent to model.matrix(fit.y)

    # Creating model.matrix.z_astar
    model_matrix_z_astar = np.copy(model_matrix_z)
    model_matrix_z_astar[:, 1] = 0

    # Coefficients from the models
    theta_hat = fit_y.params
    beta_hat = fit_z.params
    alpha_hat = fit_a.params

    # Select relevant covariates
    cov_vals_y = cov_vals_all[:, np.where(i_y == 1)[0]]
    cov_vals_z = cov_vals_all[:, np.where(i_z == 1)[0]]
    cov_vals_a = cov_vals_all[:, np.where(i_a == 1)[0]]

    # Calculate means
    z_mean_astar = np.dot(model_matrix_z_astar, beta_hat)
    z_mean_ind = np.dot(model_matrix_z, beta_hat)
    a_mean = expit(np.dot(model_matrix_a, alpha_hat))
    y_mean = np.dot(model_matrix_y, theta_hat)

    # Calculate sum.a
    if interaction == 1:
        sum_a = anp.dot(anp.column_stack((anp.ones(n), a_mean, model_matrix_y[:, 2], a_mean * model_matrix_y[:, 2], cov_vals_y)), theta_hat)
    else:
        sum_a = anp.dot(anp.column_stack((anp.ones(n), a_mean, model_matrix_y[:, 2], cov_vals_y)), theta_hat)

    # Calculate sum.z
    if interaction == 1:
        sum_z = anp.dot(anp.column_stack((anp.ones(n), model_matrix_y[:, 1], z_mean_astar, model_matrix_y[:, 1] * z_mean_astar, cov_vals_y)), theta_hat)
    else:
        sum_z = anp.dot(anp.column_stack((anp.ones(n), model_matrix_y[:, 1], z_mean_astar, cov_vals_y)), theta_hat)

    # Calculate sum.az
    if interaction == 1:
        sum_az = anp.dot(anp.column_stack((anp.ones(n), a_mean, z_mean_ind, a_mean * z_mean_ind, cov_vals_y)), theta_hat)
    else:
        sum_az = anp.dot(anp.column_stack((anp.ones(n), a_mean, z_mean_ind, cov_vals_y)), theta_hat)

    # Calculate psi.sp.ind
    psi_sp_ind = ((outcome - y_mean) *
                  (norm.pdf(model_matrix_y[:, 2], z_mean_astar, sigma) / norm.pdf(model_matrix_y[:, 2], z_mean_ind, sigma))
                  + ((1 - model_matrix_y[:, 1]) / (1 - a_mean)) * (sum_a - sum_az)
                  + sum_z)

    # Calculate piie.sp.ind and piie.sp
    piie_sp_ind = outcome - psi_sp_ind
    piie_sp = np.mean(outcome) - np.mean(psi_sp_ind)

    # Calculate score.sp
    diff_a_mean = (exposure - a_mean)    # Exposure - a_mean
    diff_z_mean = (intermediate - z_mean_ind)    # Intermediate - z_mean_ind
    diff_y_mean = (outcome - y_mean)    # Outcome - y_mean
    diff_piie_est = (piie_sp_ind - piie_sp)    # PIIE estimate - PIIE
    score_sp = np.column_stack((
        model_matrix_a * diff_a_mean[:, anp.newaxis],
        model_matrix_z * diff_z_mean[:, anp.newaxis],
        model_matrix_y * diff_y_mean[:, anp.newaxis],
        diff_piie_est[:, anp.newaxis] #(piie_sp_ind - piie_sp)
    ))

    # Combine the estimates for alpha, beta, and theta
    estimates = np.concatenate([alpha_hat, beta_hat, theta_hat])
    len_a = len(alpha_hat)
    len_z = len(beta_hat)

    # Take the derivative using autograd's jacobian function
    deriv_sp = jacobian(U_sp)(np.concatenate([estimates, [piie_sp]]),
                                model_matrix_a, model_matrix_z, model_matrix_y,
                                exposure, intermediate, outcome,
                                #outcome, intermediate, exposure,
                                i_y, i_z, i_a,
                                cov_vals_all, len_a, len_z, n, sigma, interaction)
    # Calculate variance matrix
    var_sp = np.linalg.inv(deriv_sp) @ score_sp.T @ score_sp @ np.linalg.inv(deriv_sp).T

    # Variance for the estimator
    piie_var_sp = var_sp[-1, -1]

    # Prepare the output
    output = np.array([np.mean(psi_sp_ind), piie_sp, piie_var_sp[0]])
    output_labels = ["PSI", "PIIE", "Var PIIE"]

    return dict(zip(output_labels, output))

def U_sp(estimates, model_matrix_a, model_matrix_z, model_matrix_y, data_a, data_z, data_y, i_y, i_z, i_a, cov_vals_all, len_a, len_z, n, sigma, interaction):
    # Extract estimates
    alpha_hat = estimates[:len_a]
    beta_hat = estimates[len_a:(len_a + len_z)]
    theta_hat = estimates[(len_a + len_z):-1]
    piie_est = estimates[-1]
    '''
    For sanity check, check the above values (DONE)
        alpha_hat [0.54194274 0.4764959  0.3738905  0.33140127 0.13821203]
        beta_hat [ 1.3139815   0.86492654 -2.15923318  1.80421404  8.09725206]
        theta_hat [ 1.0867734   1.90697671  1.97601842 -7.9720439   2.86613653  0.89728913  1.16240122  1.07033716]
        piie_est -4.364293926405566
    '''

    # Creating model.matrix.z_astar
    model_matrix_z_astar = np.copy(model_matrix_z)
    model_matrix_z_astar[:, 1] = 0

    # Select relevant covariates for Y, Z, and A
    cov_vals_y = cov_vals_all[:, np.where(i_y == 1)[0]]
    cov_vals_z = cov_vals_all[:, np.where(i_z == 1)[0]]
    cov_vals_a = cov_vals_all[:, np.where(i_a == 1)[0]]

    # Calculate means
    z_mean_astar = anp.dot(model_matrix_z_astar, beta_hat)
    z_mean_ind = anp.dot(model_matrix_z, beta_hat)
    a_mean = sigmoid(anp.dot(model_matrix_a, alpha_hat))  #expit(np.dot(model_matrix_a, alpha_hat))
    y_mean = anp.dot(model_matrix_y, theta_hat)

    """
    For sanity check, check the above values (DONE)
        z_mean_astar 0.0
        z_mean_ind 0.0
        a_mean 0.5419427422363281
    """

    # Calculate sum.a
    if interaction == 1:
        sum_a = anp.dot(anp.column_stack((anp.ones(n), a_mean, model_matrix_y[:, 2], a_mean * model_matrix_y[:, 2], cov_vals_y)), theta_hat)
    else:
        sum_a = anp.dot(anp.column_stack((anp.ones(n), a_mean, model_matrix_y[:, 2], cov_vals_y)), theta_hat)

    # Calculate sum.z
    if interaction == 1:
        sum_z = anp.dot(anp.column_stack((anp.ones(n), model_matrix_y[:, 1], z_mean_astar, model_matrix_y[:, 1] * z_mean_astar, cov_vals_y)), theta_hat)
    else:
        sum_z = anp.dot(np.column_stack((anp.ones(n), model_matrix_y[:, 1], z_mean_astar, cov_vals_y)), theta_hat)

    # Calculate sum.az
    if interaction == 1:
        sum_az = anp.dot(anp.column_stack((anp.ones(n), a_mean, z_mean_ind, a_mean * z_mean_ind, cov_vals_y)), theta_hat)
    else:
        sum_az = anp.dot(anp.column_stack((anp.ones(n), a_mean, z_mean_ind, cov_vals_y)), theta_hat)

    # Calculate psi.sp.ind
    '''
    Note: autograd for automatic differentiation, 
              we cannot directly use functions like scipy.stats.norm.pdf (not compatible with autograd) \
              Instead, we need to manually calculate the pdf using the normal distribution formula.
    '''
    # psi_sp_ind = ((data_y - y_mean) *
    #               (norm.pdf(model_matrix_y[:, 2], z_mean_astar, sigma) / norm.pdf(model_matrix_y[:, 2], z_mean_ind, sigma))
    #               + ((1 - model_matrix_y[:, 1]) / (1 - a_mean)) * (sum_a - sum_az)
    #               + sum_z)

    # Define a custom normal PDF function using autograd-compatible operations
    def norm_pdf_autograd(x, mu, sigma):
        return (1.0 / (anp.sqrt(2 * anp.pi) * sigma)) * anp.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Now use norm_pdf_autograd in your psi_sp_ind calculation
    psi_sp_ind = ((data_y - y_mean) *
                  (norm_pdf_autograd(model_matrix_y[:, 2], z_mean_astar, sigma) /
                   norm_pdf_autograd(model_matrix_y[:, 2], z_mean_ind, sigma))
                  + ((1 - model_matrix_y[:, 1]) / (1 - a_mean)) * (sum_a - sum_az)
                  + sum_z)

    # Calculate piie.sp.ind
    piie_sp_ind = data_y - psi_sp_ind
    piie_sp = anp.mean(data_y) - anp.mean(psi_sp_ind._value)

    # Calculate score.sp
    diff_a_mean = (data_a - a_mean)    # Exposure - a_mean
    diff_z_mean = (data_z - z_mean_ind)    # Intermediate - z_mean_ind
    diff_y_mean = (data_y - y_mean)    # Outcome - y_mean
    diff_piie_est = (piie_sp_ind - piie_est)    # PIIE estimate - PIIE
    score_sp = anp.column_stack((
        model_matrix_a * diff_a_mean[:, anp.newaxis], #(data_a - a_mean),
        model_matrix_z * diff_z_mean[:, anp.newaxis], #(data_z - z_mean_ind),
        model_matrix_y * diff_y_mean[:, anp.newaxis], #(data_y - y_mean),
        diff_piie_est[:, anp.newaxis] #(piie_sp_ind - piie_est)
    ))

    """
    Sanity check: check the above values (DONE)
        psi_sp_ind [-18.20763131 -18.20763131 -18.20763131 -18.20763131 -18.20763131 -18.20763131 -18.20763131 -18.20763131 -18.20763131 -18.20763131]
    """
    # Calculate the derivative (Jacobian)
    # deriv = anp.dot(anp.ones((1, n)), score_sp) #anp.ones((1, n)) @ score_sp
    deriv = anp.ones((1, n)) @ score_sp
    value = anp.array([d._value for d in deriv])
    return deriv


def chi_square_distance_gaussians(mu1, sigma1, mu2, sigma2, num_samples=1000, num_bins=50):
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

def run_experiments():
    n = 1000

    # Generate the dataset
    data = gen_med_data_continuous(n, p_c1, p_c3, alpha, omega, beta, theta, sigma_m, sigma_y)
    print(data)

    # Fit linear model for the mediator (m) ~ a + c1 + c2 + (c1 * c2)
    fit_z = smf.ols('m ~ a + c1 + c2 + I(c1*c2)', data=data).fit()

    # Fit linear model for the outcome (y) ~ a + m + (a * m) + c1 + c2 + (c1 * c2) + c3
    fit_y = smf.ols('y ~ a + m + I(a*m) + c1 + c2 + I(c1*c2) + c3', data=data).fit()

    # Fit logistic model for the binary exposure a ~ c1 + c2 + (c1 * c2) + c3
    fit_a = smf.logit('a ~ c1 + c2 + I(c1*c2) + c3', data=data).fit()

    # Extracting the coefficients
    alpha_hat = fit_a.params.values  # Coefficients from logistic regression
    beta_hat = np.append(fit_z.params.values, 0)  # Coefficients from linear model for m (add 0 for consistency)
    theta_hat = fit_y.params.values  # Coefficients from linear model for y

    # Create the confounders matrix (c1, c2, c1 * c2, c3)
    confounders = np.c_[data.iloc[:, 3],  # c1
                        data.iloc[:, 4],  # c2
                        data.iloc[:, 3] * data.iloc[:, 4],  # c1 * c2
                        data.iloc[:, 5]]  # c3

    # Output for coefficients (optional)
    print("Alpha hat (logit model coefficients):", alpha_hat)
    print("Beta hat (linear model coefficients):", beta_hat)
    print("Theta hat (linear model coefficients):", theta_hat)

    '''
    ####  Calculate PSI MLE
    
    # Assuming the following inputs:
    # confounders: Your confounder matrix (numpy array or pandas DataFrame)
    # data['a']: Exposure data (numpy array or pandas Series)
    # theta_hat: Estimated theta parameters (numpy array)
    # beta_truth: True beta values (numpy array)
    # alpha_hat: Estimated alpha parameters (numpy array)
    # truth_est[3]: The true estimate (scalar)
    # fit_z, fit_y: The fitted model objects
    '''

    # Call the PIIE MLE variance function
    out_mle = piie_mle_variance_function_cont(confounders, data['a'].values, theta_hat, beta, alpha_hat, 0, 1, fit_z, fit_y)

    # Calculate proportional bias (same as in R code)
    prop_bias_mle = (out_mle['PIIE'][0] - truth_est[2]) / truth_est[2]

    # Compute the 95% confidence interval
    ci_low = out_mle['PIIE'][0] - norm.ppf(0.975) * np.sqrt(out_mle['Var PIIE'][0])
    ci_up = out_mle['PIIE'][0] + norm.ppf(0.975) * np.sqrt(out_mle['Var PIIE'][0])

    # Determine if the true value is within the CI (coverage)
    ci_coverage_mle = int(truth_est[3] >= ci_low and truth_est[3] <= ci_up)

    # Append the results to the output (mimics the cbind operation in R)
    out_mle['Prop Bias MLE'] = prop_bias_mle
    out_mle['CI Coverage MLE'] = ci_coverage_mle

    # Display the final output
    print(out_mle)

    '''
    ### Calculate PSI SP 1 
    
    Assuming the following inputs:
    data['m']: Mediator data (numpy array or pandas Series)
    data['y']: Outcome data (numpy array or pandas Series)
    fit_z: The fitted model object for the mediator (statsmodels object)
    truth_est[2]: The true estimate (scalar)
    '''

    # SP 1 - Call the function for variance estimation
    out_sp_1 = piie_sp_1_variance_function_cont(data['m'].values, data['y'].values, fit_z, 0, 1)

    # Calculate proportion bias
    prop_bias_sp_1 = (out_sp_1['PIIE.1'] - truth_est[2]) / truth_est[2]

    # Confidence interval calculation (similar to qnorm(.975) in R, which is 1.96)
    ci_low = out_sp_1['PIIE.1'] - norm.ppf(0.975) * np.sqrt(out_sp_1['Var PIIE.1'])
    ci_up = out_sp_1['PIIE.1'] + norm.ppf(0.975) * np.sqrt(out_sp_1['Var PIIE.1'])

    # Coverage check
    ci_coverage_sp_1 = int(truth_est[2] >= ci_low and truth_est[2] <= ci_up)

    # Combine the results into a final output
    out_sp_1 = np.append(out_sp_1, [prop_bias_sp_1, ci_coverage_sp_1])

    # Assuming out_sp_1 is now a NumPy array, you can convert it to a DataFrame if needed
    # or use it directly.
    print(out_sp_1)

    ''' 
    ### Calculate PSI SP 2   
    
    Assuming the following inputs:
    confounders: Your confounder matrix (numpy array or pandas DataFrame)
    data['a']: Exposure data (numpy array or pandas Series)
    data['y']: Outcome data (numpy array or pandas Series)
    fit_a: The fitted model object for the binary exposure (statsmodels object)
    fit_y: The fitted model object for the outcome (statsmodels object)
    truth_est[2]: The true estimate (scalar)
    '''

    # Step 1: Call the piie.sp.2.variance.function.cont equivalent function
    # Assuming `out_sp_2` returns a list or array with [PSI, PIIE, Var PIIE]
    out_sp_2 = piie_sp_2_variance_function_cont(confounders, data['a'].values, data['y'].values, np.array([1, 1, 1, 1]), fit_a, fit_y, 0, 1)

    # Step 2: Calculate proportional bias
    truth_est_3 = truth_est[2]  # Assuming truth_est is a list or array
    prop_bias_sp_2 = (out_sp_2["PIIE.2"] - truth_est_3) / truth_est_3  # This calculates (PIIE - truth.est[3]) / truth.est[3]


    # Step 3: Calculate confidence interval
    ci_low = out_sp_2["PIIE.2"] - norm.ppf(0.975) * np.sqrt(out_sp_2["Var PIIE.2"])  # Lower bound of the CI
    ci_up = out_sp_2["PIIE.2"] + norm.ppf(0.975) * np.sqrt(out_sp_2["Var PIIE.2"])  # Upper bound of the CI

    # Step 4: Check if truth.est[3] is within the confidence interval
    ci_coverage_sp_2 = int(truth_est_3 >= ci_low and truth_est_3 <= ci_up)  # This returns 1 if covered, otherwise 0

    # Step 5: Append the calculated values (prop_bias_sp_2 and ci_coverage_sp_2) to the out_sp_2 array
    out_sp_2 = np.append(out_sp_2, [prop_bias_sp_2, ci_coverage_sp_2])

    # Print or return the final result
    print(out_sp_2)

    '''
    DML implementation
    '''

    # Call the variance function with the appropriate arguments
    out_sp = piie_sp_variance_function_cont(confounders, data['a'].values, data['m'].values, data['y'].values,
                                            np.array([1, 1, 1, 1]), np.array([1, 1, 1, 0]),
                                            np.array([1, 1, 1, 1]), fit_a, fit_z, fit_y, 0, 1)

    """
    ["PSI", "PIIE", "Var PIIE"]
    """
    # Proportional Bias Calculation
    prop_bias_sp = (out_sp["PIIE"] - truth_est[2]) / truth_est[2]

    # Calculate Confidence Interval (CI)
    ci_low = out_sp["PIIE"] - norm.ppf(0.975) * np.sqrt(out_sp["Var PIIE"])
    ci_up = out_sp["PIIE"] + norm.ppf(0.975) * np.sqrt(out_sp["Var PIIE"])

    # Coverage of the confidence interval
    ci_coverage_sp = 1 if truth_est[2] >= ci_low and truth_est[2] <= ci_up else 0

    # Combine the results into a single array
    out_sp = np.hstack([out_sp, prop_bias_sp, ci_coverage_sp])

    print(out_sp)

    # Calculate mean of y (outcome)
    mean_y = np.mean(data['y'])

    print(mean_y)
    # Return or store the final result
    results = [out_mle, out_sp_1, out_sp_2, out_sp, mean_y]

    print("-------------------------------------------------")
    print("Now we check the results")
    print("MLE PIIE", out_mle['PIIE'][0])
    print("SP 1 PIIE", out_sp_1[0]['PIIE.1'])
    print("SP 2 PIIE", out_sp_2[0]['PIIE.2'])
    print("SP PIIE", out_sp[0]['PIIE'])
    print("Mean y", mean_y)
    print("-------------------------------------------------")

def run_experiment(sample_size, num_experiments):
    out_mle_piie = []
    out_sp_1_piie = []
    out_sp_2_piie = []
    out_sp_piie = []
    mean_Y = []

    chi_square_mle = []
    chi_square_sp_1 = []
    chi_square_sp_2 = []
    chi_square_sp = []

    mse_yyhat_dr = []
    for _ in range(num_experiments):
        print("sample_size", sample_size)

        # Generate the dataset
        data = gen_med_data_continuous(sample_size, p_c1, p_c3, alpha, omega, beta, theta, sigma_m, sigma_y)
        print(data)

        # Fit linear model for the mediator (m) ~ a + c1 + c2 + (c1 * c2)
        fit_z = smf.ols('m ~ a + c1 + c2 + I(c1*c2)', data=data).fit()

        # Fit linear model for the outcome (y) ~ a + m + (a * m) + c1 + c2 + (c1 * c2) + c3
        fit_y = smf.ols('y ~ a + m + I(a*m) + c1 + c2 + I(c1*c2) + c3', data=data).fit()

        # Fit logistic model for the binary exposure a ~ c1 + c2 + (c1 * c2) + c3
        fit_a = smf.logit('a ~ c1 + c2 + I(c1*c2) + c3', data=data).fit()

        # Extracting the coefficients
        alpha_hat = fit_a.params.values  # Coefficients from logistic regression
        beta_hat = np.append(fit_z.params.values, 0)  # Coefficients from linear model for m (add 0 for consistency)
        theta_hat = fit_y.params.values  # Coefficients from linear model for y

        # Create the confounders matrix (c1, c2, c1 * c2, c3)
        confounders = np.c_[data.iloc[:, 3],  # c1
        data.iloc[:, 4],  # c2
        data.iloc[:, 3] * data.iloc[:, 4],  # c1 * c2
        data.iloc[:, 5]]  # c3

        # Output for coefficients (optional)
        print("Alpha hat (logit model coefficients):", alpha_hat)
        print("Beta hat (linear model coefficients):", beta_hat)
        print("Theta hat (linear model coefficients):", theta_hat)

        '''
        ####  Calculate PSI MLE

        # Assuming the following inputs:
        # confounders: Your confounder matrix (numpy array or pandas DataFrame)
        # data['a']: Exposure data (numpy array or pandas Series)
        # theta_hat: Estimated theta parameters (numpy array)
        # beta_truth: True beta values (numpy array)
        # alpha_hat: Estimated alpha parameters (numpy array)
        # truth_est[3]: The true estimate (scalar)
        # fit_z, fit_y: The fitted model objects
        '''

        # Call the PIIE MLE variance function
        out_mle = piie_mle_variance_function_cont(confounders, data['a'].values, theta_hat, beta, alpha_hat, 0, 1,
                                                  fit_z, fit_y)

        # Calculate proportional bias (same as in R code)
        prop_bias_mle = (out_mle['PIIE'][0] - truth_est[2]) / truth_est[2]

        # Compute the 95% confidence interval
        ci_low = out_mle['PIIE'][0] - norm.ppf(0.975) * np.sqrt(out_mle['Var PIIE'][0])
        ci_up = out_mle['PIIE'][0] + norm.ppf(0.975) * np.sqrt(out_mle['Var PIIE'][0])

        # Determine if the true value is within the CI (coverage)
        ci_coverage_mle = int(truth_est[3] >= ci_low and truth_est[3] <= ci_up)

        # Append the results to the output (mimics the cbind operation in R)
        out_mle['Prop Bias MLE'] = prop_bias_mle
        out_mle['CI Coverage MLE'] = ci_coverage_mle

        # Display the final output
        print(out_mle)

        '''
        ### Calculate PSI SP 1 

        Assuming the following inputs:
        data['m']: Mediator data (numpy array or pandas Series)
        data['y']: Outcome data (numpy array or pandas Series)
        fit_z: The fitted model object for the mediator (statsmodels object)
        truth_est[2]: The true estimate (scalar)
        '''

        # SP 1 - Call the function for variance estimation
        out_sp_1 = piie_sp_1_variance_function_cont(data['m'].values, data['y'].values, fit_z, 0, 1)

        # Calculate proportion bias
        prop_bias_sp_1 = (out_sp_1['PIIE.1'] - truth_est[2]) / truth_est[2]

        # Confidence interval calculation (similar to qnorm(.975) in R, which is 1.96)
        ci_low = out_sp_1['PIIE.1'] - norm.ppf(0.975) * np.sqrt(out_sp_1['Var PIIE.1'])
        ci_up = out_sp_1['PIIE.1'] + norm.ppf(0.975) * np.sqrt(out_sp_1['Var PIIE.1'])

        # Coverage check
        ci_coverage_sp_1 = int(truth_est[2] >= ci_low and truth_est[2] <= ci_up)

        # Combine the results into a final output
        out_sp_1 = np.append(out_sp_1, [prop_bias_sp_1, ci_coverage_sp_1])

        # Assuming out_sp_1 is now a NumPy array, you can convert it to a DataFrame if needed
        # or use it directly.
        print(out_sp_1)

        ''' 
        ### Calculate PSI SP 2   

        Assuming the following inputs:
        confounders: Your confounder matrix (numpy array or pandas DataFrame)
        data['a']: Exposure data (numpy array or pandas Series)
        data['y']: Outcome data (numpy array or pandas Series)
        fit_a: The fitted model object for the binary exposure (statsmodels object)
        fit_y: The fitted model object for the outcome (statsmodels object)
        truth_est[2]: The true estimate (scalar)
        '''

        # Step 1: Call the piie.sp.2.variance.function.cont equivalent function
        # Assuming `out_sp_2` returns a list or array with [PSI, PIIE, Var PIIE]
        out_sp_2 = piie_sp_2_variance_function_cont(confounders, data['a'].values, data['y'].values,
                                                    np.array([1, 1, 1, 1]), fit_a, fit_y, 0, 1)

        # Step 2: Calculate proportional bias
        truth_est_3 = truth_est[2]  # Assuming truth_est is a list or array
        prop_bias_sp_2 = (out_sp_2[
                              "PIIE.2"] - truth_est_3) / truth_est_3  # This calculates (PIIE - truth.est[3]) / truth.est[3]

        # Step 3: Calculate confidence interval
        ci_low = out_sp_2["PIIE.2"] - norm.ppf(0.975) * np.sqrt(out_sp_2["Var PIIE.2"])  # Lower bound of the CI
        ci_up = out_sp_2["PIIE.2"] + norm.ppf(0.975) * np.sqrt(out_sp_2["Var PIIE.2"])  # Upper bound of the CI

        # Step 4: Check if truth.est[3] is within the confidence interval
        ci_coverage_sp_2 = int(truth_est_3 >= ci_low and truth_est_3 <= ci_up)  # This returns 1 if covered, otherwise 0

        # Step 5: Append the calculated values (prop_bias_sp_2 and ci_coverage_sp_2) to the out_sp_2 array
        out_sp_2 = np.append(out_sp_2, [prop_bias_sp_2, ci_coverage_sp_2])

        # Print or return the final result
        print(out_sp_2)

        '''
        DML implementation
        '''

        # Call the variance function with the appropriate arguments
        out_sp = piie_sp_variance_function_cont(confounders, data['a'].values, data['m'].values, data['y'].values,
                                                np.array([1, 1, 1, 1]), np.array([1, 1, 1, 0]),
                                                np.array([1, 1, 1, 1]), fit_a, fit_z, fit_y, 0, 1)

        """
        ["PSI", "PIIE", "Var PIIE"]
        """
        # Proportional Bias Calculation
        prop_bias_sp = (out_sp["PIIE"] - truth_est[2]) / truth_est[2]

        # Calculate Confidence Interval (CI)
        ci_low = out_sp["PIIE"] - norm.ppf(0.975) * np.sqrt(out_sp["Var PIIE"])
        ci_up = out_sp["PIIE"] + norm.ppf(0.975) * np.sqrt(out_sp["Var PIIE"])

        # Coverage of the confidence interval
        ci_coverage_sp = 1 if truth_est[2] >= ci_low and truth_est[2] <= ci_up else 0

        # Combine the results into a single array
        out_sp = np.hstack([out_sp, prop_bias_sp, ci_coverage_sp])

        print(out_sp)

        # Calculate mean of y (outcome)
        mean_y = np.mean(data['y'])

        print(mean_y)
        # Return or store the final result
        results = [out_mle, out_sp_1, out_sp_2, out_sp, mean_y]

        print("-------------------------------------------------")
        print("Now we check the results")
        print("MLE PIIE", out_mle['PIIE'][0])
        print("SP 1 PIIE", out_sp_1[0]['PIIE.1'])
        print("SP 2 PIIE", out_sp_2[0]['PIIE.2'])
        print("SP PIIE", out_sp[0]['PIIE'])
        print("Mean y", mean_y)
        print("-------------------------------------------------")
        out_mle_piie.append(out_mle['PIIE'][0])
        out_sp_1_piie.append(out_sp_1[0]['PIIE.1'])
        out_sp_2_piie.append(out_sp_2[0]['PIIE.2'])
        out_sp_piie.append(out_sp[0]['PIIE'])
        mean_Y.append(mean_y)

        print("Chi-square MLE", chi_square_distance_gaussians(mu1=-4.6056, sigma1=1, mu2=out_mle['PIIE'][0], sigma2=1, num_samples=sample_size, num_bins=50))

        chi_square_mle.append(chi_square_distance_gaussians(mu1=-4.6056, sigma1=1, mu2=out_mle['PIIE'][0], sigma2=1, num_samples=sample_size, num_bins=50))
        chi_square_sp_1.append(chi_square_distance_gaussians(mu1=-4.6056, sigma1=1, mu2=out_sp_1[0]['PIIE.1'], sigma2=1, num_samples=sample_size, num_bins=50))
        chi_square_sp_2.append(chi_square_distance_gaussians(mu1=-4.6056, sigma1=1, mu2=out_sp_2[0]['PIIE.2'], sigma2=1, num_samples=sample_size, num_bins=50))
        chi_square_sp.append(chi_square_distance_gaussians(mu1=-4.6056, sigma1=1, mu2=out_sp[0]['PIIE'], sigma2=1, num_samples=sample_size, num_bins=50))


        print('truth', truth_est[2])
        print('out_sp', out_sp[0]['PIIE'])
        print("mse_yyhat_dr", (truth_est[2]  - out_sp[0]['PIIE'])**2)
        mse = (truth_est[2]  - out_sp[0]['PIIE'])**2
        mse_yyhat_dr.append(mse)
        return out_mle_piie, out_sp_1_piie, out_sp_2_piie, out_sp_piie, mean_y, chi_square_mle, chi_square_sp_1, chi_square_sp_2, chi_square_sp, mse_yyhat_dr


if __name__ == '__main__':
    sample_sizes = [100, 300,  500, 700,  900, 1100, 1300, 1500]
    num_experiments = 100

    out_mle_piie_all = []
    out_sp_1_piie_all = []
    out_sp_2_piie_all = []
    out_sp_piie_all = []

    chi_square_mle_all = []
    chi_square_sp_1_all = []
    chi_square_sp_2_all = []
    chi_square_sp_all = []

    mse_yyhat_dr_all = []

    mean_Y = []

    for size in sample_sizes:
        out_mle_piie, out_sp_1_piie, out_sp_2_piie, out_sp_piie, mean_y, chi_square_mle, chi_square_sp_1, chi_square_sp_2, chi_square_sp, mse_yyhat_dr  =run_experiment(size, num_experiments)

        out_mle_piie_all.append(out_mle_piie)
        out_sp_1_piie_all.append(out_sp_1_piie)
        out_sp_2_piie_all.append(out_sp_2_piie)
        out_sp_piie_all.append(out_sp_piie)
        mean_Y.append(mean_y)

        chi_square_mle_all.append(chi_square_mle)
        chi_square_sp_1_all.append(chi_square_sp_1)
        chi_square_sp_2_all.append(chi_square_sp_2)
        chi_square_sp_all.append(chi_square_sp)

        mse_yyhat_dr_all.append(mse_yyhat_dr)


    # Flatten the list using list comprehension
    chi_square_sp_all_flat = [item[0] for item in chi_square_sp_all]
    mse_yyhat_dr_all_flat = [item[0] for item in mse_yyhat_dr_all]

    print('chi square SP all', chi_square_sp_all_flat)
    print('mse_yyhat_dr_all', mse_yyhat_dr_all_flat)

    # Create the scatter plot
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(chi_square_sp_all, mse_yyhat_dr_all, c=sample_sizes, cmap='viridis', s=100)
    # Add labels and title
    plt.xlabel('Chi Square SP All')
    plt.ylabel('MSE yyhat DR All')
    plt.title('Scatter Plot of Chi Square SP All vs MSE yyhat DR All')

    # Add colorbar for the sample sizes
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Sample Size')
    sns.lineplot(x=chi_square_sp_all_flat, y=mse_yyhat_dr_all_flat, label='Lineplot', color='blue', marker='o')
    # Regression plot for fitting a regression line
    sns.regplot(x=chi_square_sp_all_flat, y=mse_yyhat_dr_all_flat, scatter=False, label='Regression Line', color='red')

    # Display the plot
    plt.show()
    '''
    # Create a DataFrame for the chi-square results
    chi_square_df = pd.DataFrame({
        'Sample Size': np.repeat(sample_sizes, num_experiments),
        'Double Robust': np.concatenate(chi_square_sp_all),
        'Propensity Score': np.concatenate(chi_square_sp_1_all),
        'Outcome Regression': np.concatenate(chi_square_sp_2_all),
        'MSE error': np.concatenate(mse_yyhat_dr_all)
    })

    # Melt the DataFrame for plotting
    chi_square_melted = chi_square_df.melt(id_vars=['Sample Size'], value_vars=['Double Robust', 'Propensity Score', 'Outcome Regression'],
                                           var_name='Method', value_name='Chi-Square Distance')

    # Plot the results
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Plot the data
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=mean_values, x='Double Robust', y='MSE error', hue='Sample Size', style='Sample Size', s=100,
                    markers=['o', 's', 'D'])
    sns.lineplot(data=mean_values, x='Double Robust', y='MSE error', hue='Sample Size', markers=True, dashes=False)
    sns.regplot(data=mean_values, x='Double Robust', y='MSE error', scatter=False, color='grey',
                line_kws={"linestyle": "--"})

    plt.title('Mean MSE error vs Mean Double Robust')
    plt.xlabel('Mean Double Robust')
    plt.ylabel('Mean MSE error')
    plt.grid(True)
    plt.show()
    '''