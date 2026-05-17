import math
import warnings
import torch
from torch import Tensor
from typing import Optional
from scipy.stats import norm
from scipy.optimize import fsolve, brentq
from scipy.integrate import quad
import numpy as np

# The order of each parameter
# mu_y, sigma_y, sigma, kappa, lmbda, theta, p_init, T


def _steady_state_equations(params, sigma, kappa, theta):
    """
    Define steady-state equations
    params[0] = mu_y (mean)
    params[1] = sigma_y (standard deviation)
    """
    mu_y, sigma_y = params

    # if sigma \le 0 or negative, return large residuals
    if sigma_y <= 0:
        return [1e10, 1e10]

    # calculate standardized threshold
    alpha = (theta - mu_y) / sigma_y

    # standard normal distribution PDF and CDF
    phi_alpha = norm.pdf(alpha)
    Phi_alpha = norm.cdf(alpha)

    # Equation 1: Mean balance equation
    eq1 = mu_y - kappa * (mu_y * Phi_alpha - sigma_y * phi_alpha)

    # Equation 2: Variance balance equation
    E_y2 = mu_y**2 + sigma_y**2
    E_y2_I = E_y2 * Phi_alpha - sigma_y * (theta + mu_y) * phi_alpha
    eq2 = E_y2 - (kappa**2 * E_y2_I + sigma**2)

    return [eq1, eq2]


def _find_steady_state(sigma, kappa, theta, initial_guess=None):
    """
    finding the steady-state distribution's mean and variance
    
    parameters:
    sigma: noise standard deviation
    kappa: decay factor
    theta: threshold
    initial_guess: initial guess [mu_y, sigma_y]
    
    returns:
    mu_y: steady-state mean
    sigma_y: steady-state standard deviation
    success: whether the solution converged
    """
    if initial_guess is None:
        # Use AR(1) approximation as initial guess
        sigma_y_guess = sigma / np.sqrt(1 - kappa**2)
        initial_guess = [0, sigma_y_guess]

    # Use fsolve to solve the nonlinear system of equations
    result = fsolve(_steady_state_equations, initial_guess, args=(sigma, kappa, theta),
                    full_output=True)

    mu_y, sigma_y = result[0]
    success = result[2] == 1  # Check if converged

    return mu_y, sigma_y, success


def _find_sigma(kappa, theta, p_init, sigma_low=1e-5, sigma_high=10.0, tol=1e-5, max_k=1000):
    """
    binary search q(p_init*sigma^2) = p_init
    """
    for _ in range(max_k):
        sigma_mid = (sigma_low + sigma_high) / 2
        mu_y, sigma_y, success = _find_steady_state(sigma_mid * np.sqrt(p_init), kappa, theta)
        if not success:
            raise RuntimeError("Steady-state solver did not converge")
        q = norm.sf((theta - mu_y) / sigma_y)
        if abs(q - p_init) < tol:
            return sigma_mid
        if q < p_init:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    raise RuntimeError("Failed to converge to a solution for sigma")


def _P_static(sigma, kappa, theta, tol=1e-6, max_k=100):
    """
    calculate firing rate of direct coding layer
    """
    if sigma <= 0:
        return 0.0

    term1 = norm.sf(theta / sigma)

    total_sum = 0.0
    k = 1
    S_prev = 1.0  # S0
    S_curr = (1 - kappa**2) / (1 - kappa)  # S1

    while k <= max_k:
        a_prev = theta / (sigma * S_prev)
        a_curr = theta / (sigma * S_curr)
        term = (1.0 / (k + 1)) * (norm.cdf(a_prev) - norm.cdf(a_curr))
        total_sum += term

        # Stop summation if the term is sufficiently small
        if abs(term) < tol:
            break
        S_prev = S_curr
        k += 1
        S_curr = (1 - kappa**(k + 1)) / (1 - kappa)

    return term1 + total_sum


def _find_static_sigma(kappa, theta, p_init, sigma_low=1e-5, sigma_high=10.0, tol=1e-6):
    """
    Given kappa, theta, and P_target, solve for sigma_static.
    """

    # objective function
    def f(sigma):
        return _P_static(sigma, kappa, theta, tol=tol) - p_init

    # Use Brentq method to find root
    sigma_sol = brentq(f, sigma_low, sigma_high, xtol=tol)
    return sigma_sol


def calculate_parameters(kappa: float = 0.5, lmbda: float = 0.5, theta: float = 1.0,
                         p_init: float = 0.1):
    """
    Calculate initialization parameters for spiking neural networks.
    
    parameters:
    kappa: decay factor
    lmbda: input decay factor
    theta: threshold
    p_init: initial firing rate
    
    returns:
    sigma: noise standard deviation for spiking initialization
    sigma_static: noise standard deviation for static layer initialization
    mu_y: mean of membrane potential of the spiking neuron
    sigma_y: standard deviation of membrane potential of the spiking neuron
    """
    print("Calculating initialization parameters...")
    sigma = _find_sigma(kappa, theta, p_init) / lmbda
    sigma_x = sigma * np.sqrt(p_init) * lmbda
    mu_y, sigma_y, success = _find_steady_state(sigma_x, kappa, theta)
    if not success:
        raise RuntimeError("Failed to converge to a solution for steady-state parameters.")
    sigma_static = _find_static_sigma(kappa, theta, p_init) / lmbda
    print(f"initialization parameters: sigma={sigma}, sigma_static={sigma_static}, "
          f"mu_y={mu_y}, sigma_y={sigma_y}")
    return sigma, sigma_static, mu_y, sigma_y


def _simulate(sigma, kappa, theta, T, N=1000000):
    """
    simulate membrane potential by Monte Carlo method
    """
    y = np.zeros(N)
    for _ in range(T):
        x = np.random.normal(0, sigma, N)
        mask_accumulate = y < theta
        y = np.where(mask_accumulate, kappa * y + x, x)
    return y


def _find_sigma_sim(kappa, theta, p_init, T, sigma_low=1e-5, sigma_high=10.0, tol=1e-5, max_k=100):
    """
    binary search q(p_init*sigma^2) = p_init using simulation
    """
    for _ in range(max_k):
        sigma_mid = (sigma_low + sigma_high) / 2
        y = _simulate(sigma_mid * np.sqrt(p_init), kappa, theta, T)
        q = np.sum(y >= theta) / len(y)
        if abs(q - p_init) < tol or abs(sigma_high - sigma_low) < tol:
            return sigma_mid
        if q < p_init:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    raise RuntimeError("Failed to converge to a solution for sigma")


def _P_static_sim(sigma, kappa, theta, T, N=1000000):
    """
    calculate firing rate of direct coding layer by simulation
    """
    y = np.zeros(N)
    x = np.random.normal(0, sigma, N)
    for _ in range(T):
        mask_accumulate = y < theta
        y = np.where(mask_accumulate, kappa * y + x, x)
    return np.sum(y >= theta) / N


def _find_static_sigma_sim(kappa, theta, p_init, T, sigma_low=1e-5, sigma_high=10.0, tol=1e-6,
                           max_k=100):
    for _ in range(max_k):
        sigma_mid = (sigma_low + sigma_high) / 2
        p = _P_static_sim(sigma_mid, kappa, theta, T)
        if abs(p - p_init) < tol or abs(sigma_high - sigma_low) < tol:
            return sigma_mid
        if p < p_init:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    raise RuntimeError("Failed to converge to a solution for sigma_static")


def calculate_parameters_sim(kappa: float = 0.5, lmbda: float = 0.5, theta: float = 1.0,
                             p_init: float = 0.1, T: int = 4):
    """
    Calculate initialization parameters for spiking neural networks using simulation.
    
    parameters:
    kappa: decay factor
    lmbda: input decay factor
    theta: threshold
    p_init: initial firing rate
    T: number of time steps
    
    returns:
    sigma: noise standard deviation for spiking initialization
    sigma_static: noise standard deviation for static layer initialization
    mu_y: mean of membrane potential of the spiking neuron
    sigma_y: standard deviation of membrane potential of the spiking neuron
    """
    print("Calculating initialization parameters using simulation...")
    sigma = _find_sigma_sim(kappa, theta, p_init, T) / lmbda
    sigma_x = sigma * np.sqrt(p_init) * lmbda
    y = _simulate(sigma_x, kappa, theta, T)
    mu_y = np.mean(y)
    sigma_y = np.std(y)
    sigma_static = _find_static_sigma_sim(kappa, theta, p_init, T) / lmbda
    print(f"initialization parameters: sigma={sigma}, sigma_static={sigma_static}, "
          f"mu_y={mu_y}, sigma_y={sigma_y}")
    return sigma, sigma_static, mu_y, sigma_y


def calculate_alpha(mu_y, sigma_y, sigma, kappa, lmbda, theta, p_init):
    """
    Calculates alpha such that E[f(x)] matches the target_expectation.
    This is a universal method that calculates expectation using numerical integration.
    For other surrogate functions, modify the integrand accordingly.
    
    Parameters:
        mu_y (float): Mean of the normal distribution Y.
        sigma_y (float): Standard deviation of the normal distribution Y.
        theta (float): Parameter theta in the function f(x).
        target_expectation (float): The desired expectation value.
        
    Returns:
        float: The calculated value of alpha.
    """

    target_expectation = (1.0 - (1.0 - p_init) * kappa**2) / ((sigma * lmbda)**2)

    # The integrand function: f(x) * pdf(x)
    # f(x) = (alpha * exp(-2*alpha*|x-theta|))^2 = alpha^2 * exp(-4*alpha*|x-theta|)
    def integrand(x, alpha):
        fx = (alpha**2) * np.exp(-4 * alpha * np.abs(x - theta))
        pdf = norm.pdf(x, loc=mu_y, scale=sigma_y)
        return fx * pdf

    # The objective function to find the root for: E[f(x)] - target = 0
    def objective(alpha):
        if alpha <= 0:
            return -target_expectation
        expectation, error = quad(integrand, -np.inf, np.inf, args=(alpha, ))
        return expectation - target_expectation

    if target_expectation <= 0:
        raise ValueError("Target expectation must be positive for this squared function.")

    # 1. Find a valid bracket for the root finder
    # We sweep exponentially to find a range [a, b] where the sign changes.
    # Expectation generally increases with alpha for this function setup.
    a, b = 1e-4, 1.0
    val_a = objective(a)

    # Expand upper bound until we cross 0 (expectation > target)
    for _ in range(20):  # Safety limit
        val_b = objective(b)
        if val_a * val_b < 0:
            break
        a = b
        val_a = val_b
        b *= 10
    else:
        raise ValueError(
            "Could not find a valid bracket for alpha. Target expectation might be too high.")

    # 2. Solve for alpha using Brent's method
    alpha_sol = brentq(objective, a, b)
    print(f"Calculated alpha: {alpha_sol}")

    return alpha_sol


def _compute_M2_numerical(y_samples, theta, alpha):
    """
    Computes the expected squared derivative M2 using numerical samples.
    """
    # Function: (alpha * exp(-2*alpha*|y - theta|))^2
    squared_derivatives = (alpha**2) * np.exp(-4 * alpha * np.abs(y_samples - theta))
    M2_est = np.mean(squared_derivatives)
    return M2_est


def calculate_alpha_sim(sigma, kappa, lmbda, theta, p_init, T, N=1000000):
    """
    Calculates alpha such that E[f(x)] matches the target_expectation using simulation.
    This is a universal method that calculates expectation using numerical samples.
    For other surrogate functions, modify the M2 accordingly.
    
    Parameters:
        mu_y (float): Mean of the normal distribution Y.
        sigma_y (float): Standard deviation of the normal distribution Y.
        theta (float): Parameter theta in the function f(x).
        target_expectation (float): The desired expectation value.
    """
    target_expectation = 0
    for t in range(T):
        target_expectation += (1.0 - math.pow(
            (1.0 - p_init) * kappa**2, t + 1)) / (1.0 - (1.0 - p_init) * kappa**2) * (
                (sigma * lmbda)**2) / T
    target_expectation = 1.0 / target_expectation
    y_samples = _simulate(sigma * np.sqrt(p_init) * lmbda, kappa, theta, T=T, N=N)

    # The objective function to find the root for: E[f(x)] - target = 0
    def objective(alpha):
        if alpha <= 0:
            return -target_expectation
        M2_est = _compute_M2_numerical(y_samples, theta, alpha)
        return M2_est - target_expectation

    if target_expectation <= 0:
        raise ValueError("Target expectation must be positive for this squared function.")

    # 1. Find a valid bracket for the root finder
    a, b = 1e-4, 1.0
    val_a = objective(a)

    # Expand upper bound until we cross 0 (expectation > target)
    for _ in range(20):  # Safety limit
        val_b = objective(b)
        if val_a * val_b < 0:
            break
        a = b
        val_a = val_b
        b *= 10
    else:
        raise ValueError(
            "Could not find a valid bracket for alpha. Target expectation might be too high.")

    # 2. Solve for alpha using Brent's method
    alpha_sol = brentq(objective, a, b)
    print(f"Calculated alpha (simulation): {alpha_sol}")

    return alpha_sol


def _calculate_fan_in(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    # num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    # fan_out = num_output_fmaps * receptive_field_size

    return fan_in


def spiking_normal_(
    tensor: Tensor,
    sigma: float = 1.0,
    generator: Optional[torch.Generator] = None,
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_fan_in(tensor)
    std = sigma / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)


## ablation


def ding_normal_(
    tensor: Tensor,
    lmbda: float = 0.5,
    theta: float = 1.0,
    generator: Optional[torch.Generator] = None,
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_fan_in(tensor)
    gain = math.sqrt(2.0) * theta / lmbda
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)


def fluctuation_normal_(
    tensor: Tensor,
    kappa: float = 0.5,
    lmbda: float = 1.0,
    theta: float = 1.0,
    xi: float = 2.0,
    p_init: float = 0.1,
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_fan_in(tensor)
    gain = theta * math.sqrt(1 - math.pow(kappa, 2)) / (lmbda * math.sqrt(p_init) * xi)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def micheli_normal_(
    tensor: Tensor,
    theta: float = 1.0,
    generator: Optional[torch.Generator] = None,
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_fan_in(tensor)
    gain = 1.0 / math.sqrt(norm.sf(theta))
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)


if __name__ == "__main__":
    # Example usage
    kappa = 0.5
    lmbda = 0.5
    theta = 0.5
    p_init = 0.1
    T = 10

    sigma, sigma_static, mu_y, sigma_y = calculate_parameters(kappa, lmbda, theta, p_init)
    alpha = calculate_alpha(mu_y, sigma_y, sigma, kappa, lmbda, theta, p_init)

    sigma_sim, sigma_static_sim, mu_y_sim, sigma_y_sim = calculate_parameters_sim(
        kappa, lmbda, theta, p_init, T)
    alpha_sim = calculate_alpha_sim(sigma_sim, kappa, lmbda, theta, p_init, T)
