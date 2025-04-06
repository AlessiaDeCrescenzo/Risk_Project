import numpy as np
from scipy.optimize import fsolve
from scipy.stats import rv_discrete


def equations(vars, mu, cv, gamma1):
    a, b = vars
    c = 3 * mu - a - b  # Express c in terms of a and b
    
    # Compute variance
    var = (a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18
    sigma = np.sqrt(var)
    
    # Compute coefficient of variation
    cv_calc = sigma / mu
    
    # Compute skewness
    num = np.sqrt(2) * (a + b - 2*c) * (2*a - b - c) * (a - 2*b + c)
    denom = 5 * (var ** 1.5)
    gamma1_calc = num / denom
    
    return [cv_calc - cv, gamma1_calc - gamma1]

def find_discrete_triangular(mu, cv, gamma1, a_guess, b_guess):
    solution = fsolve(equations, (a_guess, b_guess), args=(mu, cv, gamma1))
    a, b = solution
    c = 3 * mu - a - b
    
    #Ensure integer values and correct ordering
    a, b, c = sorted([round(a), round(b), round(c)])
    
    return a, c, b


def discrete_triangular_rv(mean, cv, gamma1, a1_guess= 10, a2_guess= 20):
    """Creates an asymmetric discrete triangular distribution using the correct PMF."""
    a1,m, a2 = find_discrete_triangular(mean, cv, gamma1, a1_guess, a2_guess)
    values = np.arange(m - a1, m + a2 + 1)  # Support of the distribution

    # Compute PMF using the exact formula from the definition
    normalization_factor = (a1 + a2 + 2) / 2
    probs = np.array([
        (1 - (m - y) / (a1 + 1)) / normalization_factor if y < m else
        (1 - (y - m) / (a2 + 1)) / normalization_factor
        for y in values
    ])
    
    probs /= np.sum(probs)  # Normalize to ensure the total probability is 1

    # Safety check
    assert np.isclose(np.sum(probs), 1), "PMF probabilities do not sum to 1!"
    
    return rv_discrete(name='asym_discrete_triangular', values=(values, probs))