import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def log_likelihood_and_grad(y, X, w, beta):
    hplane = X.T @ w + beta * np.ones_like(y)
    sig = 1. / (1. + np.exp(-hplane))
    log_likelihood = -np.dot(1.-y, hplane) + np.sum(np.log(sig))
    grad_w = np.sum(X @ np.diag(y - sig), axis=1)
    grad_beta = np.sum(y - sig)
    return -log_likelihood, (-grad_w, -grad_beta)

def steepest_descent_with_momentum(y, X, a, gamma, tol):
    w = np.zeros(X.shape[0])
    beta = 0.
    d_w = np.zeros_like(w)
    d_beta = 0
    log_likelihood, (grad_w, grad_beta) = log_likelihood_and_grad(y, X, w, beta)
    grad_norm = np.sqrt(np.linalg.norm(grad_w)**2. + grad_beta**2.)
    print(f"Initial log likelihood: {log_likelihood}")
    print(f"Initial gradient norm: {grad_norm}")
    while grad_norm > tol:
        log_likelihood, (grad_w, grad_beta) = log_likelihood_and_grad(y, X, w, beta)
        grad_norm = np.sqrt(np.linalg.norm(grad_w)**2. + grad_beta**2.)
        d_w = a * d_w - gamma * grad_w
        d_beta = a * d_beta - gamma * grad_beta
        w += d_w
        beta += d_beta
        #print(f"Log likelihood: {log_likelihood}")
        #print(f"Gradient norm: {grad_norm}")
    print(f"Final log likelihood: {log_likelihood}")
    print(f"Final gradient norm: {grad_norm}")
    return w, beta


