"""
Utility functions for MAML-LQR implementation.
Based on patterns from legacy utils_leo.py but adapted for new environment structure.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import time


def check_stability(A: np.ndarray, B: np.ndarray, K: np.ndarray) -> bool:
    """
    Check if the closed-loop system (A - B*K) is stable.

    Parameters:
    A (np.ndarray): State transition matrix
    B (np.ndarray): Control matrix
    K (np.ndarray): Feedback gain matrix

    Returns:
    bool: True if stable (all eigenvalues < 1), False otherwise
    """
    A_cl = A - B @ K
    eigenvals = np.linalg.eigvals(A_cl)
    max_eig = np.max(np.abs(eigenvals))
    return max_eig < 1.0


def generate_perturbations(shape: Tuple[int, ...], num_samples: int,
                         perturbation_scale: float, normalize: bool = True) -> np.ndarray:
    """
    Generate normalized random perturbations for gradient estimation.

    Parameters:
    shape: Shape of perturbation matrices
    num_samples: Number of perturbation samples
    perturbation_scale: Scale of perturbations (r parameter)
    normalize: Whether to normalize perturbations to unit sphere

    Returns:
    np.ndarray: Array of perturbations with shape (num_samples, *shape)
    """
    # Generate random perturbations
    U = np.random.normal(0, 1, (num_samples, *shape))

    if normalize:
        # Normalize each perturbation to unit sphere, then scale
        U_norms = np.linalg.norm(U, axis=tuple(range(1, len(U.shape))), keepdims=True)
        U = (U / U_norms) * perturbation_scale
    else:
        U = U * perturbation_scale

    return U


def zeroth_order_gradient(env, K: np.ndarray, perturbations: np.ndarray,
                         perturbation_scale: float, rollout_length: int = None,
                         num_trajectories: int = None, use_analytical: bool = True) -> np.ndarray:
    """
    Estimate policy gradient using zeroth-order (finite difference) method.

    Parameters:
    env: Environment instance (Boeing_dynamics or BaseLQREnv)
    K: Current policy parameters (gain matrix)
    perturbations: Array of perturbation directions
    perturbation_scale: Perturbation magnitude (r)
    rollout_length: Length of rollouts for cost estimation (unused if use_analytical=True)
    num_trajectories: Number of trajectories per policy evaluation (unused if use_analytical=True)
    use_analytical: Whether to use analytical cost formula (much faster)

    Returns:
    np.ndarray: Estimated gradient
    """
    state_dim, action_dim = K.shape[1], K.shape[0]
    gradient = np.zeros_like(K)

    valid_samples = 0

    for i, U_i in enumerate(perturbations):
        # Perturbed policies
        K_plus = K + U_i
        K_minus = K - U_i

        if use_analytical and hasattr(env, 'analytical_cost'):
            # Use analytical cost computation
            cost_plus = env.analytical_cost(K_plus)
            cost_minus = env.analytical_cost(K_minus)

            # Skip if either cost is infinite (unstable)
            if cost_plus == float('inf') or cost_minus == float('inf'):
                continue
        else:
            # Fallback to rollout-based estimation
            # Check stability (optional - can be disabled for speed)
            if not (check_stability(env.A, env.B, K_plus) and
                    check_stability(env.A, env.B, K_minus)):
                continue

            # Estimate costs with multiple trajectories
            cost_plus = 0
            cost_minus = 0

            # Use defaults if not provided
            _rollout_length = rollout_length or 50
            _num_trajectories = num_trajectories or 1

            for _ in range(_num_trajectories):
                # Cost with positive perturbation
                states_plus, actions_plus, costs_plus = env.rollout(K_plus, steps=_rollout_length)
                cost_plus += np.mean(costs_plus)

                # Cost with negative perturbation
                states_minus, actions_minus, costs_minus = env.rollout(K_minus, steps=_rollout_length)
                cost_minus += np.mean(costs_minus)

            cost_plus /= _num_trajectories
            cost_minus /= _num_trajectories

        # Finite difference gradient estimate
        gradient += ((cost_plus - cost_minus) / (2 * perturbation_scale**2)) * U_i
        valid_samples += 1

    if valid_samples > 0:
        # Normalize by number of valid samples and add dimension scaling
        gradient = (state_dim * action_dim / valid_samples) * gradient

    return gradient


def zeroth_order_hessian_diag(env, K: np.ndarray, perturbations: np.ndarray,
                             perturbation_scale: float, rollout_length: int = None,
                             num_trajectories: int = None, use_analytical: bool = True) -> np.ndarray:
    """
    Estimate Hessian using zeroth-order method with (U @ U.T - I) formulation.

    This implements the formula: H ≈ E[(f(K + rU) - f(K)) * (U @ U.T - I) / r²]
    where U is flattened perturbation vector.

    Note: Called "_diag" because it estimates the full Hessian matrix in the flattened
    parameter space, which corresponds to diagonal blocks in the tensor view.
    For LQR K (action_dim × state_dim), this returns (param_dim × param_dim) matrix
    where param_dim = action_dim * state_dim.

    Parameters:
    env: Environment instance
    K: Current policy parameters (action_dim × state_dim)
    perturbations: Array of perturbation directions (num_perturbations, action_dim, state_dim)
    perturbation_scale: Perturbation magnitude (r)
    rollout_length: Length of rollouts (unused if use_analytical=True)
    num_trajectories: Number of trajectories per evaluation (unused if use_analytical=True)
    use_analytical: Whether to use analytical cost formula (much faster)

    Returns:
    np.ndarray: Estimated Hessian matrix (param_dim × param_dim)
    """
    param_dim = K.size  # Total number of parameters (action_dim * state_dim)
    hessian = np.zeros((param_dim, param_dim))
    valid_samples = 0

    # Get baseline cost
    if use_analytical and hasattr(env, 'analytical_cost'):
        baseline_cost = env.analytical_cost(K)
        if baseline_cost == float('inf'):
            # If baseline is unstable, return zero hessian
            return hessian
    else:
        # Use defaults if not provided
        _rollout_length = rollout_length or 50
        _num_trajectories = num_trajectories or 1

        baseline_costs = []
        for _ in range(_num_trajectories):
            _, _, costs = env.rollout(K, steps=_rollout_length)
            baseline_costs.append(np.mean(costs))
        baseline_cost = np.mean(baseline_costs)

    for i, U_i in enumerate(perturbations):
        K_plus = K + U_i

        if use_analytical and hasattr(env, 'analytical_cost'):
            # Use analytical cost computation
            perturbed_cost = env.analytical_cost(K_plus)

            # Skip if cost is infinite (unstable)
            if perturbed_cost == float('inf'):
                continue
        else:
            # Fallback to rollout-based estimation
            if not check_stability(env.A, env.B, K_plus):
                continue

            # Cost with perturbation
            perturbed_costs = []
            for _ in range(_num_trajectories):
                _, _, costs = env.rollout(K_plus, steps=_rollout_length)
                perturbed_costs.append(np.mean(costs))
            perturbed_cost = np.mean(perturbed_costs)

        # Cost difference
        cost_diff = perturbed_cost - baseline_cost

        # Flatten perturbation to vector (as in legacy code)
        U_vec = U_i.flatten()  # (param_dim,)

        # Hessian contribution: (U @ U.T - I) formulation
        U_outer = np.outer(U_vec, U_vec)  # (param_dim, param_dim)
        I_mat = np.eye(param_dim)

        # Hessian estimate: coefficient * cost_diff * (U @ U.T - I)
        hessian += cost_diff * (U_outer - I_mat)
        valid_samples += 1

    # Normalize (following legacy code scaling)
    if valid_samples > 0:
        scale_factor = (param_dim) / (valid_samples * perturbation_scale**2)
        hessian *= scale_factor

    return hessian


def estimate_covariance_matrix(env, K: np.ndarray, num_trajectories: int = 20,
                              rollout_length: int = 30, use_analytical: bool = True) -> np.ndarray:
    """
    Estimate the covariance matrix of state trajectories under policy K.
    This approximates the Fisher Information Matrix for natural gradients.

    Parameters:
    env: Environment instance
    K: Current policy parameters (gain matrix)
    num_trajectories: Number of trajectory samples (unused if use_analytical=True)
    rollout_length: Length of each trajectory (unused if use_analytical=True)
    use_analytical: Whether to use analytical covariance computation

    Returns:
    np.ndarray: Estimated covariance matrix (state_dim x state_dim)
    """
    state_dim = K.shape[1]

    if use_analytical and hasattr(env, 'analytical_covariance'):
        # Use analytical covariance computation
        return env.analytical_covariance(K)
    else:
        # Fallback to empirical estimation
        # Collect state samples from multiple trajectories
        all_states = []

        for _ in range(num_trajectories):
            try:
                states, _, _ = env.rollout(K, steps=rollout_length)
                all_states.extend(states)
            except:
                # Skip failed rollouts (unstable policies)
                continue

        if not all_states:
            # Fallback to identity if no successful rollouts
            return np.eye(state_dim)

        # Convert to numpy array
        states_array = np.array(all_states)  # (num_samples, state_dim)

        # Compute empirical covariance matrix
        # Cov = E[xx^T] where x are the state vectors
        covariance = np.cov(states_array.T)  # (state_dim, state_dim)

        # Ensure positive definite by adding small regularization
        covariance += 1e-6 * np.eye(state_dim)

        return covariance


def natural_gradient_with_covariance(gradient: np.ndarray, covariance: np.ndarray,
                                   damping: float = 1e-4) -> np.ndarray:
    """
    Compute natural gradient using estimated covariance matrix.
    Natural gradient = gradient @ Cov^(-1) for LQR case.

    Parameters:
    gradient: Policy gradient (action_dim x state_dim)
    covariance: State covariance matrix (state_dim x state_dim)
    damping: Regularization for numerical stability

    Returns:
    np.ndarray: Natural gradient (same shape as gradient)
    """
    state_dim = covariance.shape[0]

    # Add damping for numerical stability
    regularized_cov = covariance + damping * np.eye(state_dim)

    try:
        # Natural gradient: gradient @ Cov^(-1)
        # Each row of gradient is multiplied by inverse covariance
        cov_inv = np.linalg.inv(regularized_cov)
        natural_grad = gradient @ cov_inv

        return natural_grad

    except np.linalg.LinAlgError:
        # Fallback to regular gradient if inversion fails
        print("Warning: Covariance matrix inversion failed, using regular gradient")
        return gradient


def natural_gradient(gradient: np.ndarray, hessian_diag: np.ndarray,
                    damping: float = 1e-6) -> np.ndarray:
    """
    Compute natural gradient using diagonal Hessian approximation.
    [Legacy function - kept for backward compatibility]

    Parameters:
    gradient: Policy gradient
    hessian_diag: Diagonal Hessian estimate
    damping: Damping factor for numerical stability

    Returns:
    np.ndarray: Natural gradient
    """
    # Add damping to avoid division by zero
    damped_hessian = np.abs(hessian_diag) + damping

    # Natural gradient = H^(-1) * gradient (element-wise for diagonal)
    natural_grad = gradient / damped_hessian

    return natural_grad


def clip_gradient(gradient: np.ndarray, max_norm: float = 10.0) -> np.ndarray:
    """
    Clip gradient norm to prevent instability.

    Parameters:
    gradient: Gradient to clip
    max_norm: Maximum allowed norm

    Returns:
    np.ndarray: Clipped gradient
    """
    grad_norm = np.linalg.norm(gradient)
    if grad_norm > max_norm:
        gradient = gradient * (max_norm / grad_norm)
    return gradient


def create_baseline_policy(environment_class, environment_kwargs: dict,
                          state_dim: int, action_dim: int,
                          baseline_type: str = "random") -> np.ndarray:
    """
    Create a baseline policy for fair comparison between algorithms.

    Parameters:
    environment_class: Environment class
    environment_kwargs: Environment initialization kwargs
    state_dim: State dimension
    action_dim: Action dimension
    baseline_type: Type of baseline ("random", "lqr", "zero")

    Returns:
    np.ndarray: Baseline policy matrix K
    """
    if baseline_type == "random":
        # Random stable policy
        np.random.seed(42)  # Fixed seed for reproducibility
        K = np.random.normal(0, 0.1, size=(action_dim, state_dim))

    elif baseline_type == "lqr":
        # LQR solution for nominal environment (without perturbations)
        try:
            env = environment_class(**environment_kwargs)
            K = env.get_optimal_K()
        except:
            # Fallback to random if LQR fails
            np.random.seed(42)
            K = np.random.normal(0, 0.1, size=(action_dim, state_dim))

    elif baseline_type == "zero":
        # Zero policy (no control)
        K = np.zeros((action_dim, state_dim))

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    return K


def evaluate_policy_performance(envs: List, K: np.ndarray,
                               rollout_length: int = 50,
                               num_evaluations: int = 10,
                               use_analytical: bool = True) -> Tuple[float, float]:
    """
    Evaluate policy performance across multiple environments.

    Parameters:
    envs: List of environments
    K: Policy parameters (gain matrix)
    rollout_length: Length of evaluation rollouts (unused if use_analytical=True)
    num_evaluations: Number of evaluations per environment (unused if use_analytical=True)
    use_analytical: Whether to use analytical cost formula (much faster)

    Returns:
    Tuple[float, float]: (mean_cost, std_cost)
    """
    all_costs = []

    for env in envs:
        if use_analytical and hasattr(env, 'analytical_cost'):
            # Use analytical cost - single evaluation per environment
            cost = env.analytical_cost(K)
            if cost != float('inf'):
                all_costs.append(cost)
        else:
            # Fallback to rollout-based evaluation
            env_costs = []
            for _ in range(num_evaluations):
                try:
                    _, _, costs = env.rollout(K, steps=rollout_length)
                    env_costs.append(np.mean(costs))
                except Exception as e:
                    # Skip unstable rollouts
                    continue

            if env_costs:
                all_costs.extend(env_costs)

    if not all_costs:
        return float('inf'), float('inf')

    return np.mean(all_costs), np.std(all_costs)


def compute_regret(current_costs: List[float], optimal_costs: List[float]) -> float:
    """
    Compute regret as ratio of cost difference to optimal cost.

    Parameters:
    current_costs: Current policy costs for each task
    optimal_costs: Optimal costs for each task

    Returns:
    float: Average regret ratio
    """
    if len(current_costs) != len(optimal_costs):
        raise ValueError("Cost lists must have same length")

    regrets = []
    for curr, opt in zip(current_costs, optimal_costs):
        if opt > 0:
            regret = (curr - opt) / opt
            regrets.append(regret)

    return np.mean(regrets) if regrets else float('inf')


class Timer:
    """Simple context manager for timing code sections."""

    def __init__(self, description: str = ""):
        self.description = description
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.description:
            print(f"{self.description}: {self.elapsed:.4f} seconds")


def analytical_policy_gradient(env, K: np.ndarray, stability_margin: float = 0.99) -> np.ndarray:
    """
    Compute analytical policy gradient for LQR systems.

    For LQR cost J(K) = trace(P_K @ Psi), the gradient is:
    ∇_K J = 2 * (R + B^T P_K B) * K * Σ - 2 * B^T P_K A * Σ

    Parameters:
    env: LQR environment with analytical_cost and analytical_covariance methods
    K: Current policy parameters (action_dim x state_dim)
    stability_margin: Eigenvalue threshold for stability

    Returns:
    np.ndarray: Analytical gradient (same shape as K)
    """
    try:
        A, B, Q, R = env.A, env.B, env.Q, env.R

        # Check if policy is stable
        A_cl = A - B @ K
        eigenvals = np.linalg.eigvals(A_cl)
        max_eigenval = np.max(np.abs(eigenvals))

        if max_eigenval >= stability_margin:
            # Return zero gradient for unstable policies
            return np.zeros_like(K)

        # Solve discrete-time Riccati equation for P_K
        from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

        # P_K satisfies: P_K = A_cl^T P_K A_cl + Q_cl
        # where Q_cl = Q + K^T R K
        Q_cl = Q + K.T @ R @ K
        P_K = solve_discrete_lyapunov(A_cl.T, Q_cl)

        # Get steady-state covariance matrix Σ
        if hasattr(env, 'analytical_covariance'):
            Sigma = env.analytical_covariance(K)
        else:
            # Fallback: solve Σ = A_cl Σ A_cl^T + Ψ
            Sigma = solve_discrete_lyapunov(A_cl, env.Psi)

        # Compute gradient: ∇_K J = 2 * (R + B^T P_K B) * K * Σ - 2 * B^T P_K A * Σ
        BT_PK_B = B.T @ P_K @ B
        BT_PK_A = B.T @ P_K @ A

        gradient = 2 * (R + BT_PK_B) @ K @ Sigma - 2 * BT_PK_A @ Sigma

        return gradient

    except Exception as e:
        # Fallback to zero gradient if computation fails
        print(f"Warning: Analytical gradient computation failed: {e}")
        return np.zeros_like(K)


def log_metrics(logger, metrics: dict, step: int):
    """
    Log metrics to tensorboard logger.

    Parameters:
    logger: SummaryWriter instance
    metrics: Dictionary of metric_name -> value
    step: Training step/epoch
    """
    if logger is not None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                logger.add_scalar(name, value, step)