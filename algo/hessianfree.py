"""
Hessian-Free MAML Algorithm for LQR Control.

Implementation of Model-Agnostic Meta-Learning (MAML) using zeroth-order
(derivative-free) optimization for Linear Quadratic Regulator problems.

Based on the two-timescale algorithm from the original mamllqr.py but
restructured as a modular class that works with Boeing_dynamics and BaseLQREnv.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path
import pickle
import time
from tensorboardX import SummaryWriter


from utils.utils import (
    generate_perturbations, zeroth_order_gradient, clip_gradient,
    evaluate_policy_performance, compute_regret, Timer, log_metrics,
    check_stability, estimate_covariance_matrix, natural_gradient_with_covariance,
    analytical_policy_gradient
)


class HessianFreeMAML:
    """
    Hessian-Free Model-Agnostic Meta-Learning for LQR Control.

    This class implements a derivative-free version of MAML specifically designed
    for LQR control problems. It uses zeroth-order optimization with two-timescale
    updates: inner loop for task adaptation, outer loop for meta-learning.
    """

    def __init__(
        self,
        environment_class,
        environment_kwargs: Dict[str, Any] = None,
        state_dim: int = 4,
        action_dim: int = 2,
        inner_lr: float = 8e-6,
        outer_lr: float = 8e-6,
        perturbation_scale: float = 0.01,
        inner_perturbation_scale: float = 0.001,
        num_inner_perturbations: int = 30,
        num_inner_steps: int = 1,
        num_perturbations: int = 50,
        rollout_length: int = 100,
        use_natural_gradients: bool = False,
        grad_clip: float = 10.0,
        use_adam: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-7,
        use_analytical_gradients: bool = False,
        random_seed: Optional[int] = None
    ):
        """
        Initialize HessianFreeMAML algorithm.

        Parameters:
        environment_class: Environment class (Boeing_dynamics or BaseLQREnv etc.)
        environment_kwargs: Keyword arguments for environment initialization
        state_dim: Dimensionality of state space
        action_dim: Dimensionality of action space
        inner_lr: Learning rate for inner loop (task adaptation)
        outer_lr: Learning rate for outer loop (meta-learning)
        perturbation_scale: Scale of perturbations for gradient estimation (r)
        num_inner_steps: Number of inner loop gradient steps (d)
        num_perturbations: Number of perturbations for gradient estimation (m)
        rollout_length: Length of rollouts for cost estimation (l)
        use_natural_gradients: Whether to use natural gradient updates
        grad_clip: Maximum gradient norm (for stability)
        use_adam: Whether to use Adam optimizer for adaptive learning rates
        adam_beta1: Adam exponential decay rate for first moment (momentum)
        adam_beta2: Adam exponential decay rate for second moment (RMSprop)
        adam_eps: Adam epsilon for numerical stability
        use_analytical_gradients: Whether to use analytical gradients for inner loop (LQR only)
        random_seed: Random seed for reproducibility
        """
        # Store parameters
        self.environment_class = environment_class
        self.environment_kwargs = environment_kwargs or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr  # adaptation rate
        self.outer_lr = outer_lr  # meta training rate
        self.perturbation_scale = perturbation_scale  # Stein's method U sampling scale (r)
        self.inner_perturbation_scale = inner_perturbation_scale
        self.num_inner_steps = num_inner_steps  # number of inner adaptation steps
        self.num_perturbations = num_perturbations  # d in paper, assume identical, number of zero-th order perturbation samples
        self.num_inner_perturbations = num_inner_perturbations  # number of perturbations for inner loop gradient estimation
        self.rollout_length = rollout_length  # ell which is the length of the rollout for cost estimation
        self.use_natural_gradients = use_natural_gradients  # flag for whether using natural gradient, which requires estimation of covariance matrix
        self.grad_clip = grad_clip

        # Adam optimizer parameters
        self.use_adam = use_adam
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps

        # Analytical gradient option
        self.use_analytical_gradients = use_analytical_gradients

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize policy parameters (LQR gain matrix)
        #self.K = np.zeros((action_dim, state_dim))
        K_opt_nominal = self.environment_class(**self.environment_kwargs).get_optimal_K()
        self.K = K_opt_nominal + 0.25 * np.random.rand(action_dim, state_dim)

        # Adam state variables for outer loop (meta-learning)
        if self.use_adam:
            self.meta_m = np.zeros_like(self.K)  # First moment estimate
            self.meta_v = np.zeros_like(self.K)  # Second moment estimate
            self.meta_t = 0  # Time step

        # Training state
        self.training_history = []
        self.current_epoch = 0

        # Logging
        self.logger = None

    def create_task_environments(self, num_tasks: int,
                               perturbation_scale: float = 1.1e-07) -> List:
        """
        Create a set of related task environments for meta-learning.

        Parameters:
        num_tasks: Number of task environments to create
        perturbation_scale: Scale of perturbations between tasks

        Returns:
        List of environment instances
        """
        # Create base environment
        base_env = self.environment_class(**self.environment_kwargs)

        # Generate task variants
        task_envs = base_env.get_task_variants(
            num_variants=num_tasks,
            perturbation_scale=perturbation_scale
        )

        return task_envs

    def inner_loop_update(self, env, K_init: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform inner loop adaptation for a single task.

        Parameters:
        env: Task environment
        K_init: Initial policy parameters

        Returns:
        Tuple[np.ndarray, List[float]]: (adapted_policy, cost_history)
        """
        K = K_init.copy()
        cost_history = []

        # Note: Adam disabled for inner loop - use simple SGD for fast task adaptation

        for step in range(self.num_inner_steps):
            # Generate perturbations for gradient estimation
            perturbations = generate_perturbations(
                K.shape, self.num_inner_perturbations, self.inner_perturbation_scale
            )

            # Estimate gradient using zeroth-order method
            gradient = zeroth_order_gradient(
                env, K, perturbations, self.inner_perturbation_scale,
                self.rollout_length, num_trajectories=30
            )

            # Apply natural gradients if enabled
            if self.use_natural_gradients:
                # Estimate covariance matrix of states under current policy
                covariance = estimate_covariance_matrix(
                    env, K, num_trajectories=30, rollout_length=self.rollout_length
                )

                # Convert to natural gradient
                gradient = natural_gradient_with_covariance(gradient, covariance)

            # Clip gradient for stability
            gradient = clip_gradient(gradient, self.grad_clip)

            # Inner loop update (always use SGD for simplicity and speed)
            K = K - self.inner_lr * gradient

            # Evaluate current policy
            try:
                cost = env.analytical_cost(K)
                cost_history.append(cost)
            except:
                cost_history.append(float('inf'))
                break

        return K, cost_history

    def analytical_inner_loop_update(self, env, K_init: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform inner loop adaptation using analytical gradients (for LQR systems).

        Much faster and more accurate than zeroth-order gradient estimation.

        Parameters:
        env: Task environment (must support analytical_policy_gradient)
        K_init: Initial policy parameters

        Returns:
        Tuple[np.ndarray, List[float]]: (adapted_policy, cost_history)
        """
        K = K_init.copy()
        cost_history = []

        for step in range(self.num_inner_steps):
            # Compute analytical gradient
            gradient = analytical_policy_gradient(env, K)

            # Apply natural gradients if enabled
            if self.use_natural_gradients:
                # Estimate covariance matrix of states under current policy
                covariance = estimate_covariance_matrix(
                    env, K, num_trajectories=30, rollout_length=self.rollout_length
                )

                # Convert to natural gradient
                gradient = natural_gradient_with_covariance(gradient, covariance)

            # Clip gradient for stability
            gradient = clip_gradient(gradient, self.grad_clip)

            # Inner loop update (always use SGD for simplicity and speed)
            K = K - self.inner_lr * gradient

            # Evaluate current policy
            try:
                cost = env.analytical_cost(K)
                cost_history.append(cost)
            except:
                cost_history.append(float('inf'))
                break

        return K, cost_history

    def outer_loop_update(self, task_envs: List, K_meta: np.ndarray) -> np.ndarray:
        """
        Perform outer loop meta-update across tasks.

        Parameters:
        task_envs: List of task environments
        K_meta: Current meta-policy parameters

        Returns:
        np.ndarray: Updated meta-policy parameters
        """
        meta_gradient = np.zeros_like(K_meta)

        # Sample subset of tasks for this update
        num_tasks_batch = min(len(task_envs), max(1, len(task_envs) // 2))
        task_indices = np.random.choice(len(task_envs), num_tasks_batch, replace=False)

        for task_idx in task_indices:
            env = task_envs[task_idx]

            # Outer perturbation for meta-gradient estimation
            outer_perturbations = generate_perturbations(
                K_meta.shape, self.num_perturbations, self.perturbation_scale
            )

            task_meta_gradient = np.zeros_like(K_meta)

            for d, U_d in enumerate(outer_perturbations):
                # Perturbed meta-policy
                K_perturbed = K_meta + U_d

                # Inner loop adaptation with perturbed initialization
                if self.use_analytical_gradients:
                    K_adapted, _ = self.analytical_inner_loop_update(env, K_perturbed)
                else:
                    K_adapted, _ = self.inner_loop_update(env, K_perturbed)

                # Evaluate adapted policy
                # try:
                #     _, _, costs = env.rollout(K_adapted, steps=self.rollout_length)
                #     adapted_cost = np.mean(costs)
                # except:
                #     adapted_cost = float('inf')
                adapted_cost = env.analytical_cost(K_adapted)

                # Accumulate meta-gradient using MAML objective
                if not np.isinf(adapted_cost):
                    task_meta_gradient += adapted_cost * U_d

            # Scale by number of perturbations and perturbation scale
            dk = self.state_dim * self.action_dim
            task_meta_gradient *= (dk / (self.num_perturbations * self.perturbation_scale**2))

            meta_gradient += task_meta_gradient

        # Average across tasks
        meta_gradient = meta_gradient / num_tasks_batch

        # Clip meta-gradient
        meta_gradient = clip_gradient(meta_gradient, self.grad_clip)

        # Meta-update
        if self.use_adam:
            # Update time step
            self.meta_t += 1

            # Adam update for outer loop
            self.meta_m = self.adam_beta1 * self.meta_m + (1 - self.adam_beta1) * meta_gradient
            self.meta_v = self.adam_beta2 * self.meta_v + (1 - self.adam_beta2) * (meta_gradient ** 2)

            # Bias correction
            meta_m_hat = self.meta_m / (1 - self.adam_beta1 ** self.meta_t)
            meta_v_hat = self.meta_v / (1 - self.adam_beta2 ** self.meta_t)

            # Update
            K_meta_new = K_meta - self.outer_lr * meta_m_hat / (np.sqrt(meta_v_hat) + self.adam_eps)
        else:
            # Standard SGD update
            K_meta_new = K_meta - self.outer_lr * meta_gradient

        return K_meta_new

    def reptile_outer_loop_update(self, task_envs: List, K_meta: np.ndarray) -> np.ndarray:
        """
        Perform Reptile outer loop meta-update across tasks.

        Reptile is simpler than MAML: it just moves the meta-parameters toward
        the adapted parameters without computing gradients through the adaptation.

        Parameters:
        task_envs: List of task environments
        K_meta: Current meta-policy parameters

        Returns:
        np.ndarray: Updated meta-policy parameters
        """
        meta_direction = np.zeros_like(K_meta)

        # Sample subset of tasks for this update
        num_tasks_batch = min(len(task_envs), max(1, len(task_envs) // 2))
        task_indices = np.random.choice(len(task_envs), num_tasks_batch, replace=False)

        for task_idx in task_indices:
            env = task_envs[task_idx]

            # Adapt to this task (same inner loop as MAML)
            if self.use_analytical_gradients:
                K_adapted, _ = self.analytical_inner_loop_update(env, K_meta)
            else:
                K_adapted, _ = self.inner_loop_update(env, K_meta)

            # Reptile direction: adapted_params - meta_params
            direction = (K_adapted - K_meta)
            meta_direction += direction

        # Average across tasks
        meta_direction = meta_direction / num_tasks_batch

        # Clip meta-direction for stability
        meta_direction = clip_gradient(meta_direction, self.grad_clip)

        # Reptile meta-update: move toward adapted parameters
        if self.use_adam:
            # Update time step
            self.meta_t += 1

            # Adam update for outer loop (treating direction as "gradient")
            self.meta_m = self.adam_beta1 * self.meta_m + (1 - self.adam_beta1) * meta_direction
            self.meta_v = self.adam_beta2 * self.meta_v + (1 - self.adam_beta2) * (meta_direction ** 2)

            # Bias correction
            meta_m_hat = self.meta_m / (1 - self.adam_beta1 ** self.meta_t)
            meta_v_hat = self.meta_v / (1 - self.adam_beta2 ** self.meta_t)

            # Update
            K_meta_new = K_meta + self.outer_lr * meta_m_hat / (np.sqrt(meta_v_hat) + self.adam_eps)
        else:
            # Standard Reptile update
            K_meta_new = K_meta + self.outer_lr * meta_direction

        return K_meta_new

    def train(
        self,
        num_tasks: int = 5,
        num_epochs: int = 1000,
        task_perturbation_scale: float = 1.1e-07,
        eval_interval: int = 5,
        log_dir: Optional[str] = None,
        save_interval: int = 100,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train the MAML algorithm.

        Parameters:
        num_tasks: Number of training tasks
        num_epochs: Number of training epochs
        task_perturbation_scale: Scale of task perturbations
        eval_interval: Evaluate every N epochs
        log_dir: Directory for logging (optional)
        save_interval: Save progress every N epochs
        verbose: Print progress

        Returns:
        Dict containing training history
        """
        # Setup logging
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.logger = SummaryWriter(str(log_path))

        # Create task environments
        if verbose:
            print(f"Creating {num_tasks} task environments...")

        task_envs = self.create_task_environments(num_tasks, task_perturbation_scale)

        # Compute optimal policies for comparison
        optimal_Ks = []
        optimal_costs = []

        for env in task_envs:
            try:
                K_opt = env.get_optimal_K()
                optimal_Ks.append(K_opt)

                cost = env.analytical_cost(K_opt)
                optimal_costs.append(cost)
            except:
                optimal_Ks.append(np.zeros_like(self.K))
                optimal_costs.append(float('inf'))

        if verbose:
            print(f"Average optimal cost: {np.mean([c for c in optimal_costs if not np.isinf(c)]):.4f}")

        # Training loop
        training_history = {
            'epoch': [],
            'mean_cost': [],  # Cost after adaptation
            'cost_std': [],
            'zero_shot_cost': [],  # Cost before adaptation (meta-policy directly)
            'zero_shot_cost_std': [],
            'regret': [],
            'policy_diff': [],
            'meta_gradient_norm': []
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            with Timer(f"Epoch {epoch}" if verbose and epoch % 10 == 0 else ""):
                # Meta-update
                K_old = self.K.copy()
                self.K = self.outer_loop_update(task_envs, self.K)

                # Compute meta-gradient norm for monitoring
                meta_grad_norm = np.linalg.norm(self.K - K_old) / self.outer_lr

            # Evaluation
            if epoch % eval_interval == 0:
                # Evaluate current meta-policy on all tasks
                all_costs = []  # Costs after adaptation
                all_zero_shot_costs = []  # Costs before adaptation

                for env in task_envs:
                    # Evaluate zero-shot (meta-policy directly, no adaptation)
                    zero_shot_cost = env.analytical_cost(self.K)
                    if not np.isinf(zero_shot_cost):
                        all_zero_shot_costs.append(zero_shot_cost)

                    # Quick adaptation
                    if self.use_analytical_gradients:
                        K_adapted, _ = self.analytical_inner_loop_update(env, self.K)
                    else:
                        K_adapted, _ = self.inner_loop_update(env, self.K)

                    # Evaluate adapted cost
                    adapted_cost = env.analytical_cost(K_adapted)

                    if not np.isinf(adapted_cost):
                        all_costs.append(adapted_cost)

                if all_costs:
                    mean_cost = np.mean(all_costs)
                    cost_std = np.std(all_costs)
                    regret = compute_regret(all_costs, optimal_costs)

                    # Zero-shot statistics
                    zero_shot_mean = np.mean(all_zero_shot_costs) if all_zero_shot_costs else float('inf')
                    zero_shot_std = np.std(all_zero_shot_costs) if all_zero_shot_costs else float('inf')

                    # Policy difference from optimal
                    policy_diffs = []
                    for K_opt in optimal_Ks:
                        if not np.any(np.isinf(K_opt)):
                            policy_diffs.append(np.linalg.norm(self.K - K_opt))

                    policy_diff = np.mean(policy_diffs) if policy_diffs else float('inf')

                    # Store history
                    training_history['epoch'].append(epoch)
                    training_history['mean_cost'].append(mean_cost)
                    training_history['cost_std'].append(cost_std)
                    training_history['zero_shot_cost'].append(zero_shot_mean)
                    training_history['zero_shot_cost_std'].append(zero_shot_std)
                    training_history['regret'].append(regret)
                    training_history['policy_diff'].append(policy_diff)
                    training_history['meta_gradient_norm'].append(meta_grad_norm)

                    if verbose:
                        print(f"Epoch {epoch:4d}: Adapted={mean_cost:.4f}±{cost_std:.6f}, "
                              f"Zero-shot={zero_shot_mean:.4f}±{zero_shot_std:.6f}, "
                              f"Regret={regret:.4f}")

                    # Log to tensorboard
                    if self.logger:
                        metrics = {
                            'cost/adapted_mean': mean_cost,
                            'cost/adapted_std': cost_std,
                            'cost/zero_shot_mean': zero_shot_mean,
                            'cost/zero_shot_std': zero_shot_std,
                            'regret': regret,
                            'policy_difference': policy_diff,
                            'meta_gradient_norm': meta_grad_norm
                        }
                        log_metrics(self.logger, metrics, epoch)

            # Save progress
            if save_interval and epoch % save_interval == 0 and log_dir:
                save_path = Path(log_dir) / f"checkpoint_epoch_{epoch}.pkl"
                self.save_checkpoint(save_path)

                # Also save intermediate training history
                history_path = Path(log_dir) / f"training_history_epoch_{epoch}.pkl"
                with open(history_path, 'wb') as f:
                    pickle.dump(training_history, f)

        # Final save
        if log_dir:
            final_path = Path(log_dir) / "final_model.pkl"
            self.save_checkpoint(final_path)

            # Save training history
            history_path = Path(log_dir) / "training_history.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(training_history, f)

            # Also save as JSON for easy reading
            import json
            history_json = {k: [float(x) if not np.isnan(x) and not np.isinf(x) else None for x in v] for k, v in training_history.items()}
            json_path = Path(log_dir) / "training_history.json"
            with open(json_path, 'w') as f:
                json.dump(history_json, f, indent=2)

        self.training_history = training_history
        return training_history

    def train_reptile(
        self,
        num_tasks: int = 5,
        num_epochs: int = 1000,
        task_perturbation_scale: float = 1.1e-07,
        eval_interval: int = 5,
        log_dir: Optional[str] = None,
        save_interval: int = 100,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train using the Reptile algorithm (simpler than MAML).

        Reptile moves meta-parameters toward adapted parameters without
        computing gradients through the adaptation steps.

        Parameters:
        num_tasks: Number of training tasks
        num_epochs: Number of training epochs
        task_perturbation_scale: Scale of task perturbations
        eval_interval: Evaluate every N epochs
        log_dir: Directory for logging (optional)
        save_interval: Save progress every N epochs
        verbose: Print progress

        Returns:
        Dict containing training history
        """
        # Setup logging
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.logger = SummaryWriter(str(log_path))

        # Create task environments
        if verbose:
            print(f"Creating {num_tasks} task environments for Reptile training...")

        task_envs = self.create_task_environments(num_tasks, task_perturbation_scale)

        # Compute optimal policies for comparison
        optimal_Ks = []
        optimal_costs = []

        for env in task_envs:
            try:
                K_opt = env.get_optimal_K()
                optimal_Ks.append(K_opt)

                cost = env.analytical_cost(K_opt)
                optimal_costs.append(cost)
            except:
                optimal_Ks.append(np.zeros_like(self.K))
                optimal_costs.append(float('inf'))

        if verbose:
            print(f"Average optimal cost: {np.mean([c for c in optimal_costs if not np.isinf(c)]):.4f}")

        # Training loop
        training_history = {
            'epoch': [],
            'mean_cost': [],  # Cost after adaptation
            'cost_std': [],
            'zero_shot_cost': [],  # Cost before adaptation (meta-policy directly)
            'zero_shot_cost_std': [],
            'regret': [],
            'policy_diff': [],
            'meta_direction_norm': []  # Track magnitude of Reptile updates
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            with Timer(f"Epoch {epoch}" if verbose and epoch % 10 == 0 else ""):
                # Reptile meta-update
                K_old = self.K.copy()
                self.K = self.reptile_outer_loop_update(task_envs, self.K)

                # Compute meta-direction norm for monitoring
                meta_direction_norm = np.linalg.norm(self.K - K_old)

            # Evaluation
            if epoch % eval_interval == 0:
                # Evaluate current meta-policy on all tasks
                all_costs = []  # Costs after adaptation
                all_zero_shot_costs = []  # Costs before adaptation

                for env in task_envs:
                    # Evaluate zero-shot (meta-policy directly, no adaptation)
                    zero_shot_cost = env.analytical_cost(self.K)
                    if not np.isinf(zero_shot_cost):
                        all_zero_shot_costs.append(zero_shot_cost)

                    # Quick adaptation
                    if self.use_analytical_gradients:
                        K_adapted, _ = self.analytical_inner_loop_update(env, self.K)
                    else:
                        K_adapted, _ = self.inner_loop_update(env, self.K)

                    # Evaluate adapted cost
                    adapted_cost = env.analytical_cost(K_adapted)

                    if not np.isinf(adapted_cost):
                        all_costs.append(adapted_cost)

                if all_costs:
                    mean_cost = np.mean(all_costs)
                    cost_std = np.std(all_costs)
                    regret = compute_regret(all_costs, optimal_costs)

                    # Zero-shot statistics
                    zero_shot_mean = np.mean(all_zero_shot_costs) if all_zero_shot_costs else float('inf')
                    zero_shot_std = np.std(all_zero_shot_costs) if all_zero_shot_costs else float('inf')

                    # Policy difference from optimal
                    policy_diffs = []
                    for K_opt in optimal_Ks:
                        if not np.any(np.isinf(K_opt)):
                            policy_diffs.append(np.linalg.norm(self.K - K_opt))

                    policy_diff = np.mean(policy_diffs) if policy_diffs else float('inf')

                    # Store history
                    training_history['epoch'].append(epoch)
                    training_history['mean_cost'].append(mean_cost)
                    training_history['cost_std'].append(cost_std)
                    training_history['zero_shot_cost'].append(zero_shot_mean)
                    training_history['zero_shot_cost_std'].append(zero_shot_std)
                    training_history['regret'].append(regret)
                    training_history['policy_diff'].append(policy_diff)
                    training_history['meta_direction_norm'].append(meta_direction_norm)

                    if verbose:
                        print(f"Epoch {epoch:4d}: Adapted={mean_cost:.4f}±{cost_std:.6f}, "
                              f"Zero-shot={zero_shot_mean:.4f}±{zero_shot_std:.6f}, "
                              f"Regret={regret:.4f}, Direction_norm={meta_direction_norm:.6f}")

                    # Log to tensorboard
                    if self.logger:
                        metrics = {
                            'cost/adapted_mean': mean_cost,
                            'cost/adapted_std': cost_std,
                            'cost/zero_shot_mean': zero_shot_mean,
                            'cost/zero_shot_std': zero_shot_std,
                            'regret': regret,
                            'policy_difference': policy_diff,
                            'meta_direction_norm': meta_direction_norm
                        }
                        log_metrics(self.logger, metrics, epoch)

            # Save progress
            if save_interval and epoch % save_interval == 0 and log_dir:
                save_path = Path(log_dir) / f"reptile_checkpoint_epoch_{epoch}.pkl"
                self.save_checkpoint(save_path)

                # Also save intermediate training history
                history_path = Path(log_dir) / f"reptile_training_history_epoch_{epoch}.pkl"
                with open(history_path, 'wb') as f:
                    pickle.dump(training_history, f)

        # Final save
        if log_dir:
            final_path = Path(log_dir) / "reptile_final_model.pkl"
            self.save_checkpoint(final_path)

            # Save training history
            history_path = Path(log_dir) / "reptile_training_history.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(training_history, f)

            # Also save as JSON for easy reading
            import json
            history_json = {k: [float(x) if not np.isnan(x) and not np.isinf(x) else None for x in v]
                           for k, v in training_history.items()}
            json_path = Path(log_dir) / "reptile_training_history.json"
            with open(json_path, 'w') as f:
                json.dump(history_json, f, indent=2)

        self.training_history = training_history
        return training_history

    def train_combine(
        self,
        num_tasks: int = 5,
        num_epochs: int = 1000,
        task_perturbation_scale: float = 1.1e-07,
        eval_interval: int = 5,
        log_dir: Optional[str] = None,
        save_interval: int = 100,
        verbose: bool = True,
        reptile_weight: float = 0.5
    ) -> Dict[str, List]:
        """
        Train using convex combination of Reptile and HessianFree MAML updates.

        Meta-update: θ ← θ + α * [λ * reptile_direction + (1-λ) * maml_gradient]

        Parameters:
        reptile_weight: Weight for reptile direction (λ ∈ [0,1])
                       λ=0: Pure HessianFree MAML
                       λ=1: Pure Reptile
                       λ=0.5: Equal combination
        ... (other parameters same as train)

        Returns:
        Dict containing training history
        """
        # Setup logging
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.logger = SummaryWriter(str(log_path))

        # Create task environments
        if verbose:
            print(f"Creating {num_tasks} task environments for Combined training...")
            print(f"Reptile weight: {reptile_weight:.2f}, MAML weight: {1-reptile_weight:.2f}")

        task_envs = self.create_task_environments(num_tasks, task_perturbation_scale)

        # Compute optimal policies for comparison
        optimal_Ks = []
        optimal_costs = []

        for env in task_envs:
            try:
                K_opt = env.get_optimal_K()
                optimal_Ks.append(K_opt)

                cost = env.analytical_cost(K_opt)
                optimal_costs.append(cost)
            except:
                optimal_Ks.append(np.zeros_like(self.K))
                optimal_costs.append(float('inf'))

        if verbose:
            print(f"Average optimal cost: {np.mean([c for c in optimal_costs if not np.isinf(c)]):.4f}")

        # Training loop
        training_history = {
            'epoch': [],
            'mean_cost': [],  # Cost after adaptation
            'cost_std': [],
            'zero_shot_cost': [],  # Cost before adaptation (meta-policy directly)
            'zero_shot_cost_std': [],
            'regret': [],
            'policy_diff': [],
            'combined_update_norm': [],  # Track magnitude of combined updates
            'reptile_component_norm': [],  # Track Reptile component
            'maml_component_norm': []  # Track MAML component
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            with Timer(f"Epoch {epoch}" if verbose and epoch % 10 == 0 else ""):
                K_old = self.K.copy()

                # Compute Reptile direction
                reptile_direction = np.zeros_like(self.K)
                num_tasks_batch = min(len(task_envs), max(1, len(task_envs) // 2))
                task_indices = np.random.choice(len(task_envs), num_tasks_batch, replace=False)

                # for task_idx in task_indices:
                #     env = task_envs[task_idx]
                #     # Adapt to this task
                #     if self.use_analytical_gradients:
                #         K_adapted, _ = self.analytical_inner_loop_update(env, self.K)
                #     else:
                #         K_adapted, _ = self.inner_loop_update(env, self.K)

                #     # Reptile direction: adapted_params - meta_params
                #     reptile_direction += (K_adapted - self.K) * 100

                # reptile_direction = reptile_direction / num_tasks_batch

                # Compute MAML meta-gradient (using existing outer_loop_update logic)
                maml_meta_gradient = np.zeros_like(self.K)

                for task_idx in task_indices:
                    env = task_envs[task_idx]

                    # Outer perturbations for meta-gradient estimation
                    outer_perturbations = generate_perturbations(
                        self.K.shape, self.num_perturbations, self.perturbation_scale
                    )

                    task_meta_gradient = np.zeros_like(self.K)

                    for d, U_d in enumerate(outer_perturbations):
                        # Perturbed meta-policy
                        K_perturbed = self.K + U_d

                        # Inner loop adaptation with perturbed initialization
                        if self.use_analytical_gradients:
                            K_adapted, _ = self.analytical_inner_loop_update(env, K_perturbed)
                        else:
                            K_adapted, _ = self.inner_loop_update(env, K_perturbed)

                        # Evaluate adapted policy
                        adapted_cost = env.analytical_cost(K_adapted)
                        
                        reptile_direction += (K_adapted - K_perturbed)/(num_tasks_batch * self.inner_lr)

                        # Accumulate meta-gradient using MAML objective
                        if not np.isinf(adapted_cost):
                            task_meta_gradient += adapted_cost * U_d

                    # Scale by number of perturbations and perturbation scale
                    dk = self.state_dim * self.action_dim
                    task_meta_gradient *= (dk / (self.num_perturbations * self.perturbation_scale**2))

                    maml_meta_gradient -= task_meta_gradient / num_tasks_batch

                
                # Convex combination of directions
                combined_direction = (reptile_weight * reptile_direction +
                                    (1 - reptile_weight) * maml_meta_gradient)

                # Clip combined direction
                combined_direction = clip_gradient(combined_direction, self.grad_clip)

                # Apply combined update with Adam if enabled
                if self.use_adam:
                    # Update time step
                    self.meta_t += 1

                    # Adam update (treating combined direction as "gradient")
                    self.meta_m = self.adam_beta1 * self.meta_m + (1 - self.adam_beta1) * combined_direction
                    self.meta_v = self.adam_beta2 * self.meta_v + (1 - self.adam_beta2) * (combined_direction ** 2)

                    # Bias correction
                    meta_m_hat = self.meta_m / (1 - self.adam_beta1 ** self.meta_t)
                    meta_v_hat = self.meta_v / (1 - self.adam_beta2 ** self.meta_t)

                    # Update
                    self.K = self.K + self.outer_lr * meta_m_hat / (np.sqrt(meta_v_hat) + self.adam_eps)
                else:
                    # Standard update with combined direction
                    self.K = self.K + self.outer_lr * combined_direction

                # Compute norms for monitoring
                combined_update_norm = np.linalg.norm(self.K - K_old)
                reptile_component_norm = np.linalg.norm(reptile_direction)
                maml_component_norm = np.linalg.norm(maml_meta_gradient)

            # Evaluation
            if epoch % eval_interval == 0:
                # Evaluate current meta-policy on all tasks
                all_costs = []  # Costs after adaptation
                all_zero_shot_costs = []  # Costs before adaptation

                for env in task_envs:
                    # Evaluate zero-shot (meta-policy directly, no adaptation)
                    zero_shot_cost = env.analytical_cost(self.K)
                    if not np.isinf(zero_shot_cost):
                        all_zero_shot_costs.append(zero_shot_cost)

                    # Quick adaptation
                    if self.use_analytical_gradients:
                        K_adapted, _ = self.analytical_inner_loop_update(env, self.K)
                    else:
                        K_adapted, _ = self.inner_loop_update(env, self.K)

                    # Evaluate adapted cost
                    adapted_cost = env.analytical_cost(K_adapted)

                    if not np.isinf(adapted_cost):
                        all_costs.append(adapted_cost)

                if all_costs:
                    mean_cost = np.mean(all_costs)
                    cost_std = np.std(all_costs)
                    regret = compute_regret(all_costs, optimal_costs)

                    # Zero-shot statistics
                    zero_shot_mean = np.mean(all_zero_shot_costs) if all_zero_shot_costs else float('inf')
                    zero_shot_std = np.std(all_zero_shot_costs) if all_zero_shot_costs else float('inf')

                    # Policy difference from optimal
                    policy_diffs = []
                    for K_opt in optimal_Ks:
                        if not np.any(np.isinf(K_opt)):
                            policy_diffs.append(np.linalg.norm(self.K - K_opt))

                    policy_diff = np.mean(policy_diffs) if policy_diffs else float('inf')

                    # Store history
                    training_history['epoch'].append(epoch)
                    training_history['mean_cost'].append(mean_cost)
                    training_history['cost_std'].append(cost_std)
                    training_history['zero_shot_cost'].append(zero_shot_mean)
                    training_history['zero_shot_cost_std'].append(zero_shot_std)
                    training_history['regret'].append(regret)
                    training_history['policy_diff'].append(policy_diff)
                    training_history['combined_update_norm'].append(combined_update_norm)
                    training_history['reptile_component_norm'].append(reptile_component_norm)
                    training_history['maml_component_norm'].append(maml_component_norm)

                    if verbose:
                        print(f"Epoch {epoch:4d}: Adapted={mean_cost:.4f}±{cost_std:.6f}, "
                              f"Zero-shot={zero_shot_mean:.4f}±{zero_shot_std:.6f}, "
                              f"Regret={regret:.4f}, Combined_norm={combined_update_norm:.6f}")

                    # Log to tensorboard
                    if self.logger:
                        metrics = {
                            'cost/adapted_mean': mean_cost,
                            'cost/adapted_std': cost_std,
                            'cost/zero_shot_mean': zero_shot_mean,
                            'cost/zero_shot_std': zero_shot_std,
                            'regret': regret,
                            'policy_difference': policy_diff,
                            'combined_update_norm': combined_update_norm,
                            'reptile_component_norm': reptile_component_norm,
                            'maml_component_norm': maml_component_norm
                        }
                        log_metrics(self.logger, metrics, epoch)

            # Save progress
            if save_interval and epoch % save_interval == 0 and log_dir:
                save_path = Path(log_dir) / f"combined_checkpoint_epoch_{epoch}.pkl"
                self.save_checkpoint(save_path)

                # Also save intermediate training history
                history_path = Path(log_dir) / f"combined_training_history_epoch_{epoch}.pkl"
                with open(history_path, 'wb') as f:
                    pickle.dump(training_history, f)

        # Final save
        if log_dir:
            final_path = Path(log_dir) / "combined_final_model.pkl"
            self.save_checkpoint(final_path)

            # Save training history
            history_path = Path(log_dir) / "combined_training_history.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(training_history, f)

            # Also save as JSON for easy reading
            import json
            history_json = {k: [float(x) if not np.isnan(x) and not np.isinf(x) else None for x in v]
                           for k, v in training_history.items()}
            json_path = Path(log_dir) / "combined_training_history.json"
            with open(json_path, 'w') as f:
                json.dump(history_json, f, indent=2)

        self.training_history = training_history
        return training_history

    def adapt_to_task(self, env, num_adaptation_steps: Optional[int] = None) -> np.ndarray:
        """
        Adapt the meta-policy to a specific task.

        Parameters:
        env: Target task environment
        num_adaptation_steps: Number of adaptation steps (uses default if None)

        Returns:
        np.ndarray: Adapted policy parameters
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = self.num_inner_steps

        # Temporarily modify num_inner_steps
        original_steps = self.num_inner_steps
        self.num_inner_steps = num_adaptation_steps

        # Perform adaptation
        K_adapted, _ = self.inner_loop_update(env, self.K)

        # Restore original setting
        self.num_inner_steps = original_steps

        return K_adapted

    def save_checkpoint(self, filepath: Union[str, Path]):
        """Save algorithm state to file."""
        checkpoint = {
            'K': self.K,
            'training_history': self.training_history,
            'current_epoch': self.current_epoch,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'inner_lr': self.inner_lr,
                'outer_lr': self.outer_lr,
                'perturbation_scale': self.perturbation_scale,
                'num_inner_steps': self.num_inner_steps,
                'num_perturbations': self.num_perturbations,
                'rollout_length': self.rollout_length,
                'use_natural_gradients': self.use_natural_gradients,
                'grad_clip': self.grad_clip,
                'use_adam': self.use_adam,
                'adam_beta1': self.adam_beta1,
                'adam_beta2': self.adam_beta2,
                'adam_eps': self.adam_eps,
                'use_analytical_gradients': self.use_analytical_gradients
            }
        }

        # Save Adam state if using Adam
        if self.use_adam:
            checkpoint['adam_state'] = {
                'meta_m': self.meta_m,
                'meta_v': self.meta_v,
                'meta_t': self.meta_t
            }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, filepath: Union[str, Path]):
        """Load algorithm state from file."""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        self.K = checkpoint['K']
        self.training_history = checkpoint.get('training_history', [])
        self.current_epoch = checkpoint.get('current_epoch', 0)

        # Optionally restore hyperparameters
        if 'hyperparameters' in checkpoint:
            for key, value in checkpoint['hyperparameters'].items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Restore Adam state if available
        if 'adam_state' in checkpoint and self.use_adam:
            adam_state = checkpoint['adam_state']
            self.meta_m = adam_state['meta_m']
            self.meta_v = adam_state['meta_v']
            self.meta_t = adam_state['meta_t']


if __name__ == "__main__":
    # Example usage
    from envs.Boeing import Boeing_dynamics

    print("Testing HessianFreeMAML with Boeing dynamics...")
    print("=" * 50)

    # Test 1: Regular gradients
    print("\n1. Testing with regular gradients:")
    maml_regular = HessianFreeMAML(
        environment_class=Boeing_dynamics,
        environment_kwargs={'noise_scale': 0.05, 'max_steps': 100},
        state_dim=4,
        action_dim=2,
        inner_lr=8e-5,
        outer_lr=5e-3,
        perturbation_scale=1.1e-03,
        inner_perturbation_scale=1.1e-05,
        num_inner_perturbations=150,
        num_inner_steps=1,
        num_perturbations=50,  # 
        rollout_length=100,     #
        use_natural_gradients=True,
        random_seed=2025,
        use_analytical_gradients=True,
        use_adam=False,
    )

    # history_regular = maml_regular.train(
    #     num_tasks=10,
    #     num_epochs=1000,
    #     eval_interval=1,
    #     log_dir="logs/regular_gradients",
    #     task_perturbation_scale=1.1e-04,
    #     verbose=True
    # )
    
    # history_reptile = maml_regular.train_reptile(
    #     num_tasks=10,
    #     num_epochs=1000,
    #     eval_interval=1,
    #     log_dir="logs/reptile",
    #     task_perturbation_scale=1.1e-04,
    #     verbose=True
    # )
    history_reptile = maml_regular.train_combine(
        reptile_weight=0.5,
        num_tasks=10,
        num_epochs=1000,
        eval_interval=1,
        log_dir="logs/reptile",
        task_perturbation_scale=1.1e-04,
        verbose=True,
    )
    
    # Test 2: Natural gradients
    # print("\n2. Testing with natural gradients:")
    # maml_natural = HessianFreeMAML(
    #     environment_class=Boeing_dynamics,
    #     environment_kwargs={'noise_scale': 0.05, 'max_steps': 100},
    #     state_dim=4,
    #     action_dim=2,
    #     inner_lr=8e-6,
    #     outer_lr=8e-6,
    #     perturbation_scale=0.01,
    #     num_inner_steps=1,
    #     num_perturbations=30,
    #     rollout_length=100,
    #     use_natural_gradients=True,  # Enable natural gradients
    #     random_seed=42
    # )

    # history_natural = maml_natural.train(
    #     num_tasks=3,
    #     num_epochs=20,
    #     eval_interval=5,
    #     log_dir="logs/natural_gradients",
    #     verbose=True
    # )

    # # Compare results
    # print("\n" + "=" * 50)
    # print("COMPARISON RESULTS:")
    # print(f"Regular gradients  - Final cost: {history_regular['mean_cost'][-1]:.4f}, "
    #       f"Final regret: {history_regular['regret'][-1]:.4f}")
    # print(f"Natural gradients  - Final cost: {history_natural['mean_cost'][-1]:.4f}, "
    #       f"Final regret: {history_natural['regret'][-1]:.4f}")

    # # Simple convergence comparison
    # regular_improvement = history_regular['mean_cost'][0] - history_regular['mean_cost'][-1]
    # natural_improvement = history_natural['mean_cost'][0] - history_natural['mean_cost'][-1]

    # print(f"\nImprovement (cost reduction):")
    # print(f"Regular gradients: {regular_improvement:.4f}")
    # print(f"Natural gradients: {natural_improvement:.4f}")

   