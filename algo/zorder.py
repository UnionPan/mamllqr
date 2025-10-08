"""
Zeroth-Order MAML (ZOMAML) Algorithm for LQR Control.

Implementation of ZOMAML using explicit Hessian estimation with the
(U @ U.T - I) formulation for second-order meta-learning updates.

This differs from HessianFreeMAML by explicitly computing the Hessian matrix
and using it in the meta-gradient computation: (I - eta_inner * H) @ gradient
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path
import pickle
import time
from tensorboardX import SummaryWriter


from utils.utils import (
    generate_perturbations, zeroth_order_gradient, zeroth_order_hessian_diag,
    clip_gradient, evaluate_policy_performance, compute_regret, Timer,
    log_metrics, check_stability, estimate_covariance_matrix,
    natural_gradient_with_covariance
)


class ZOMAML:
    """
    Zeroth-Order Model-Agnostic Meta-Learning for LQR Control.

    This class implements ZOMAML with explicit Hessian estimation using the
    (U @ U.T - I) formulation. The key difference from HessianFreeMAML is:

    Meta-gradient = _tasks [(I - eta_inner * H_task) @ grad_task]

    where H_task is the estimated Hessian matrix.
    """

    def __init__(
        self,
        environment_class,
        environment_kwargs: Dict[str, Any] = None,
        state_dim: int = 4,
        action_dim: int = 2,
        inner_lr: float = 1e-3,
        outer_lr: float = 1e-3,
        perturbation_scale: float = 0.01,
        num_inner_steps: int = 3,
        num_perturbations: int = 100,
        rollout_length: int = 30,
        use_natural_gradients: bool = False,
        grad_clip: float = 10.0,
        hessian_damping: float = 1e-4,
        random_seed: Optional[int] = None
    ):
        """
        Initialize ZOMAML algorithm.

        Parameters:
        environment_class: Environment class (Boeing_dynamics or BaseLQREnv)
        environment_kwargs: Keyword arguments for environment initialization
        state_dim: Dimensionality of state space
        action_dim: Dimensionality of action space
        inner_lr: Learning rate for inner loop (task adaptation) - eta_inner
        outer_lr: Learning rate for outer loop (meta-learning) - eta_outer
        perturbation_scale: Scale of perturbations for gradient estimation (r)
        num_inner_steps: Number of inner loop gradient steps
        num_perturbations: Number of perturbations for gradient/Hessian estimation
        rollout_length: Length of rollouts for cost estimation
        use_natural_gradients: Whether to use natural gradient updates
        grad_clip: Maximum gradient norm (for stability)
        hessian_damping: Damping factor for Hessian regularization
        random_seed: Random seed for reproducibility
        """
        # Store parameters
        self.environment_class = environment_class
        self.environment_kwargs = environment_kwargs or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr  # eta_inner
        self.outer_lr = outer_lr  # eta_outer
        self.perturbation_scale = perturbation_scale
        self.num_inner_steps = num_inner_steps
        self.num_perturbations = num_perturbations
        self.rollout_length = rollout_length
        self.use_natural_gradients = use_natural_gradients
        self.grad_clip = grad_clip
        self.hessian_damping = hessian_damping

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize policy parameters (LQR gain matrix)
        K_opt_nominal = self.environment_class(**self.environment_kwargs).get_optimal_K()
        self.K = K_opt_nominal + 0.25 * np.random.rand(action_dim, state_dim)
        
        self.param_dim = action_dim * state_dim

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
        This is similar to HessianFreeMAML but we'll track gradients for meta-learning.

        Parameters:
        env: Task environment
        K_init: Initial policy parameters

        Returns:
        Tuple[np.ndarray, List[float]]: (adapted_policy, cost_history)
        """
        K = K_init.copy()
        cost_history = []

        for step in range(self.num_inner_steps):
            # Generate perturbations for gradient estimation
            perturbations = generate_perturbations(
                K.shape, self.num_perturbations, self.perturbation_scale
            )

            # Estimate gradient using zeroth-order method
            gradient = zeroth_order_gradient(
                env, K, perturbations, self.perturbation_scale,
                self.rollout_length, num_trajectories=1
            )

            # Apply natural gradients if enabled
            if self.use_natural_gradients:
                covariance = estimate_covariance_matrix(
                    env, K, num_trajectories=30, rollout_length=self.rollout_length
                )
                gradient = natural_gradient_with_covariance(gradient, covariance)

            # Clip gradient for stability
            gradient = clip_gradient(gradient, self.grad_clip)

            # Inner loop update
            K = K - self.inner_lr * gradient

            # Evaluate current policy
            try:
                cost = env.analytical_cost(K)
                cost_history.append(cost)
            except:
                cost_history.append(float('inf'))
                break

        return K, cost_history

    def compute_meta_gradient_zomaml(self, task_envs: List, K_meta: np.ndarray) -> np.ndarray:
        """
        Compute meta-gradient using ZOMAML with explicit Hessian estimation.

        The ZOMAML meta-gradient for each task is:
        meta_grad_task = (I - eta_inner * H_task) @ grad_adapted_task

        Parameters:
        task_envs: List of task environments
        K_meta: Current meta-policy parameters

        Returns:
        np.ndarray: Meta-gradient for policy update
        """
        meta_gradient_flat = np.zeros(self.param_dim)  # Flattened meta-gradient

        # Sample subset of tasks for this update
        num_tasks_batch = min(len(task_envs), max(1, len(task_envs) // 2))
        task_indices = np.random.choice(len(task_envs), num_tasks_batch, replace=False)

        for task_idx in task_indices:
            env = task_envs[task_idx]

            # Step 1: Adapt to this task
            K_adapted, _ = self.inner_loop_update(env, K_meta)

            # Step 2: Compute gradient at adapted policy
            perturbations = generate_perturbations(
                K_adapted.shape, self.num_perturbations, self.perturbation_scale
            )

            gradient_adapted = zeroth_order_gradient(
                env, K_adapted, perturbations, self.perturbation_scale,
                self.rollout_length, num_trajectories=30
            )

            # Step 3: Estimate Hessian at meta-policy (not adapted policy!)
            hessian_perturbations = generate_perturbations(
                K_meta.shape, self.num_perturbations, self.perturbation_scale
            )

            hessian_matrix = zeroth_order_hessian_diag(
                env, K_meta, hessian_perturbations, self.perturbation_scale,
                self.rollout_length, num_trajectories=30
            )

            # Step 4: Apply ZOMAML formula (I - eta_inner * H) @ gradient
            # IMPORTANT: Both hessian and gradient must be in flattened space!

            # Flatten gradient
            gradient_flat = gradient_adapted.flatten()  # (param_dim,)

            # Add damping to Hessian for numerical stability
            I_matrix = np.eye(self.param_dim)
            hessian_regularized = hessian_matrix + self.hessian_damping * I_matrix

            # Compute (I - eta_inner * H)
            maml_matrix = I_matrix - self.inner_lr * hessian_regularized

            # Meta-gradient contribution: (I - eta_inner * H) @ gradient
            task_meta_gradient = maml_matrix @ gradient_flat

            meta_gradient_flat += task_meta_gradient

        # Average across tasks
        meta_gradient_flat = meta_gradient_flat / num_tasks_batch

        # Reshape back to K shape
        meta_gradient = meta_gradient_flat.reshape(K_meta.shape)

        return meta_gradient

    def outer_loop_update(self, task_envs: List, K_meta: np.ndarray) -> np.ndarray:
        """
        Perform ZOMAML meta-update across tasks.

        Parameters:
        task_envs: List of task environments
        K_meta: Current meta-policy parameters

        Returns:
        np.ndarray: Updated meta-policy parameters
        """
        # Compute meta-gradient using ZOMAML
        meta_gradient = self.compute_meta_gradient_zomaml(task_envs, K_meta)

        # Clip meta-gradient
        meta_gradient = clip_gradient(meta_gradient, self.grad_clip)

        # Meta-update
        K_meta_new = K_meta - self.outer_lr * meta_gradient

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
        Train the ZOMAML algorithm.

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
        if log_dir and SummaryWriter is not None:
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
            'meta_gradient_norm': [],
            'hessian_condition_number': []
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            with Timer(f"Epoch {epoch}" if verbose and epoch % 10 == 0 else ""):
                # Meta-update using ZOMAML
                K_old = self.K.copy()
                self.K = self.outer_loop_update(task_envs, self.K)

                # Compute meta-gradient norm for monitoring
                meta_grad_norm = np.linalg.norm(self.K - K_old) / self.outer_lr

            # Evaluation
            if epoch % eval_interval == 0:
                # Evaluate current meta-policy on all tasks
                all_costs = []  # Costs after adaptation
                all_zero_shot_costs = []  # Costs before adaptation
                hessian_conditions = []

                for env in task_envs:
                    # Evaluate zero-shot (meta-policy directly, no adaptation)
                    zero_shot_cost = env.analytical_cost(self.K)
                    #print(self.K)
                    if not np.isinf(zero_shot_cost):
                        all_zero_shot_costs.append(zero_shot_cost)

                   # Quick adaptation
                    K_adapted, _ = self.inner_loop_update(env, self.K)

                    # Evaluate adapted cost
                    adapted_cost = env.analytical_cost(K_adapted)
                    
                    if not np.isinf(adapted_cost):
                        all_costs.append(adapted_cost)

                    # Monitor Hessian condition number (for diagnostic)
                    try:
                        hess_perturbations = generate_perturbations(
                            self.K.shape, min(20, self.num_perturbations),
                            self.perturbation_scale
                        )
                        hessian = zeroth_order_hessian_diag(
                            env, self.K, hess_perturbations, self.perturbation_scale,
                            self.rollout_length, num_trajectories=1
                        )
                        cond_num = np.linalg.cond(hessian + self.hessian_damping * np.eye(self.param_dim))
                        hessian_conditions.append(cond_num)
                    except:
                        pass

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
                    hess_cond = np.mean(hessian_conditions) if hessian_conditions else float('inf')

                    # Store history
                    training_history['epoch'].append(epoch)
                    training_history['mean_cost'].append(mean_cost)
                    training_history['cost_std'].append(cost_std)
                    training_history['zero_shot_cost'].append(zero_shot_mean)
                    training_history['zero_shot_cost_std'].append(zero_shot_std)
                    training_history['regret'].append(regret)
                    training_history['policy_diff'].append(policy_diff)
                    training_history['meta_gradient_norm'].append(meta_grad_norm)
                    training_history['hessian_condition_number'].append(hess_cond)

                    if verbose:
                        print(f"Epoch {epoch:4d}: Adapted={mean_cost:.4f}±{cost_std:.6f}, "
                              f"Zero-shot={zero_shot_mean:.4f}±{zero_shot_std:.6f}, "
                              f"Regret={regret:.4f}, Hess_cond={hess_cond:.2e}")

                    # Log to tensorboard
                    if self.logger:
                        metrics = {
                            'cost/adapted_mean': mean_cost,
                            'cost/adapted_std': cost_std,
                            'cost/zero_shot_mean': zero_shot_mean,
                            'cost/zero_shot_std': zero_shot_std,
                            'regret': regret,
                            'policy_difference': policy_diff,
                            'meta_gradient_norm': meta_grad_norm,
                            'hessian_condition_number': hess_cond
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
            history_json = {k: [float(x) if not np.isnan(x) and not np.isinf(x) else None
                               for x in v] for k, v in training_history.items()}
            json_path = Path(log_dir) / "training_history.json"
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
                'hessian_damping': self.hessian_damping
            }
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


if __name__ == "__main__":
    # Example usage and comparison with HessianFreeMAML
    from envs.Boeing import Boeing_dynamics

    print("Testing ZOMAML with Boeing dynamics...")
    print("=" * 60)

    # Test ZOMAML
    print("\n Testing ZOMAML (with explicit Hessian):")
    zomaml = ZOMAML(
        environment_class=Boeing_dynamics,
        environment_kwargs={'noise_scale': 0.05, 'max_steps': 100},
        state_dim=4,
        action_dim=2,
        inner_lr=8e-4,
        outer_lr=5e-2,
        perturbation_scale=1.1e-03,
        num_inner_steps=1,  # Reduced for faster testing
        num_perturbations=50,  # Reduced for faster testing
        rollout_length=100,
        use_natural_gradients=True,
        hessian_damping=1e-4,
        random_seed=2025
    )

    history_zomaml = zomaml.train(
        num_tasks=10,
        num_epochs=1000,
        eval_interval=1,
        log_dir="logs/zomaml_test",
        task_perturbation_scale=1.1e-04,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("ZOMAML RESULTS:")
    print(f"Final cost: {history_zomaml['mean_cost'][-1]:.4f}")
    print(f"Final regret: {history_zomaml['regret'][-1]:.4f}")
    print(f"Final Hessian condition number: {history_zomaml['hessian_condition_number'][-1]:.2e}")

    # Cost improvement
    improvement = history_zomaml['mean_cost'][0] - history_zomaml['mean_cost'][-1]
    print(f"Cost improvement: {improvement:.4f}")

    print("\nTraining completed! Check logs/zomaml_test/ for detailed results.")
    print("Key features of ZOMAML:")
    print("- Explicit Hessian estimation with (U @ U.T - I)")
    print("- Meta-gradient: (I - eta * H) @ gradient")
    print("- Proper flattening/reshaping of parameters")
    print("- Hessian condition number monitoring")