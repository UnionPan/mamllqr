#!/usr/bin/env python3
"""
Experimental Comparison of Meta-Learning Algorithms for LQR Control.

Compares:
- HessianFree MAML
- ZOMAML

Metrics:
- Training curves (adapted cost, zero-shot cost)
- Meta-testing performance with different few-shot adaptation steps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
import argparse

from envs.Boeing import Boeing_dynamics
from algo.hessianfree import HessianFreeMAML
from algo.zorder import ZOMAML

# Set plotting aesthetics
sns.set_style("darkgrid")
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f8f8',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.2,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.8,
    'text.color': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# color palette
COLORS = {
    'HessianFree': '#2E86AB',     # Deep blue
    'HessianFree_MAML': '#2E86AB', # Deep blue (alternative name)
    'ZOMAML': '#A23B72',          # Deep magenta
    'Combined': '#F18F01',        # Orange (if needed)
    'Reptile':'#C73E1D'         # Deep red (if needed)
}


def create_shared_initial_policy(environment_class, environment_kwargs,
                                state_dim, action_dim, seed=2025):
    """
    Create a shared initial policy for fair comparison.

    Returns:
    np.ndarray: Initial policy matrix K (action_dim x state_dim)
    """
    np.random.seed(seed)

    # Create environment to get optimal policy as reference
    env = environment_class(**environment_kwargs)
    K_opt = env.get_optimal_K()

    # Initialize as optimal + noise for realistic starting point
    K_init = K_opt + 0.25 * np.random.normal(0, 0.5, (action_dim, state_dim))

    return K_init


def create_test_environments(environment_class, environment_kwargs,
                            num_test_tasks=5, test_perturbation_scale=1e-4):
    """
    Create separate test environments for meta-testing evaluation.
    """
    base_env = environment_class(**environment_kwargs)
    test_envs = base_env.get_task_variants(
        num_variants=num_test_tasks,
        perturbation_scale=test_perturbation_scale
    )
    return test_envs


def run_single_algorithm_round(algorithm_class, algorithm_params, K_init, common_params,
                              num_tasks, num_epochs, eval_interval, log_dir_base,
                              task_perturbation_scale, round_num, is_combined=False):
    """Run a single round of training for an algorithm."""

    # Create algorithm instance
    algorithm = algorithm_class(**algorithm_params, **common_params)
    algorithm.K = K_init.copy()

    # Create unique log directory for this round
    log_dir = f"{log_dir_base}/round_{round_num}"

    start_time = time.time()
    if is_combined:
        history = algorithm.train_combine(
            reptile_weight=args.reptile_weight,
            num_tasks=num_tasks,
            num_epochs=num_epochs,
            eval_interval=eval_interval,
            log_dir=log_dir,
            task_perturbation_scale=task_perturbation_scale,
            verbose=False
        )
    else:
        history = algorithm.train(
            num_tasks=num_tasks,
            num_epochs=num_epochs,
            eval_interval=eval_interval,
            log_dir=log_dir,
            task_perturbation_scale=task_perturbation_scale,
            verbose=False
        )
    training_time = time.time() - start_time

    return history, training_time, algorithm


def run_multiple_rounds(algorithm_name, algorithm_class, algorithm_params, K_init_template,
                       common_params, num_tasks, num_epochs, eval_interval,
                       task_perturbation_scale, num_rounds=10, is_combined=False):
    """Run multiple rounds of training for an algorithm."""

    print(f"\n--- Running {num_rounds} rounds of {algorithm_name} ---")

    all_histories = []
    all_training_times = []
    all_trained_algorithms = []  # Store algorithms after certain epochs

    for round_num in range(num_rounds):
        print(f"  Round {round_num + 1}/{num_rounds}")

        # Create fresh initial policy for this round (same distribution, different noise)
        np.random.seed(2025 + round_num)  # Different seed for each round
        env = common_params['environment_class'](**common_params['environment_kwargs'])
        K_opt = env.get_optimal_K()
        K_init = K_opt + 0.25 * np.random.normal(0, 0.5, K_init_template.shape)

        log_dir_base = f"logs/comparison/{algorithm_name.lower()}"

        history, training_time, trained_algorithm = run_single_algorithm_round(
            algorithm_class, algorithm_params, K_init, common_params,
            num_tasks, num_epochs, eval_interval, log_dir_base,
            task_perturbation_scale, round_num, is_combined
        )

        # Get initial policy cost for this round
        initial_cost = env.analytical_cost(K_init)

        # Prepend initial policy performance
        history['epoch'] = [0] + history['epoch']
        history['mean_cost'] = [initial_cost] + history['mean_cost']
        history['zero_shot_cost'] = [initial_cost] + history['zero_shot_cost']
        history['cost_std'] = [0.0] + history['cost_std']
        history['zero_shot_cost_std'] = [0.0] + history['zero_shot_cost_std']

        all_histories.append(history)
        all_training_times.append(training_time)
        all_trained_algorithms.append(trained_algorithm)

    return all_histories, all_training_times, all_trained_algorithms


def evaluate_few_shot_adaptation(algorithm, test_envs, adaptation_steps_list=[0, 1, 3, 5]):
    """
    Evaluate algorithm's few-shot adaptation performance.

    Parameters:
    algorithm: Trained meta-learning algorithm
    test_envs: List of test environments
    adaptation_steps_list: Different numbers of adaptation steps to test

    Returns:
    Dict: Results for each adaptation step count
    """
    results = {}

    for num_steps in adaptation_steps_list:
        step_costs = []

        for env in test_envs:
            # Adapt meta-policy to this test task
            if hasattr(algorithm, 'adapt_to_task'):
                K_adapted = algorithm.adapt_to_task(env, num_adaptation_steps=num_steps)
            else:
                # For ZOMAML, use inner_loop_update
                if num_steps == 0:
                    K_adapted = algorithm.K.copy()
                else:
                    # Temporarily set num_inner_steps
                    original_steps = algorithm.num_inner_steps
                    algorithm.num_inner_steps = num_steps
                    K_adapted, _ = algorithm.inner_loop_update(env, algorithm.K)
                    algorithm.num_inner_steps = original_steps

            # Evaluate adapted policy cost
            cost = env.analytical_cost(K_adapted)
            if not np.isinf(cost):
                step_costs.append(cost)

        results[num_steps] = {
            'mean_cost': np.mean(step_costs) if step_costs else float('inf'),
            'std_cost': np.std(step_costs) if step_costs else float('inf'),
            'num_tasks': len(step_costs)
        }

    return results


def run_experimental_comparison():
    """
    Main experimental comparison function.
    """
    print("=" * 60)
    print("EXPERIMENTAL COMPARISON: Meta-Learning for LQR Control")
    print("=" * 60)

    # Experimental setup
    environment_class = Boeing_dynamics
    environment_kwargs = {'noise_scale': 0.05, 'max_steps': 100}
    state_dim = 4
    action_dim = 2
    num_tasks = 10
    num_epochs = 3
    eval_interval = 1

    # Create shared initial policy for fair comparison
    print(f"\n 1. Creating shared initial policy (seed=2025)...")
    K_init = create_shared_initial_policy(
        environment_class, environment_kwargs, state_dim, action_dim, seed=2025
    )
    print(f"Initial policy shape: {K_init.shape}")
    print(f"Initial policy norm: {np.linalg.norm(K_init):.4f}")

    # Create test environments for meta-testing
    print(f"\n 2. Creating test environments...")
    test_envs = create_test_environments(
        environment_class, environment_kwargs,
        num_test_tasks=8, test_perturbation_scale=1e-4
    )
    print(f"Created {len(test_envs)} test environments")

    # Evaluate initial policy on test environments
    print(f"\n  Evaluating initial policy on test environments...")
    initial_policy_costs = []
    for env in test_envs:
        cost = env.analytical_cost(K_init)
        if not np.isinf(cost):
            initial_policy_costs.append(cost)

    initial_policy_mean = np.mean(initial_policy_costs) if initial_policy_costs else float('inf')
    print(f"  Initial policy mean cost on test tasks: {initial_policy_mean:.4f}")

    # Common hyperparameters for fair comparison (matching testing sections)
    common_params = {
        'environment_class': environment_class,
        'environment_kwargs': environment_kwargs,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'inner_lr': 8e-5,  # From hessianfree.py test
        'outer_lr': 5e-3,  # From hessianfree.py test
        'num_inner_steps': 1,
        'rollout_length': 100,
        'use_natural_gradients': True,
        'random_seed': 2025
    }

    results = {}

    # Algorithm 1: HessianFree MAML (actually Combined method)
    print(f"\n 3. Training HessianFree MAML...")
    hf_maml = HessianFreeMAML(
        perturbation_scale=1.1e-3,  
        inner_perturbation_scale=1.1e-5,  
        num_inner_perturbations=150,  
        num_perturbations=50,  
        use_adam=False,  
        use_analytical_gradients=True,
        **common_params
    )
    # Set shared initial policy
    hf_maml.K = K_init.copy()

    start_time = time.time()
    history_hf = hf_maml.train_combine(
        reptile_weight=0.5,
        num_tasks=num_tasks,
        num_epochs=num_epochs,
        eval_interval=eval_interval,
        log_dir="logs/comparison/hf_maml",
        task_perturbation_scale=1.1e-4,
        verbose=True
    )
    hf_time = time.time() - start_time

    # Prepend initial policy performance to training history
    history_hf['epoch'] = [0] + history_hf['epoch']
    history_hf['mean_cost'] = [initial_policy_mean] + history_hf['mean_cost']
    history_hf['zero_shot_cost'] = [initial_policy_mean] + history_hf['zero_shot_cost']
    history_hf['cost_std'] = [0.0] + history_hf['cost_std']
    history_hf['zero_shot_cost_std'] = [0.0] + history_hf['zero_shot_cost_std']

    # Meta-testing
    hf_few_shot = evaluate_few_shot_adaptation(hf_maml, test_envs)

    results['HessianFree_MAML'] = {
        'training_history': history_hf,
        'few_shot_results': hf_few_shot,
        'training_time': hf_time
    }
    print(f"  Training time: {hf_time:.2f}s")
    print(f"  Final adapted cost: {history_hf['mean_cost'][-1]:.4f}")

    # Algorithm 2: ZOMAML
    print(f"\n4. Training ZOMAML...")
    zomaml = ZOMAML(
        perturbation_scale=1.1e-3,  # From zorder.py test
        num_perturbations=50,  # From zorder.py test
        hessian_damping=1e-4,  # From zorder.py test
        inner_lr=8e-4,  # From zorder.py test (different from hessianfree)
        outer_lr=5e-2,  # From zorder.py test (different from hessianfree)
        **{k: v for k, v in common_params.items() if k not in ['inner_lr', 'outer_lr']}
    )
    # Set shared initial policy
    zomaml.K = K_init.copy()

    start_time = time.time()
    history_zo = zomaml.train(
        num_tasks=num_tasks,
        num_epochs=num_epochs,
        eval_interval=eval_interval,
        log_dir="logs/comparison/zomaml",
        task_perturbation_scale=1.1e-4,  # From testing sections
        verbose=True
    )
    zo_time = time.time() - start_time

    # Prepend initial policy performance to training history
    history_zo['epoch'] = [0] + history_zo['epoch']
    history_zo['mean_cost'] = [initial_policy_mean] + history_zo['mean_cost']
    history_zo['zero_shot_cost'] = [initial_policy_mean] + history_zo['zero_shot_cost']
    history_zo['cost_std'] = [0.0] + history_zo['cost_std']
    history_zo['zero_shot_cost_std'] = [0.0] + history_zo['zero_shot_cost_std']

    # Meta-testing
    zo_few_shot = evaluate_few_shot_adaptation(zomaml, test_envs)

    results['ZOMAML'] = {
        'training_history': history_zo,
        'few_shot_results': zo_few_shot,
        'training_time': zo_time
    }
    print(f"  Training time: {zo_time:.2f}s")
    print(f"  Final adapted cost: {history_zo['mean_cost'][-1]:.4f}")

    return results


def create_algorithm_at_epoch(algorithm_class, algorithm_params, common_params,
                            K_init, target_epoch=50, is_combined=False):
    """Create an algorithm instance with meta-policy at target epoch."""

    # Create fresh algorithm
    algorithm = algorithm_class(**algorithm_params, **common_params)
    algorithm.K = K_init.copy()

    if target_epoch > 0:
        # Run training up to the target epoch
        if is_combined:
            # For combined method, run partial training
            algorithm.train_combine(
                reptile_weight=args.reptile_weight,
                num_tasks=10,  # Same as main training
                num_epochs=target_epoch,
                eval_interval=target_epoch,  # Only evaluate at the end
                log_dir=None,
                task_perturbation_scale=1.1e-4,
                verbose=False
            )
        else:
            # For ZOMAML, run partial training
            algorithm.train(
                num_tasks=10,  # Same as main training
                num_epochs=target_epoch,
                eval_interval=target_epoch,  # Only evaluate at the end
                log_dir=None,
                task_perturbation_scale=1.1e-4,
                verbose=False
            )

    return algorithm


def aggregate_training_histories(all_histories):
    """Aggregate multiple training histories into mean and std."""
    if not all_histories:
        return {}

    # Get common epochs (should be same across all runs)
    epochs = all_histories[0]['epoch']

    # Aggregate each metric
    aggregated = {'epoch': epochs}

    metrics = ['mean_cost', 'zero_shot_cost', 'cost_std', 'zero_shot_cost_std']
    for metric in metrics:
        # Stack all runs for this metric
        all_values = np.array([history[metric] for history in all_histories])

        # Compute mean and std across runs
        aggregated[f'{metric}_mean'] = np.mean(all_values, axis=0)
        aggregated[f'{metric}_std'] = np.std(all_values, axis=0)

        # Also keep raw values for seaborn plotting
        aggregated[f'{metric}_all'] = all_values

    return aggregated


def create_seaborn_dataframe(results_multi):
    """Create a DataFrame suitable for seaborn plotting."""
    data_rows = []

    for algorithm_name, data in results_multi.items():
        aggregated = data['aggregated_history']
        epochs = aggregated['epoch']

        # For adapted cost
        for run_idx, run_values in enumerate(aggregated['mean_cost_all']):
            for epoch_idx, (epoch, cost) in enumerate(zip(epochs, run_values)):
                data_rows.append({
                    'Algorithm': algorithm_name,
                    'Epoch': epoch,
                    'Cost': cost,
                    'Metric': 'Adapted Cost',
                    'Run': run_idx
                })

        # For zero-shot cost
        for run_idx, run_values in enumerate(aggregated['zero_shot_cost_all']):
            for epoch_idx, (epoch, cost) in enumerate(zip(epochs, run_values)):
                data_rows.append({
                    'Algorithm': algorithm_name,
                    'Epoch': epoch,
                    'Cost': cost,
                    'Metric': 'Zero-shot Cost',
                    'Run': run_idx
                })

    return pd.DataFrame(data_rows)


def plot_comparison_results_multifold(results_multi, target_epoch=50):
    """
    Plot comparison results using RL-style aesthetics with mean ± std curves.
    """
    # Create DataFrame for seaborn
    df = create_seaborn_dataframe(results_multi)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')

    # Plot 1: Training curves - Adapted Cost
    ax1 = axes[0]
    adapted_df = df[df['Metric'] == 'Adapted Cost']
    sns.lineplot(data=adapted_df, x='Epoch', y='Cost', hue='Algorithm',
                ax=ax1, linewidth=3, palette=COLORS, alpha=0.9,
                marker='o', markersize=3, markevery=10)
    ax1.set_xlabel('Training Epoch', fontweight='semibold')
    ax1.set_ylabel('Adapted Cost', fontweight='semibold')
    ax1.set_title('Meta-Training: Task Adaptation Performance', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax1.set_facecolor('#fafafa')
    legend1 = ax1.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    legend1.get_frame().set_facecolor('white')

    # Plot 2: Training curves - Zero-shot Cost
    ax2 = axes[1]
    zeroshot_df = df[df['Metric'] == 'Zero-shot Cost']
    sns.lineplot(data=zeroshot_df, x='Epoch', y='Cost', hue='Algorithm',
                ax=ax2, linewidth=3, palette=COLORS, alpha=0.9,
                marker='s', markersize=3, markevery=10)
    ax2.set_xlabel('Training Epoch', fontweight='semibold')
    ax2.set_ylabel('Zero-shot Cost', fontweight='semibold')
    ax2.set_title('Meta-Training: Zero-shot Performance', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax2.set_facecolor('#fafafa')
    legend2 = ax2.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    legend2.get_frame().set_facecolor('white')

    # Plot 3: Few-shot adaptation performance with enhanced styling
    ax3 = axes[2]
    adaptation_steps = [0, 1, 3, 5]

    for algorithm_name, data in results_multi.items():
        # Aggregate few-shot results across all runs
        all_few_shot = data['all_few_shot_results']

        mean_costs = []
        std_costs = []

        for steps in adaptation_steps:
            step_costs = []
            for few_shot in all_few_shot:
                step_costs.append(few_shot[steps]['mean_cost'])

            mean_costs.append(np.mean(step_costs))
            std_costs.append(np.std(step_costs))

        color = COLORS.get(algorithm_name, '#666666')
        ax3.errorbar(adaptation_steps, mean_costs, yerr=std_costs,
                    label=algorithm_name, marker='D', linewidth=3,
                    capsize=6, capthick=2.5, markersize=7, color=color,
                    elinewidth=2.5, alpha=0.9)

    ax3.set_xlabel('Gradient Steps', fontweight='semibold')
    ax3.set_ylabel('Test Performance', fontweight='semibold')
    ax3.set_title(f'Few-shot Adaptation\n(Meta-policy from Epoch {target_epoch})',
                  fontweight='bold', pad=15)
    ax3.set_facecolor('#fafafa')
    ax3.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    legend3 = ax3.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    legend3.get_frame().set_facecolor('white')

    # Enhanced subplot spacing and borders
    plt.tight_layout(pad=2.0)

    # Add subtle border around entire figure
    for spine in ['top', 'right', 'bottom', 'left']:
        for ax in axes:
            ax.spines[spine].set_color('#cccccc')
            ax.spines[spine].set_linewidth(1.2)

    return fig  # Return figure for saving with custom name


def parse_arguments():
    """Parse command line arguments for experimental comparison."""
    parser = argparse.ArgumentParser(
        description="Meta-Learning Experimental Comparison for LQR Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument("--num_rounds", type=int, default=10,
                       help="Number of training rounds for each algorithm")
    parser.add_argument("--num_epochs", type=int, default=300,
                       help="Number of training epochs per round")
    parser.add_argument("--num_tasks", type=int, default=10,
                       help="Number of tasks per training round")
    parser.add_argument("--eval_interval", type=int, default=1,
                       help="Evaluation interval during training")
    parser.add_argument("--target_epoch", type=int, default=25,
                       help="Epoch to use for few-shot adaptation evaluation")

    # Environment parameters
    parser.add_argument("--noise_scale", type=float, default=0.05,
                       help="Noise scale for environment")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--num_test_tasks", type=int, default=8,
                       help="Number of test tasks for meta-testing")
    parser.add_argument("--test_perturbation_scale", type=float, default=1e-4,
                       help="Perturbation scale for test environments")

    # Algorithm hyperparameters
    parser.add_argument("--inner_lr", type=float, default=8e-5,
                       help="Inner loop learning rate")
    parser.add_argument("--outer_lr_hf", type=float, default=5e-3,
                       help="Outer loop learning rate for HessianFree")
    parser.add_argument("--outer_lr_zo", type=float, default=5e-2,
                       help="Outer loop learning rate for ZOMAML")
    parser.add_argument("--perturbation_scale", type=float, default=1.1e-3,
                       help="Perturbation scale for gradients")
    parser.add_argument("--reptile_weight", type=float, default=0.5,
                       help="Weight for Reptile in combined method (0=pure MAML, 1=pure Reptile)")

    # Output parameters
    parser.add_argument("--output_prefix", type=str, default="meta_learning_comparison",
                       help="Prefix for output plot filename")
    parser.add_argument("--seed", type=int, default=2025,
                       help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output during training")

    return parser.parse_args()


def run_multifold_experimental_comparison(args):
    """
    Main experimental comparison function with multiple rounds.
    """
    print("=" * 70)
    print("MULTIFOLD EXPERIMENTAL COMPARISON: Meta-Learning for LQR Control")
    print(f"Running {args.num_rounds} rounds for each algorithm")
    print("=" * 70)

    # Experimental setup
    environment_class = Boeing_dynamics
    environment_kwargs = {'noise_scale': args.noise_scale, 'max_steps': args.max_steps}
    state_dim = 4
    action_dim = 2
    num_tasks = args.num_tasks
    num_epochs = args.num_epochs
    eval_interval = args.eval_interval

    # Create template initial policy for shape reference
    print(f"\n1. Setting up experimental parameters...")
    print(f"   Seed: {args.seed}, Epochs: {num_epochs}, Rounds: {args.num_rounds}")
    print(f"   Target epoch for few-shot: {args.target_epoch}")
    np.random.seed(args.seed)
    env = environment_class(**environment_kwargs)
    K_opt = env.get_optimal_K()
    K_init_template = K_opt + 0.25 * np.random.normal(0, 0.5, (action_dim, state_dim))

    # Create test environments for meta-testing
    print(f"\n2. Creating test environments...")
    test_envs = create_test_environments(
        environment_class, environment_kwargs,
        num_test_tasks=args.num_test_tasks,
        test_perturbation_scale=args.test_perturbation_scale
    )
    print(f"Created {len(test_envs)} test environments")

    # Common hyperparameters for fair comparison
    common_params = {
        'environment_class': environment_class,
        'environment_kwargs': environment_kwargs,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'inner_lr': args.inner_lr,
        'outer_lr': args.outer_lr_hf,
        'num_inner_steps': 1,
        'rollout_length': 100,
        'use_natural_gradients': True,
        'random_seed': args.seed
    }

    # Algorithm parameters
    hf_params = {
        'perturbation_scale': args.perturbation_scale,
        'inner_perturbation_scale': args.perturbation_scale / 100,  # 100x smaller
        'num_inner_perturbations': 150,
        'num_perturbations': 50,
        'use_adam': False,
        'use_analytical_gradients': True,
    }

    zo_params = {
        'perturbation_scale': args.perturbation_scale,
        'num_perturbations': 50,
        'hessian_damping': 1e-4,
        'inner_lr': args.inner_lr * 10,  # 10x higher for ZOMAML
        'outer_lr': args.outer_lr_zo,
    }

    results_multi = {}

    # Create ZOMAML-specific common params (avoid parameter conflicts)
    zo_common_params = {k: v for k, v in common_params.items()
                       if k not in ['inner_lr', 'outer_lr']}
    zo_common_params['use_natural_gradients'] = True  # Add it explicitly

    # Run HessianFree MAML (Combined) multiple rounds
    print(f"\n3. Running HessianFree MAML...")
    hf_histories, hf_times, hf_trained_algorithms = run_multiple_rounds(
        "HessianFree_MAML", HessianFreeMAML, hf_params, K_init_template,
        common_params, num_tasks, num_epochs, eval_interval,
        args.test_perturbation_scale * 10, args.num_rounds, is_combined=True
    )

    # Aggregate results
    hf_aggregated = aggregate_training_histories(hf_histories)

    # Run few-shot evaluation for each round (using meta-policy at target epoch)
    print(f"  Running few-shot evaluation using meta-policies at epoch {args.target_epoch}...")
    hf_few_shot_all = []
    for round_num in range(args.num_rounds):
        # Recreate the same initial policy used for this round
        np.random.seed(args.seed + round_num)
        env = environment_class(**environment_kwargs)
        K_opt = env.get_optimal_K()
        K_init = K_opt + 0.25 * np.random.normal(0, 0.5, K_init_template.shape)

        # Create algorithm with meta-policy at target epoch
        algorithm_target = create_algorithm_at_epoch(
            HessianFreeMAML, hf_params, common_params, K_init,
            target_epoch=args.target_epoch, is_combined=True
        )

        # Evaluate few-shot adaptation starting from the meta-policy at target epoch
        few_shot = evaluate_few_shot_adaptation(algorithm_target, test_envs)
        hf_few_shot_all.append(few_shot)

    results_multi['HessianFree_MAML'] = {
        'all_histories': hf_histories,
        'aggregated_history': hf_aggregated,
        'all_training_times': hf_times,
        'all_few_shot_results': hf_few_shot_all
    }

    # Run ZOMAML multiple rounds
    print(f"\n4. Running ZOMAML...")
    zo_histories, zo_times, zo_trained_algorithms = run_multiple_rounds(
        "ZOMAML", ZOMAML, zo_params, K_init_template,
        zo_common_params,
        num_tasks, num_epochs, eval_interval,
        args.test_perturbation_scale * 10, args.num_rounds, is_combined=False
    )

    # Aggregate results
    zo_aggregated = aggregate_training_histories(zo_histories)

    # Run few-shot evaluation for each round (using meta-policy at target epoch)
    print(f"  Running few-shot evaluation using meta-policies at epoch {args.target_epoch}...")
    zo_few_shot_all = []
    for round_num in range(args.num_rounds):
        # Recreate the same initial policy used for this round
        np.random.seed(args.seed + round_num)
        env = environment_class(**environment_kwargs)
        K_opt = env.get_optimal_K()
        K_init = K_opt + 0.25 * np.random.normal(0, 0.5, K_init_template.shape)

        # Create algorithm with meta-policy at target epoch
        algorithm_target = create_algorithm_at_epoch(
            ZOMAML, zo_params, zo_common_params,
            K_init, target_epoch=args.target_epoch, is_combined=False
        )

        # Evaluate few-shot adaptation starting from the meta-policy at target epoch
        few_shot = evaluate_few_shot_adaptation(algorithm_target, test_envs)
        zo_few_shot_all.append(few_shot)

    results_multi['ZOMAML'] = {
        'all_histories': zo_histories,
        'aggregated_history': zo_aggregated,
        'all_training_times': zo_times,
        'all_few_shot_results': zo_few_shot_all
    }

    return results_multi


def print_multifold_summary_table(results_multi):
    """
    Print a summary table of multifold results.
    """
    print("\n" + "=" * 80)
    print("MULTIFOLD EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)

    # Training performance
    print(f"\n{'Algorithm':<20} {'Final Adapted':<15} {'Final Zero-shot':<15} {'Avg Training Time':<15}")
    print("-" * 65)

    for name, data in results_multi.items():
        aggregated = data['aggregated_history']
        final_adapted_mean = aggregated['mean_cost_mean'][-1]
        final_adapted_std = aggregated['mean_cost_std'][-1]
        final_zero_shot_mean = aggregated['zero_shot_cost_mean'][-1]
        final_zero_shot_std = aggregated['zero_shot_cost_std'][-1]
        avg_training_time = np.mean(data['all_training_times'])

        print(f"{name:<20} {final_adapted_mean:.4f}±{final_adapted_std:.4f} "
              f"{final_zero_shot_mean:.4f}±{final_zero_shot_std:.4f} {avg_training_time:<15.1f}s")

    # Few-shot performance
    print(f"\n\nFEW-SHOT ADAPTATION PERFORMANCE (mean ± std)")
    print(f"Using meta-policies trained for {args.target_epoch} epochs")
    print("-" * 70)

    adaptation_steps = [0, 1, 3, 5]
    header = f"{'Algorithm':<20}"
    for steps in adaptation_steps:
        header += f"{steps}-shot{'':<15}"
    print(header)
    print("-" * (20 + 20 * len(adaptation_steps)))

    for name, data in results_multi.items():
        row = f"{name:<20}"
        all_few_shot = data['all_few_shot_results']

        for steps in adaptation_steps:
            step_costs = [few_shot[steps]['mean_cost'] for few_shot in all_few_shot]
            mean_cost = np.mean(step_costs)
            std_cost = np.std(step_costs)
            row += f"{mean_cost:.4f}±{std_cost:.3f}{'':<5}"
        print(row)
    plt.show()


def print_summary_table(results):
    """
    Print a summary table of results.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)

    # Training performance
    print(f"\n{'Algorithm':<20} {'Final Adapted':<15} {'Final Zero-shot':<15}")
    print("-" * 50)

    for name, data in results.items():
        history = data['training_history']
        final_adapted = history['mean_cost'][-1]
        final_zero_shot = history['zero_shot_cost'][-1]

        print(f"{name:<20} {final_adapted:<15.4f} {final_zero_shot:<15.4f}")

    # Few-shot performance
    print(f"\n \n FEW-SHOT ADAPTATION PERFORMANCE")
    print("-" * 50)

    adaptation_steps = [0, 1, 3, 5]
    header = f"{'Algorithm':<20}"
    for steps in adaptation_steps:
        header += f"{steps}-shot{'':<10}"
    print(header)
    print("-" * (20 + 15 * len(adaptation_steps)))

    for name, data in results.items():
        row = f"{name:<20}"
        few_shot = data['few_shot_results']
        for steps in adaptation_steps:
            cost = few_shot[steps]['mean_cost']
            row += f"{cost:<15.4f}"
        print(row)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Run multifold experimental comparison
    results_multi = run_multifold_experimental_comparison(args)

    # Print summary
    print("\n5. Results Summary...")
    print_multifold_summary_table(results_multi)

    # Plot results
    print("\n6. Generating plots...")
    fig = plot_comparison_results_multifold(results_multi, target_epoch=args.target_epoch)

    # Save plot with custom filename
    plot_filename = f"{args.output_prefix}_multifold.png"
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')

    # Show plot if not in batch mode
    if args.verbose:
        fig.show()

    print(f"\n" + "=" * 70)
    print("MULTIFOLD EXPERIMENTAL COMPARISON COMPLETED")
    print(f"Plots saved as: {plot_filename}")
    print("=" * 70)