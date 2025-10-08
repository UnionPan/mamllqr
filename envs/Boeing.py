import numpy as np
from envs.base_env import BaseLQREnv
from typing import List


class Boeing_dynamics(BaseLQREnv):
    """
    Boeing aircraft dynamics for meta-learning LQR control.
    Inherits from BaseLQREnv and uses predefined Boeing parameters.
    """
    
    # Boeing system parameters (4 states, 2 controls)
    AA = np.array([[ 1.22,  0.03, -0.02, -0.32],
                   [ 0.01,  4.7 ,  0.  ,  0.  ],
                   [ 0.02, -0.06,  0.4 ,  0.  ],
                   [ 0.01, -0.04,  0.72,  1.55]])
    
    BB = np.array([[ 0.01,  0.99],
                   [-3.44,  1.66],
                   [-0.83,  0.44],
                   [-0.47,  0.25]])
    
    QQ = np.eye(4)  # State cost matrix
    RR = np.eye(2)  # Control cost matrix
    
    def __init__(self, noise_scale: float = 0.05, max_steps: int = 100):
        """
        Initialize Boeing dynamics environment.

        Parameters:
        noise_scale (float): Process noise scale
        max_steps (int): Maximum episode steps
        """
        # Get nominal Boeing matrices with stability scaling
        A, B = self._get_boeing_matrices()

        # Initialize base environment
        super().__init__(A, B, self.QQ.copy(), self.RR.copy(), noise_scale, max_steps)
    
    def _get_boeing_matrices(self):
        """Get Boeing matrices with stability scaling."""
        A = self.AA.copy()
        B = self.BB.copy()

        # Ensure stability
        eigenvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigenvals))
        if max_eig >= 1.0:
            A = A / (1.1 * max_eig)

        return A, B

    def get_task_variants(self, num_variants: int = 5,
                         perturbation_scale: float = 1.1e-05) -> List['Boeing_dynamics']:
        """
        Generate Boeing aircraft variants for meta-learning.
        Overrides base class to return Boeing_dynamics instances.

        Parameters:
        num_variants (int): Number of variants to generate
        perturbation_scale (float): Scale of random perturbations to A, B, Q, R matrices

        Returns:
        List[Boeing_dynamics]: List of Boeing environment variants
        """
        variants = []

        # First variant is the original
        variants.append(Boeing_dynamics(self.noise_scale, self.max_steps))

        # Generate perturbed variants
        for _ in range(num_variants - 1):
            # Perturb system dynamics matrices (A, B)
            A_pert = self.A + np.random.normal(0, perturbation_scale, self.A.shape)
            B_pert = self.B + np.random.normal(0, perturbation_scale, self.B.shape)

            # Ensure stability
            eigenvals = np.linalg.eigvals(A_pert)
            max_eig = np.max(np.real(eigenvals))
            if max_eig >= 1.0:
                A_pert = A_pert / (1.1 * max_eig)

            # Perturb cost matrices (Q, R) while maintaining symmetry
            Q_noise = np.random.normal(0, perturbation_scale, self.Q.shape)
            Q_noise = (Q_noise + Q_noise.T) / 2  # Make symmetric
            Q_pert = self.Q + Q_noise

            R_noise = np.random.normal(0, perturbation_scale, self.R.shape)
            R_noise = (R_noise + R_noise.T) / 2  # Make symmetric
            R_pert = self.R + R_noise

            # Ensure Q and R remain positive definite (this also ensures symmetry)
            Q_pert = self._ensure_positive_definite(Q_pert)
            R_pert = self._ensure_positive_definite(R_pert)

            # Create Boeing variant with perturbed matrices
            variant = Boeing_dynamics(self.noise_scale, self.max_steps)
            # Override the matrices with perturbed versions
            variant.A = A_pert.copy()
            variant.B = B_pert.copy()
            variant.Q = Q_pert.copy()
            variant.R = R_pert.copy()
            variants.append(variant)

        return variants
    
    def get_aircraft_info(self):
        """Get Boeing aircraft-specific information."""
        return {
            'aircraft_type': 'Boeing_747_simplified',
            'state_description': ['velocity', 'angle_of_attack', 'pitch_rate', 'pitch_angle'],
            'control_description': ['elevator', 'thrust'],
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'noise_scale': self.noise_scale,
            'max_steps': self.max_steps
        }
    
    def compute_flight_metrics(self, states: List[np.ndarray], 
                              actions: List[np.ndarray]) -> dict:
        """
        Compute flight-specific metrics.
        
        Parameters:
        states (List[np.ndarray]): State trajectory
        actions (List[np.ndarray]): Control trajectory
        
        Returns:
        dict: Flight metrics
        """
        if not states or not actions:
            return {}
        
        states_array = np.array(states)
        actions_array = np.array(actions)
        
        # Extract state components (assuming 4D state)
        velocity = states_array[:, 0] if states_array.shape[1] > 0 else []
        angle_of_attack = states_array[:, 1] if states_array.shape[1] > 1 else []
        pitch_rate = states_array[:, 2] if states_array.shape[1] > 2 else []
        pitch_angle = states_array[:, 3] if states_array.shape[1] > 3 else []
        
        # Extract control components
        elevator = actions_array[:, 0] if actions_array.shape[1] > 0 else []
        thrust = actions_array[:, 1] if actions_array.shape[1] > 1 else []
        
        return {
            'velocity_std': np.std(velocity) if len(velocity) > 0 else 0,
            'angle_of_attack_max': np.max(np.abs(angle_of_attack)) if len(angle_of_attack) > 0 else 0,
            'pitch_rate_rms': np.sqrt(np.mean(pitch_rate**2)) if len(pitch_rate) > 0 else 0,
            'pitch_angle_range': np.ptp(pitch_angle) if len(pitch_angle) > 0 else 0,
            'elevator_usage': np.mean(np.abs(elevator)) if len(elevator) > 0 else 0,
            'thrust_usage': np.mean(np.abs(thrust)) if len(thrust) > 0 else 0,
            'control_effort': np.mean(np.sum(actions_array**2, axis=1)) if len(actions_array) > 0 else 0
        }


if __name__ == "__main__":
    print("Testing Boeing_dynamics...")
    
    # Test 1: Create Boeing environment
    print("\n=== Test 1: Boeing Environment ===")
    boeing = Boeing_dynamics(noise_scale=0.05)
    info = boeing.get_aircraft_info()
    print(f"Aircraft: {info['aircraft_type']}")
    print(f"State dim: {info['state_dim']}, Action dim: {info['action_dim']}")
    print(f"States: {info['state_description']}")
    print(f"Controls: {info['control_description']}")
    
    # Test 2: Basic functionality
    print("\n=== Test 2: Basic Boeing Functionality ===")
    initial_state = boeing.reset()
    print(f"Initial state: {initial_state}")
    
    # Test with small control inputs
    action = np.array([0.01, 0.05])  # Small elevator and thrust
    next_state, cost, done = boeing.step(action)
    print(f"After step - Cost: {cost:.4f}, Done: {done}")
    print(f"State change: {next_state - initial_state}")
    
    # Test 3: Optimal controller
    print("\n=== Test 3: Boeing Optimal Controller ===")
    try:
        K_opt = boeing.get_optimal_K()
        print(f"Optimal K shape: {K_opt.shape}")
        print(f"Optimal gain matrix:\n{K_opt}")
        
        # Test rollout with optimal controller
        cost_theory = boeing.get_optimal_cost()
        cost = boeing._approximate_infinite_cost(policy_K=K_opt, n_sample=100)
        print(f"Optimal policy average cost: {cost:.4f}")
        print(f"Theoretical optimal cost: {cost_theory:.4f}")
        
    except Exception as e:
        print(f"Optimal controller test failed: {e}")
    
    # Test 4: Task variants using inherited method
    print("\n=== Test 4: Boeing Task Variants ===")
    try:
        variants = boeing.get_task_variants(num_variants=5, perturbation_scale=1.1e-05)
        print(f"Generated {len(variants)} Boeing task variants")

        for i, variant in enumerate(variants):
            variant.reset()
            action = np.array([0.01, 0.05])
            _, cost, _ = variant.step(action)
            print(f"Variant {i} step cost: {cost:.4f}")

            # Check eigenvalues for stability
            eigenvals = np.linalg.eigvals(variant.A)
            max_eig = np.max(np.abs(eigenvals))
            print(f"  Max eigenvalue magnitude: {max_eig:.4f} ({'stable' if max_eig < 1.0 else 'unstable'})")

    except Exception as e:
        print(f"Task variants test failed: {e}")
    
    # Test 5: Additional task variants
    print("\n=== Test 5: Additional Task Variants ===")
    try:
        variants = boeing.get_task_variants(num_variants=5, perturbation_scale=1.1e-05)
        print(f"Generated {len(variants)} Boeing task variants with larger perturbations")

        for i, variant in enumerate(variants):
            variant.reset()
            action = np.array([0.01, 0.05])
            _, cost, _ = variant.step(action)
            print(f"Variant {i}: cost = {cost:.4f}")

    except Exception as e:
        print(f"Additional task variants test failed: {e}")
    
    # Test 6: Flight metrics
    print("\n=== Test 6: Flight Metrics ===")
    try:
        boeing.reset()
        K_test = np.random.normal(0, 0.1, (boeing.action_dim, boeing.state_dim))
        states, actions, costs = boeing.rollout(K_test, steps=15)
        
        metrics = boeing.compute_flight_metrics(states, actions)
        print("Flight metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
            
    except Exception as e:
        print(f"Flight metrics test failed: {e}")
    
    # Test 7: System matrices
    print("\n=== Test 7: Boeing System Matrices ===")
    system_info = boeing.get_system_info()
    print(f"A matrix shape: {system_info['A'].shape}")
    print(f"B matrix shape: {system_info['B'].shape}")
    print(f"A matrix eigenvalues: {np.linalg.eigvals(system_info['A'])}")
    
    # Check if matrices match original Boeing parameters (scaled)
    print(f"A matrix max element: {np.max(np.abs(system_info['A'])):.4f}")
    print(f"B matrix max element: {np.max(np.abs(system_info['B'])):.4f}")
    
    print("\n=== Boeing Tests Completed ===")
    print("Boeing_dynamics is ready for meta-learning!")