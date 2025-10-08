import numpy as np
import scipy.stats as st
from typing import List, Dict, Tuple, Optional
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov



class BaseLQREnv:
    """
    Minimal base environment for LQR control tasks.
    Simple, clean, no external dependencies beyond numpy/scipy.
    """
    
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, 
                 noise_scale: float = 0.05, max_steps: int = 100):
        """
        Initialize LQR environment with system matrices.
        
        Parameters:
        A (np.ndarray): State transition matrix [state_dim, state_dim]
        B (np.ndarray): Control matrix [state_dim, action_dim]
        Q (np.ndarray): State cost matrix [state_dim, state_dim]
        R (np.ndarray): Control cost matrix [action_dim, action_dim]
        noise_scale (float): Process noise scale
        max_steps (int): Maximum episode steps
        """
        # System matrices
        self.A = A.copy()
        self.B = B.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        
        # Environment parameters
        self.noise_scale = noise_scale
        self.max_steps = max_steps
        
        # Dimensions
        self.state_dim = A.shape[0]
        self.action_dim = B.shape[1]
        
        # Process noise
        self.Psi = noise_scale * np.eye(self.state_dim)
        self.noise_gen = st.multivariate_normal(cov=self.Psi)
        
        # Environment state
        self.state = None
        self.step_count = 0
        self.done = False
        
        # Validation
        self._validate_matrices()
    
    def _validate_matrices(self):
        """Basic validation of system matrices."""
        assert self.A.shape[0] == self.A.shape[1], "A must be square"
        assert self.A.shape[0] == self.B.shape[0], "A and B must have compatible dimensions"
        assert self.Q.shape == self.A.shape, "Q must have same shape as A"
        assert self.R.shape[0] == self.R.shape[1] == self.action_dim, "R must be square with action_dim"
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment.
        
        Parameters:
        initial_state (np.ndarray, optional): Specific initial state
        
        Returns:
        np.ndarray: Initial state observation
        """
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            # Random initial state using covariance matrix
            self.state = np.random.multivariate_normal(
                mean=np.zeros(self.state_dim),
                cov=self.Psi
            )
        
        self.step_count = 0
        self.done = False
        
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Step the environment.
        
        Parameters:
        action (np.ndarray): Control action
        
        Returns:
        Tuple: (next_state, cost, done)
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")
        
        # Compute cost
        cost = self.compute_cost(self.state, action)
        
        # Update state: x_{t+1} = Ax_t + Bu_t + noise
        noise = self.noise_gen.rvs().reshape(-1, 1) if self.state.ndim == 2 else self.noise_gen.rvs()
        self.state = self.A @ self.state + self.B @ action + noise
        
        # Update counters
        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        
        return self.state.copy(), cost, self.done
    
    def compute_cost(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Compute LQR cost: x^T Q x + u^T R u
        
        Parameters:
        state (np.ndarray): Current state
        action (np.ndarray): Control action
        
        Returns:
        float: LQR cost
        """
        state_cost = state.T @ self.Q @ state
        action_cost = action.T @ self.R @ action
        
        # Handle scalar returns
        total_cost = state_cost + action_cost
        return float(total_cost) if np.isscalar(total_cost) else total_cost.item()
    
    def get_optimal_K(self) -> np.ndarray:
        """
        Compute optimal LQR control gain using discrete algebraic Riccati equation.
        
        Returns:
        np.ndarray: Optimal control gain K
        """
        P = self.solve_riccati_dare()
        K_opt = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        return K_opt
    
    def solve_riccati_dare(self) -> np.ndarray:
        """
        Solve discrete algebraic Riccati equation (DARE) using scipy.
        
        Returns:
        np.ndarray: Solution matrix P
        """
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        return P
        
    def solve_riccati_lyapunov(self, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
        """
        Solve discrete Riccati equation using iterative Lyapunov approach.
        
        The discrete Riccati equation is:
        P = Q + A^T P A - A^T P B (R + B^T P B)^(-1) B^T P A
        
        Parameters:
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance
        
        Returns:
        np.ndarray: Solution matrix P
        """
        P = self.Q.copy()
        
        for i in range(max_iter):
            # Compute the optimal gain for current P
            try:
                # Solve: K = (R + B^T P B)^(-1) B^T P A
                temp_matrix = self.R + self.B.T @ P @ self.B
                K = np.linalg.solve(temp_matrix, self.B.T @ P @ self.A)
                
                # Update P using closed-loop system: P = Q + (A - BK)^T P (A - BK) + K^T R K
                A_cl = self.A - self.B @ K  # Closed-loop A matrix
                P_new = self.Q + K.T @ self.R @ K + A_cl.T @ P @ A_cl
                
            except np.linalg.LinAlgError:
                # If matrix is singular, try alternative formulation
                P_new = self.Q + self.A.T @ P @ self.A - \
                       self.A.T @ P @ self.B @ np.linalg.pinv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
            
            # Check convergence
            error = np.linalg.norm(P_new - P)
            if error < tol:
                if i < 10:
                    print(f"Lyapunov converged in {i+1} iterations (error: {error:.2e})")
                break
            
            P = P_new
        else:
            print(f"Warning: Lyapunov iteration did not converge after {max_iter} iterations (error: {error:.2e})")
        
        return P
    
    def get_optimal_cost(self) -> float:
        """
        Compute optimal infinite-horizon cost.
        
        Returns:
        float: Optimal cost
        """
        try:
            P = solve_discrete_are(self.A, self.B, self.Q, self.R)
            return np.trace(P @ self.Psi)
        except Exception as e:
            # Approximate with large finite horizon
            return self._approximate_infinite_cost()

    def analytical_cost(self, K: np.ndarray, fallback_to_simulation: bool = True,
                       stability_margin: float = 0.99) -> float:
        """
        Compute analytical infinite-horizon cost for policy u = -Kx.

        Uses the exact LQR formula: J = trace(P_K @ Psi)
        where P_K satisfies the discrete Lyapunov equation:
        P_K = Q_cl + A_cl^T @ P_K @ A_cl
        with Q_cl = Q + K^T @ R @ K and A_cl = A - B @ K

        Parameters:
        K (np.ndarray): Control gain matrix [action_dim, state_dim]
        fallback_to_simulation (bool): Use Monte Carlo if analytical fails
        stability_margin (float): Eigenvalue threshold for stability check

        Returns:
        float: Infinite-horizon cost
        """
        try:
            # Closed-loop system matrix
            A_cl = self.A - self.B @ K

            # Check closed-loop stability
            eigenvals = np.linalg.eigvals(A_cl)
            max_eigenval = np.max(np.abs(eigenvals))

            if max_eigenval >= stability_margin:
                # Unstable system - infinite cost
                if fallback_to_simulation:
                    # Try a short simulation to get finite cost estimate
                    try:
                        return self._approximate_infinite_cost(policy_K=K, n_sample=30)
                    except:
                        return float('inf')
                else:
                    return float('inf')

            # Closed-loop cost matrices
            Q_cl = self.Q + K.T @ self.R @ K

            # Solve discrete Lyapunov equation: P = Q_cl + A_cl^T @ P @ A_cl
            
            P_K = solve_discrete_lyapunov(A_cl.T, Q_cl)

            # Check for numerical issues
            if not np.all(np.isfinite(P_K)) or not np.all(np.real(np.linalg.eigvals(P_K)) >= 0):
                if fallback_to_simulation:
                    return self._approximate_infinite_cost(policy_K=K, n_sample=30)
                else:
                    return float('inf')

            # Analytical infinite-horizon cost
            cost = np.trace(P_K @ self.Psi)

            # Sanity check - cost should be positive
            if cost < 0 or not np.isfinite(cost):
                if fallback_to_simulation:
                    return self._approximate_infinite_cost(policy_K=K, n_sample=30)
                else:
                    return float('inf')

            return float(cost)

        except Exception as e:
            # Fallback to simulation if analytical computation fails
            if fallback_to_simulation:
                try:
                    return self._approximate_infinite_cost(policy_K=K, n_sample=30)
                except:
                    return float('inf')
            else:
                return float('inf')

    def analytical_covariance(self, K: np.ndarray, stability_margin: float = 0.99) -> np.ndarray:
        """
        Compute analytical steady-state covariance matrix for policy u = -Kx.

        For the closed-loop system x_{t+1} = A_cl x_t + w_t where A_cl = A - BK,
        the steady-state covariance satisfies: Σ = A_cl Σ A_cl^T + Ψ

        Parameters:
        K (np.ndarray): Feedback gain matrix (action_dim x state_dim)
        stability_margin (float): Eigenvalue threshold for stability

        Returns:
        np.ndarray: Steady-state covariance matrix (state_dim x state_dim)
        """
        try:
            A_cl = self.A - self.B @ K

            # Check stability
            eigenvals = np.linalg.eigvals(A_cl)
            max_eigenval = np.max(np.abs(eigenvals))

            if max_eigenval >= stability_margin:
                # Return identity for unstable systems
                return np.eye(self.state_dim)

            # Solve discrete Lyapunov equation: Σ = A_cl Σ A_cl^T + Ψ
            from scipy.linalg import solve_discrete_lyapunov
            Sigma = solve_discrete_lyapunov(A_cl, self.Psi)

            # Ensure positive definite
            if np.any(np.linalg.eigvals(Sigma) <= 0):
                Sigma += 1e-6 * np.eye(self.state_dim)

            return Sigma

        except Exception as e:
            # Fallback to identity matrix
            return np.eye(self.state_dim)

    def _approximate_infinite_cost(self, policy_K: np.ndarray, n_sample: int = 100) -> float:
        """Approximate infinite cost with large finite horizon."""

        total_cost = 0
        successful_samples = 0

        for i in range(n_sample):
            try:
                states, actions, costs = self.rollout(policy_K, steps=self.max_steps)
                if costs:  # Check if we got valid costs
                    avg_cost = np.mean(costs)
                    total_cost += avg_cost
                    successful_samples += 1

            except Exception as e:
                print(f"  Sample {i+1} failed: {e} when evaluating the policy {policy_K}")
                continue

        if successful_samples > 0:
            final_avg = total_cost / successful_samples
            return final_avg
        else:
            print("  All samples failed!")
            return float('inf')
    
    def rollout(self, policy_K: np.ndarray, initial_state: Optional[np.ndarray] = None, 
                steps: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Rollout with linear policy u = -K*x
        
        Parameters:
        policy_K (np.ndarray): Control gain matrix
        initial_state (np.ndarray, optional): Starting state
        steps (int, optional): Number of steps (default: max_steps)
        
        Returns:
        Tuple: (states, actions, costs)
        """
        if steps is None:
            steps = self.max_steps
        
        # Reset environment
        state = self.reset(initial_state)
        
        states, actions, costs = [], [], []
        
        for _ in range(steps):
            if self.done:
                break
            
            # Linear policy: u = -K*x
            action = -policy_K @ state
            
            states.append(state.copy())
            actions.append(action.copy())
            
            # Step environment
            next_state, cost, done = self.step(action)
            costs.append(cost)
            
            state = next_state
        
        return states, actions, costs
    
    def get_task_variants(self, num_variants: int = 5,
                         perturbation_scale: float = 0.05) -> List['BaseLQREnv']:
        """
        Generate variants of this environment for meta-learning.

        Parameters:
        num_variants (int): Number of variants to generate
        perturbation_scale (float): Scale of random perturbations to A, B, Q, R matrices

        Returns:
        List[BaseLQREnv]: List of environment variants
        """
        variants = []

        # First variant is the original
        variants.append(BaseLQREnv(self.A, self.B, self.Q, self.R,
                                  self.noise_scale, self.max_steps))

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

            # Create variant
            variant = BaseLQREnv(A_pert, B_pert, Q_pert, R_pert,
                               self.noise_scale, self.max_steps)
            variants.append(variant)

        return variants

    def _ensure_positive_definite(self, matrix: np.ndarray, min_eigenval: float = 1e-6) -> np.ndarray:
        """
        Ensure a matrix is positive definite by regularizing eigenvalues.

        Parameters:
        matrix (np.ndarray): Input matrix
        min_eigenval (float): Minimum eigenvalue to enforce

        Returns:
        np.ndarray: Positive definite matrix
        """
        # Make symmetric
        matrix = (matrix + matrix.T) / 2

        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(matrix)

        # Regularize eigenvalues to be positive
        eigenvals = np.maximum(eigenvals, min_eigenval)

        # Reconstruct matrix
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def check_stability(self, verbose: bool = True) -> Dict:
        """
        Check system stability and controllability.
        
        Parameters:
        verbose (bool): Print detailed stability analysis
        
        Returns:
        Dict: Stability analysis results
        """
        results = {}
        
        # Check open-loop stability
        eigenvals_open = np.linalg.eigvals(self.A)
        max_eig_open = np.max(np.abs(eigenvals_open))
        results['open_loop_stable'] = max_eig_open < 1.0
        results['open_loop_eigenvalues'] = eigenvals_open
        results['max_eigenvalue_magnitude'] = max_eig_open
        
        # Check closed-loop stability with optimal controller
        try:
            K_opt = self.get_optimal_K()
            A_cl = self.A - self.B @ K_opt
            eigenvals_closed = np.linalg.eigvals(A_cl)
            max_eig_closed = np.max(np.abs(eigenvals_closed))
            results['closed_loop_stable'] = max_eig_closed < 1.0
            results['closed_loop_eigenvalues'] = eigenvals_closed
            results['max_eigenvalue_closed'] = max_eig_closed
            results['optimal_K'] = K_opt
        except Exception as e:
            results['closed_loop_error'] = str(e)
            results['closed_loop_stable'] = False
        
        # Check controllability
        try:
            # Construct controllability matrix
            n = self.state_dim
            C = np.zeros((n, n * self.action_dim))
            for i in range(n):
                C[:, i*self.action_dim:(i+1)*self.action_dim] = np.linalg.matrix_power(self.A, i) @ self.B
            
            rank_C = np.linalg.matrix_rank(C)
            results['controllable'] = rank_C == n
            results['controllability_rank'] = rank_C
            
        except Exception as e:
            results['controllability_error'] = str(e)
            results['controllable'] = False
        
        if verbose:
            print("=== System Stability Analysis ===")
            print(f"Open-loop stable: {results['open_loop_stable']} (max |λ| = {max_eig_open:.4f})")
            if 'closed_loop_stable' in results:
                print(f"Closed-loop stable: {results['closed_loop_stable']} (max |λ| = {results.get('max_eigenvalue_closed', 'N/A'):.4f})")
            print(f"Controllable: {results.get('controllable', 'Unknown')} (rank = {results.get('controllability_rank', 'N/A')}/{self.state_dim})")
            
            if results['open_loop_stable']:
                print("✓ System is open-loop stable")
            else:
                print("⚠ System is open-loop unstable")
                
            if results.get('closed_loop_stable', False):
                print("✓ System is stabilizable with LQR control")
            else:
                print("⚠ System may not be stabilizable")
        
        return results
    
    def compare_riccati_methods(self, verbose: bool = True) -> Dict:
        """
        Compare DARE vs Lyapunov methods for solving Riccati equation.
        
        Parameters:
        verbose (bool): Print detailed comparison
        
        Returns:
        Dict: Comparison results
        """
        results = {}
        
        try:
            # Method 1: DARE
            import time
            start_time = time.time()
            P_dare = self.solve_riccati_dare()
            time_dare = time.time() - start_time
            
            # Compute corresponding K
            K_dare = np.linalg.inv(self.R + self.B.T @ P_dare @ self.B) @ (self.B.T @ P_dare @ self.A)
            
            results['dare_success'] = True
            results['P_dare'] = P_dare
            results['K_dare'] = K_dare
            results['time_dare'] = time_dare
            
        except Exception as e:
            results['dare_success'] = False
            results['dare_error'] = str(e)
            if verbose:
                print(f"DARE method failed: {e}")
        
        try:
            # Method 2: Lyapunov iteration
            start_time = time.time()
            P_lyap = self.solve_riccati_lyapunov()
            time_lyap = time.time() - start_time
            
            # Compute corresponding K
            K_lyap = np.linalg.inv(self.R + self.B.T @ P_lyap @ self.B) @ (self.B.T @ P_lyap @ self.A)
            
            results['lyapunov_success'] = True
            results['P_lyapunov'] = P_lyap
            results['K_lyapunov'] = K_lyap
            results['time_lyapunov'] = time_lyap
            
        except Exception as e:
            results['lyapunov_success'] = False
            results['lyapunov_error'] = str(e)
            if verbose:
                print(f"Lyapunov method failed: {e}")
        
        # Compare results if both succeeded
        if results.get('dare_success') and results.get('lyapunov_success'):
            P_diff = np.linalg.norm(results['P_dare'] - results['P_lyapunov'])
            K_diff = np.linalg.norm(results['K_dare'] - results['K_lyapunov'])
            
            results['P_difference'] = P_diff
            results['K_difference'] = K_diff
            results['methods_agree'] = P_diff < 1e-6 and K_diff < 1e-6
            
            if verbose:
                print("\n=== Riccati Methods Comparison ===")
                print(f"DARE computation time: {results['time_dare']:.6f} seconds")
                print(f"Lyapunov computation time: {results['time_lyapunov']:.6f} seconds")
                print(f"P matrix difference: {P_diff:.2e}")
                print(f"K matrix difference: {K_diff:.2e}")
                
                if results['methods_agree']:
                    print("✓ Both methods agree (difference < 1e-6)")
                else:
                    print("⚠ Methods disagree - check system conditioning")
                
                # Check Riccati equation residual for both solutions
                for method, P in [('DARE', results['P_dare']), ('Lyapunov', results['P_lyapunov'])]:
                    residual = self.Q + self.A.T @ P @ self.A - P - \
                              self.A.T @ P @ self.B @ np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
                    residual_norm = np.linalg.norm(residual)
                    results[f'residual_{method.lower()}'] = residual_norm
                    print(f"{method} residual norm: {residual_norm:.2e}")
        
        return results
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'A': self.A.copy(),
            'B': self.B.copy(), 
            'Q': self.Q.copy(),
            'R': self.R.copy(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'noise_scale': self.noise_scale,
            'max_steps': self.max_steps
        }


if __name__ == "__main__":
    print("Testing BaseLQREnv functionality...")
    
    # Test 1: Create a simple 2D system
    print("\n=== Test 1: Basic Environment Creation ===")
    A = np.array([[1.1, 0.1], 
                  [0.0, 0.9]])
    B = np.array([[1.0], 
                  [0.5]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    
    env = BaseLQREnv(A, B, Q, R, noise_scale=0.05, max_steps=100)
    print(f"Created environment: {env.state_dim}D state, {env.action_dim}D action")
    print(f"A matrix:\n{env.A}")
    print(f"B matrix:\n{env.B}")
    
    # # Test 2: Reset and step
    # print("\n=== Test 2: Reset and Step ===")
    # initial_state = env.reset()
    # print(f"Initial state: {initial_state}")
    
    # action = np.array([0.1])
    # next_state, cost, done = env.step(action)
    # print(f"After step - State: {next_state}, Cost: {cost:.4f}, Done: {done}")
    
    # # Test 3: Multiple steps
    # print("\n=== Test 3: Multiple Steps ===")
    # env.reset()
    # total_cost = 0
    # for i in range(3):
    #     action = np.random.normal(0, 0.1, size=env.action_dim)
    #     state, cost, done = env.step(action)
    #     total_cost += cost
    #     print(f"Step {i+1}: Cost = {cost:.4f}, State = [{state[0]:.3f}, {state[1]:.3f}]")
    # print(f"Total cost over 3 steps: {total_cost:.4f}")
    
    # Test 4: Optimal controller
    print("\n=== Test 4: Optimal Controller ===")
    try:
        K_opt = env.get_optimal_K()
        print(f"Optimal gain K: {K_opt}")
        opt_cost = env.get_optimal_cost()
        print(f"Optimal infinite-horizon cost: {opt_cost:.4f}")
    except Exception as e:
        print(f"Optimal controller test failed: {e}")
    
    # Test 5: Rollout with optimal policy
    print("\n=== Test 5: Rollout Test ===")
    try:
        K_opt = env.get_optimal_K()
        print(f"testing optimal policy {K_opt}")
        n_sample1 = 200
        n_sample2 = 50
        cost1 = env._approximate_infinite_cost(policy_K=K_opt, n_sample=n_sample1)
        cost2 = env._approximate_infinite_cost(policy_K=K_opt, n_sample=n_sample2)
        print(f"Averaging Rollout with optimal policy:")
        print(f"  Average cost with {n_sample1} samples: {cost1:.4f}")
        print(f"  Average cost with {n_sample2} samples: {cost2:.4f}")
    except Exception as e:
        print(f"Rollout test failed: {e}")

    # Test 5b: Analytical cost vs simulation comparison
    print("\n=== Test 5b: Analytical vs Simulation Cost Comparison ===")
    try:
        K_opt = env.get_optimal_K()

        # Compare analytical vs simulation methods
        theoretical_cost = env.get_optimal_cost()
        analytical_cost = env.analytical_cost(K_opt)
        simulation_cost = env._approximate_infinite_cost(policy_K=K_opt, n_sample=100)

        print(f"Theoretical optimal cost:  {theoretical_cost:.6f}")
        print(f"Analytical cost (K_opt):   {analytical_cost:.6f}")
        print(f"Simulation cost (K_opt):   {simulation_cost:.6f}")
        print(f"Analytical vs theoretical: {abs(analytical_cost - theoretical_cost):.8f}")
        print(f"Simulation vs theoretical: {abs(simulation_cost - theoretical_cost):.8f}")

        # Test with a random policy
        K_random = np.random.normal(0, 0.1, K_opt.shape)
        analytical_random = env.analytical_cost(K_random)
        simulation_random = env._approximate_infinite_cost(policy_K=K_random, n_sample=50)

        print(f"\nRandom policy comparison:")
        print(f"Analytical cost (random):  {analytical_random:.6f}")
        print(f"Simulation cost (random):  {simulation_random:.6f}")
        print(f"Difference (ana vs sim):   {abs(analytical_random - simulation_random):.6f}")

        # Speed test
        import time

        # Time analytical method
        start = time.time()
        for _ in range(100):
            _ = env.analytical_cost(K_random)
        analytical_time = time.time() - start

        # Time simulation method
        start = time.time()
        for _ in range(100):
            _ = env._approximate_infinite_cost(policy_K=K_random, n_sample=10)
        simulation_time = time.time() - start

        print(f"\nSpeed comparison (100 evaluations):")
        print(f"Analytical method: {analytical_time:.4f} seconds")
        print(f"Simulation method: {simulation_time:.4f} seconds")
        print(f"Speedup factor:    {simulation_time/analytical_time:.1f}x")

    except Exception as e:
        print(f"Analytical cost test failed: {e}")
    
    # # Test 6: Random policy comparison
    # print("\n=== Test 6: Policy Comparison ===")
    # try:
    #     # Test optimal policy
    #     K_opt = env.get_optimal_K()
    #     _, _, costs_opt = env.rollout(K_opt, steps=10)
    #     avg_cost_opt = np.mean(costs_opt)
        
    #     # Test random policy
    #     K_random = np.random.normal(0, 0.1, K_opt.shape)
    #     _, _, costs_random = env.rollout(K_random, steps=10)
    #     avg_cost_random = np.mean(costs_random)
        
    #     print(f"Optimal policy avg cost: {avg_cost_opt:.4f}")
    #     print(f"Random policy avg cost: {avg_cost_random:.4f}")
    #     print(f"Cost ratio (random/optimal): {avg_cost_random/avg_cost_opt:.2f}")
        
    #     if avg_cost_random > avg_cost_opt:
    #         print("✓ Optimal policy performs better (as expected)")
    #     else:
    #         print("⚠ Unexpected: random policy performed better")
            
    # except Exception as e:
    #     print(f"Policy comparison test failed: {e}")
    
    # Test 7: Task variants for meta-learning
    print("\n=== Test 7: Meta-Learning Task Variants ===")
    try:
        variants = env.get_task_variants(num_variants=3, perturbation_scale=1.3e-6)
        print(f"Generated {len(variants)} task variants")
        
        for i, variant in enumerate(variants):
            # Compare A matrices
            A_diff = np.linalg.norm(variant.A - env.A)
            print(f"Variant {i}: A matrix difference = {A_diff:.4f}")
            
            # Test that each variant works
            variant.reset()
            _, cost, _ = variant.step(np.array([0.1]))
            print(f"  Step cost: {cost:.4f}")
            
    except Exception as e:
        print(f"Task variants test failed: {e}")
    
    