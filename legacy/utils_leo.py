import numpy as np
import scipy.linalg
from control import dlqr
import matplotlib.pyplot as plt
import control
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt

def generate_perturbed_samples(A, N, epsilon, symmetric = None):
    """
    Generates N samples that are perturbed from matrix A,
    such that the maximum spectral norm difference between all pairs
    is less than epsilon.

    Parameters:
    A (np.ndarray): The original matrix.
    N (int): The number of samples to generate.
    epsilon (float): The maximum allowed spectral norm difference.
    flag (boolean): Check Q is symmetric.

    Returns:
    List[np.ndarray]: A list of perturbed samples.
    """
    # Compute the spectral norm of the original matrix
    norm_A = np.linalg.norm(A, ord=2)

    # Initialize a list to hold the perturbed samples
    perturbed_samples =[]
    perturbed_samples.append(A)

    # Generate N-1 perturbed samples
    for _ in range(N-1):
        while True:
            # Generate a random perturbation matrix
            P = np.random.randn(*A.shape)

            # If symmetric flag is set, make P a symmetric matrix
            if symmetric:
                P = (P + P.T) / 2

            # Scale the perturbation matrix to have a spectral norm of epsilon
            norm_P = np.linalg.norm(P, ord=2)
            P_scaled = (P / norm_P) * epsilon  * 0.6#*  (2/3 + 1/3 *(np.random.rand()  # Random scaling factor less than epsilon

            # Add the perturbation to the original matrix
            A_perturbed = A + P_scaled

            # Ensure that the new sample's spectral norm difference is less than epsilon
            if all(np.linalg.norm(A_perturbed - other_A, ord=2) < epsilon for other_A in perturbed_samples):
                perturbed_samples.append(A_perturbed)
                break  # The perturbed sample is valid, break the loop

    # perturbed_samples = perturbed_samples[1:]

    return perturbed_samples


def grad_zo(A, B, Q, R, K, U, r, x_0, N):
    nx = A.shape[0]
    nu = B.shape[1]
    K1 = []
    K2 = []
    cost_emp1 = []
    cost_emp2 = []
    grad = np.zeros((nu, nx))
    U1 = []

    for i in range(U.shape[0]):
        # Sample policies
        # Check stability
        #if check_stable(A, B, K + U[i]) == True and check_stable(A, B, K - U[i]) == True:
        K1.append(K + U[i])
        K2.append(K - U[i])
        U1.append(U[i])

        # Compute empirical cost
        x_0 = np.random.normal(0.5, 1e-3, size=(1,nx))[0]
        cost_emp1.append(generate_task.lqr_cost(A, B, Q, R, K1[-1], x_0, N))
        cost_emp2.append(generate_task.lqr_cost(A, B, Q, R, K2[-1], x_0, N))

    ns = len(cost_emp1)
    for i in range(ns):
        grad += ((nx * nu) / (2 * ns * (r ** 2))) * (cost_emp1[i] - cost_emp2[i]) * U1[i]

    return grad


def hessian_zo(A, B, Q, R, K, U, r, x_0, N):
    nx = A.shape[0]
    nu = B.shape[1]
    K1 = []
    K2 = []
    cost_emp1 = []
    cost_emp2 = []
    hess = np.zeros((nu, nu))
    U1 = []

    for i in range(U.shape[0]):
        # Sample policies
        # Check stability
        #if check_stable(A, B, K + U[i]) == True and check_stable(A, B, K - U[i]) == True:
        K1.append(K + U[i])
        K2.append(K - U[i])
        U1.append(U[i])

        x_0 = np.random.normal(0.5, 1e-3, size=(1,nx))[0]
        # Compute empirical cost
        cost_emp1.append(generate_task.lqr_cost(A, B, Q, R, K1[-1], x_0, N))
        cost_emp2.append(generate_task.lqr_cost(A, B, Q, R, K, x_0, N))

    ns = len(cost_emp1)
    for i in range(ns):
        U_i = U1[i] @ U1[i].T
        #hess += ((nu * nu) / (2 * ns * (r ** 2))) * (cost_emp1[i] - cost_emp2[i]) * (U_i - np.eye(U_i.shape[0]))
        hess += ((nu * nu) / (ns * (r ** 2))) * (cost_emp1[i] - cost_emp2[i]) * (U_i - np.eye(U_i.shape[0]))

    return hess


class generate_task:
    def __init__(self, A, B, Q, R, M = None, flag = None, param = None, N = None):
        """
        Initialize the Discrete Time State Space system with LQR cost function.

        Parameters:
        A (np.ndarray): State transition matrix.
        B (np.ndarray): Control matrix.
        Q (np.ndarray): State cost matrix.
        R (np.ndarray): Control cost matrix.
        M (int): Number of tasks.
        flag (str): 's': system heterogeneity, 'c': cost heterogeneity, 'sc': system & cost heterogeneity.
        param (list): [ep1, ep2], [ep3,ep4], [ep1,ep2,ep3,ep4]. Heterogeneity
        N (int): Horizon.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.M = M

        if flag == 's':
            self.A_M = generate_perturbed_samples(A, M, param[0])
            self.B_M = generate_perturbed_samples(B, M, param[1])
            self.Q_M = [Q for _ in self.A_M]
            self.R_M = [R for _ in self.A_M]

        elif flag == 'c':
            self.Q_M = generate_perturbed_samples(Q, M, param[0], 1)
            self.R_M = generate_perturbed_samples(R, M, param[1], 1)
            self.A_M = [A for _ in self.Q_M]
            self.B_M = [B for _ in self.Q_M]

        else: # s & c
            self.A_M = generate_perturbed_samples(A, M, param[0])
            self.B_M = generate_perturbed_samples(B, M, param[1])
            self.Q_M = generate_perturbed_samples(Q, M, param[0], 1)
            self.R_M = generate_perturbed_samples(R, M, param[1], 1)

    def K_star(self):
        K = []
        for i, item in enumerate(self.A_M):
            k, _, _ = dlqr(item, self.B_M[i], self.Q_M[i], self.R_M[i])
            K.append(np.array(k))
        self.K_star = K

    @staticmethod
    def control_law(K, x_t):
        if K is None:
            raise ValueError("LQR solution not computed. Call K_star() first.")
        return -K @ x_t

    @staticmethod
    def simulate(A, B, N, x0):
        x_t = x0.copy()
        states = [x_t]
        # controls = []
        for i in range(N):
            u_t = generate_task.control_law(K, x_t)
            # controls.append(u_t)
            x_t = A @ x_t + B @ u_t
            states.append(x_t)
        return states   #, controls

    @staticmethod
    def lqr_cost(A, B, Q, R, K, x_0, N):
        """
        Computes the LQR cost for a given sequence of states up to a horizon N,
        using the state cost matrix Q, the control cost matrix R, and the
        feedback gain matrix K.

        Parameters:
        Q (np.ndarray): The state cost matrix.
        R (np.ndarray): The control cost matrix.
        K (np.ndarray): The feedback gain matrix.
        x_list (List[np.ndarray]): The list of state vectors.
        N (int): The horizon up to which to compute the cost.

        Returns:
        float: The computed LQR cost.
        """
    
        pk = Q.copy()
        for t in range(N + 1):
            pk = Q + K.T @ R @ K + (A - B @ K).T @ pk @ (A - B @ K)
        cost = x_0 @ pk @ x_0
        return cost

    def init_k(self, perturb, step):
        k, _, _ = dlqr(self.A, self.B, self.Q, self.R)
        k = np.array(k)
        noise = np.random.normal(0, 1, k.shape)

        # Normalize the noise vector so its norm is 1
        noise = (noise / np.linalg.norm(noise)) * perturb
        k_p = k + noise
        flag_k = check_stable(self.A ,self.B, k)
        while flag_k == False:
            flag_k = check_stable(self.A ,self.B, k + np.random.normal(0, perturb, k.shape))
        flag_task = False
      
        while flag_task == False:
            flag = []
            for i in range(len(self.A_M)):
                ki = k_p - step * grad_true(self.A_M[i], self.B_M[i], self.Q_M[i], self.R_M[i], k_p)
                flag.append(check_stable(self.A_M[i], self.B_M[i], ki))

            flag_task = all(flag)

        return k_p, k

    def C_star(self, x_0, N): # N - horizon
        C = []
        for i, item in enumerate(self.A_M):
            cost = generate_task.lqr_cost(item, self.B_M[i], self.Q_M[i], self.R_M[i], self.K_star[i], x_0, N)
            C.append(cost)
        return C

    def MAML(self, K, ita1, ita2):
        sum1 = np.zeros((K.shape))
        for i in range(len(self.A_M)):
            K_1 = K - ita2 * grad_true(self.A_M[i], self.B_M[i], self.Q_M[i], self.R_M[i], K)
            grad = grad_true(self.A_M[i], self.B_M[i], self.Q_M[i], self.R_M[i], K_1)
            hess = hessian_true(self.A_M[i], self.B_M[i], self.Q_M[i], self.R_M[i], K)
            hess = np.eye(hess.shape[0]) - ita2 * hess
            sum1 += hess@grad
        step = - ita1 * (1/self.M) * sum1
        K_MAML = K + step
        return K_MAML

    def MAML_ZO(self, K, ita1, ita2, r, sample, x_0, N):
        sum1 = np.zeros((K.shape))
        U = np.random.randn(sample, self.B.shape[1], self.A.shape[0])
        U_norms = np.linalg.norm(U, axis=(1, 2), keepdims=True)
        U = (U / U_norms)* r
    
        for i in range(len(self.A_M)):
            K_1 = K - ita2 * grad_zo(self.A_M[i], self.B_M[i], self.Q_M[i], self.R_M[i], K, U, r, x_0, N)
            grad = grad_zo(self.A_M[i], self.B_M[i], self.Q_M[i], self.R_M[i], K_1, U, r, x_0, N)
            hess = hessian_zo(self.A_M[i], self.B_M[i], self.Q_M[i], self.R_M[i], K, U, r, x_0, N)
            hess = np.eye(hess.shape[0]) - ita2 * hess
            sum1 += hess @ grad
        step = - ita1 * (1/self.M) * sum1
        K_MAML = K + step
        return K_MAML



    def C_MAML(self, K, x_0, N): # N - horizon
        C = []
        for i, item in enumerate(self.A_M):
            cost = generate_task.lqr_cost(item, self.B_M[i], self.Q_M[i], self.R_M[i], K, x_0, N)
            C.append(cost)
        return C