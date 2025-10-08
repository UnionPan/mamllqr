import scipy.stats as st
import numpy as np
from legacy.dynamics import Dynamics
import argparse
from pathlib import Path
import os
from tensorboardX import SummaryWriter
import pickle
np.random.seed(128)


def multi_systems(state_dim, action_dim, system_n):
    # first generate some random systems with random (A, B, Q, R, Psi)
    random_systems = [Dynamics(state_dim, action_dim)
                              for tmp in range(system_n)]
    # fix the first system as a templete
    tmp = random_systems[0]
    tmp_A, tmp_B, tmp_Q, tmp_R, tmp_Psi = tmp.A, tmp.B, tmp.Q, tmp.R, tmp.Psi
    # modify the rest of the systems using these templete matrices
    # so that the systems are very close 
    for sstm in random_systems[1:]:
        # generate random system matrices around the templete
        sstm.A = np.random.normal(tmp_A, scale=0.01)
        sstm.B = np.random.normal(tmp_B, scale=0.01)
        sstm.Q = tmp_Q
        sstm.R = tmp_R
        sstm.Psi = tmp_Psi

        A_eig_max = max(
            [i.real for i in np.linalg.eigvals(sstm.A) if i.imag == 0])
        # scale the matrix A such that the system can be more stable
        if A_eig_max >= 1:
            # sstm.A = sstm.A - (A_eig_max - 0.5) * np.diag([1, ] * sstm.state_dim)
            sstm.A = sstm.A / (1.05*A_eig_max)
         

        sstm.eta = st.multivariate_normal(cov=sstm.Psi)
        
        # make sure they are valid:
        

    return random_systems

    

def main(args):
    model_dir = Path('./maml-logs') / "{}-{}".format(args.state_dim,
                                                     args.action_dim) / "natural is {} smoothing parameter {} lr {} and {}".format(
        args.natural, args.r,
        args.eta, args.alpha)
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    os.makedirs(str(run_dir))
    logger = SummaryWriter((str(run_dir)))

    '''Generate a set of systems'''
    
    envs_ind = np.arange(args.system_n)
    envs_list = multi_systems(args.state_dim,args.action_dim,args.system_n)
    optimal_Ks = [env.cal_optimal_K() for env in envs_list]
    print(optimal_Ks)
    optimal_cost = [env.cal_optimal_J() for env in envs_list]
    print(optimal_cost)

    K = np.zeros((args.action_dim, args.state_dim))

    dk = args.state_dim * args.action_dim

    result_list = []
    for epoch in range(args.epoch):
        meta_gradient = np.zeros(K.shape)
        envs_ind_batch = np.random.choice(envs_ind, int(args.system_n / 2))
        meta_gradient2 = np.zeros(K.shape)
        for env_ind in envs_ind_batch:
            J_iU_d = 0
            for ind_d in range(args.d):

                # generate a perturbed policy K
                U_d = np.random.normal(
                    0, 1, K.shape[0]*K.shape[1]).reshape(*K.shape)
                U_d = (args.r / np.linalg.norm(U_d)) * U_d
                per_K = K + U_d

                # generate perturbed policy gradient
                per_grad = np.zeros(K.shape)
                per_sigma_grad = np.zeros((args.state_dim, args.state_dim))

                for ind_m in range(args.m):
                    # continue giving random perturbation to the perturbed policy
                    #print('epoch {} \n system {}\n perturb {} \n inner perturb {}'.format(epoch, env_ind, ind_d, ind_m))
                    U_i = np.random.normal(
                        0, 1, K.shape[0]*K.shape[1]).reshape(*K.shape)
                    U_i = (args.r / np.linalg.norm(U_i)) * U_i
                    state = envs_list[env_ind].reset()

                    #roll out the double perturbed policy
                    cost_list, state_list = envs_list[env_ind].rollout(
                        per_K + U_i, state, args.l)
                    C_i = np.mean(cost_list)
                    sigma_i = np.mean([np.dot(i, i.T) for i in state_list])

                    per_grad += C_i * U_i
                    per_sigma_grad += sigma_i

                per_grad *= (dk / (args.m * args.r**2))
                per_sigma_grad *= 1 / args.m
                if not args.natural:
                    grad = per_grad
                else:
                    grad = np.dot(per_grad, np.linalg.inv(per_sigma_grad))

                if np.linalg.norm(grad) >= 10:
                    grad *= (10 / np.linalg.norm(grad))

                per_K = per_K - args.eta * grad

                state = envs_list[env_ind].reset()
                cost_list_d, _ = envs_list[env_ind].rollout(
                    per_K, state, args.l)
                J_iU_d += np.mean(cost_list_d) * U_d

            J_iU_d *= (dk / args.d * args.r**2)
            meta_gradient += J_iU_d
            meta_gradient2 += (1 / args.d) * grad
        meta_gradient = meta_gradient * 2 / args.system_n
        meta_gradient2 *= 2 / args.system_n
        K = K - args.alpha * meta_gradient2

        if epoch % 5 == 0:
            cost_stat = []
            av_optimal_cost = np.mean(optimal_cost)

            for env in envs_list:
                for m in range(10):
                    state = env.reset()
                    cost_list, _ = env.rollout(K, state, 50, True)
                    cost_stat.append(np.mean(cost_list))

            cost_list = np.mean(cost_stat)
            ratio = (cost_list - av_optimal_cost) / av_optimal_cost

            print(
                "epoch is {} \n the cost difference ratio is {}\n the learned K is {} \n the policy difference is {}".format(
                    epoch, ratio, K, np.mean([K - optimal_Ks[i] for i in range(args.system_n)])
                )
            )
            logger.add_scalar("cost difference", ratio, epoch)
            logger.add_scalar("policy difference", np.mean(
                [K - optimal_Ks[i] for i in range(args.system_n)]), epoch)

            result_list.append(ratio)
        
        if epoch % 100 == 0:
            pickle.dump(result_list, open(run_dir / "summary.pkl", mode="wb"))


if __name__ == '__main__':
    #envs_list = multi_systems(3,3,5)
    #print(envs_list)
    #for i in range(5):
    #    print(envs_list[i].A, envs_list[i].B)
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_n", default=3, type=int)
    parser.add_argument("--state_dim", default=1, type=int)
    parser.add_argument("--action_dim", default=1, type=int)
    parser.add_argument("--l", default=30, type=int, help="roll-out length")
    parser.add_argument("--m", default=100, type=int,
                        help="number of trajectories")
    parser.add_argument("--r", default=0.05, type=float,
                        help="smoothing parameter")
    parser.add_argument("--d", default=3, type=int,
                        help="number of perturbation for each system")
    parser.add_argument("--epoch", default=1000, type=int,
                        help="number of training epochs")
    parser.add_argument("--eta", default=1e-3, type=float,
                        help="inner learning rate")
    parser.add_argument("--alpha", default=1e-3, type=float,
                        help="outer learning rate")
    parser.add_argument("--natural", default=False, action="store_true")
    args = parser.parse_args()
    main(args=args)
