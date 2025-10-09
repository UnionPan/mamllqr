# Meta-Learning for Linear Quadratic Regulator Control

This repository implements and compares meta-learning algorithms for Linear Quadratic Regulator (LQR) control systems, specifically focusing on Boeing aircraft dynamics. 

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Test
```bash
./scripts/experiment.sh --quick
```

### Full Experiment
```bash
./scripts/experiment.sh --full --verbose
```

### Custom Configuration
```bash
./scripts/experiment.sh --num_rounds 5 --num_epochs 100 --target_epoch 30
```

## Algorithms

- **HessianFree**: Meta-learning with Hessian-free natural gradient estimation [paper here](https://arxiv.org/html/2503.00385v1)
- **ZOMAML**: Zeroth-order Model-Agnostic Meta-Learning using finite differences [ZOMAML-LQR](https://arxiv.org/abs/2401.14534)

## Environment

The system uses simplified Boeing aircraft dynamics with a 4-dimensional state space:
- Velocity deviation
- Angle of attack
- Pitch rate
- Pitch angle deviation

Control is applied through elevator deflection to minimize quadratic cost.

## Parameters

Key configurable parameters include:
- Training rounds per algorithm (default: 10)
- Epochs per round (default: 300)
- Tasks per round (default: 10)
- Target epoch for few-shot evaluation (default: 25)
- Learning rates for inner/outer loops
- Noise and perturbation scales

See `./scripts/experiment.sh --help` for complete parameter list.

## Output

The experiment generates:
- Training curves comparing algorithm performance
- Statistical analysis with mean and standard deviation
- Visualization using seaborn plots
- Output saved as `{****}_multifold.png`

## Technical Details

The implementation has analytical cost computation via discrete Lyapunov equations for computational efficiency, serving as a replacement for Monte Carlo rollouts. Natural gradients are computed using Fisher Information Matrix approximation.