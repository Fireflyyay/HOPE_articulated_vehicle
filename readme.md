# HOPE for Articulated Vehicles

This repository is a fork of the [HOPE planner](https://github.com/jiamiya/HOPE), adapted for **articulated vehicles** (e.g., tractor-trailers). It implements a Reinforcement Learning-based Hybrid Policy Path Planner for diverse parking scenarios, specifically designed to handle the complex kinematics of articulated vehicles.

## Key Features

*   **Articulated Vehicle Model**: Simulates tractor-trailer kinematics, including hitch angle constraints and off-tracking effects.
*   **Diverse Parking Scenarios**: Supports multiple difficulty levels for both Bay and Parallel parking:
    *   **Normal**: Standard parking scenarios with reasonable space.
    *   **Complex**: Tighter spaces, with initial positions typically perpendicular to the target.
    *   **Extrem**: Very tight spaces, with initial positions typically opposite to the target, requiring complex maneuvering.
*   **Hybrid Policy**: Combines Reinforcement Learning (PPO/SAC) with Reeds-Shepp curves for efficient and robust path planning.
*   **Curriculum Learning**: The training process automatically adapts the difficulty of scenarios based on the agent's performance.

## Setup

1.  Install conda or miniconda.

2.  Clone the repo and create the environment:

    ```Shell
    git clone <your-repo-url>
    cd HOPE_articulated_vehicle
    conda create -n HOPE python==3.8
    conda activate HOPE
    pip3 install -r requirements.txt
    ```

    Make sure to install PyTorch compatible with your CUDA version from [https://pytorch.org/](https://pytorch.org/).

## Usage

### Training

To train the agent using PPO (Proximal Policy Optimization):

```Shell
cd src
python ./train/train_HOPE_ppo.py
```

To train the agent using SAC (Soft Actor-Critic):

```Shell
cd src
python ./train/train_HOPE_sac.py
```

Training logs and checkpoints will be saved in `src/log/exp/`.

### Evaluation

To evaluate a pre-trained agent:

```Shell
cd src
python ./evaluation/eval_mix_scene.py <path_to_checkpoint> --eval_episode 10 --visualize True
```

Example:
```Shell
python ./evaluation/eval_mix_scene.py ./model/ckpt/HOPE_SAC0.pt --eval_episode 10 --visualize True
```

Arguments:
*   `ckpt_path`: Path to the model checkpoint file (`.pt`).
*   `--eval_episode`: Number of episodes to evaluate.
*   `--visualize`: Whether to visualize the simulation (True/False).

## Project Structure

*   `src/configs.py`: Configuration parameters for vehicle dimensions, map levels, and RL hyperparameters.
*   `src/env/`: Environment definitions.
    *   `vehicle.py`: Articulated vehicle dynamics and kinematics.
    *   `parking_map_normal.py`: Logic for generating parking scenarios (Normal, Complex, Extrem).
    *   `map_level.py`: Difficulty level definitions.
*   `src/model/`: RL agent implementations and network architectures.
*   `src/train/`: Training scripts for PPO and SAC.
*   `src/evaluation/`: Evaluation and visualization scripts.

## Original Citation

If you find the original HOPE work useful, please cite:

```bibtex
@article{jiang2024hope,
  title={HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios},
  author={Jiang, Mingyang and Li, Yueyuan and Zhang, Songan and Chen, Siyuan and Wang, Chunxiang and Yang, Ming},
  journal={arXiv preprint arXiv:2405.20579},
  year={2024}
}
```
