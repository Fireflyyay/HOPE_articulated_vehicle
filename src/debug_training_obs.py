
import torch
import numpy as np
import os
import sys
from collections import OrderedDict

# Add src to path
sys.path.append(os.getcwd())

from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from model.agent.ppo_agent import PPOAgent
from configs import *

def debug_training_obs():
    print("Initializing Environment...")
    # Initialize environment
    env = CarParking(
        use_img_observation=USE_IMG,
        use_lidar_observation=True,
        use_action_mask=USE_ACTION_MASK
    )
    env = CarParkingWrapper(env)
    
    print("Resetting Environment...")
    obs = env.reset()
    
    print("\n=== Observation Debug Info ===")
    for key, value in obs.items():
        if value is None:
            print(f"{key}: None")
            continue
            
        print(f"\nKey: {key}")
        print(f"Type: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"Shape: {value.shape}")
            print(f"Dtype: {value.dtype}")
            print(f"Min: {np.min(value):.4f}")
            print(f"Max: {np.max(value):.4f}")
            print(f"Mean: {np.mean(value):.4f}")
            if key == 'target':
                print(f"Values: {value}")
    
    # Load Model
    print("\n=== Model Loading Debug Info ===")
    model_path = "log/exp/ppo_20251201_160820/PPO_best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Initialize agent
    agent_configs = {
        "state_dim": env.observation_shape,
        "observation_shape": env.observation_shape, # Add this
        "action_dim": env.action_space.shape[0],
        "action_range": (env.action_space.low, env.action_space.high),
        "actor_layers": ACTOR_CONFIGS,
        "critic_layers": CRITIC_CONFIGS
    }
    
    agent = PPOAgent(agent_configs, load_params=True)
    
    # Load checkpoint manually to inspect
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    if 'state_norm' in checkpoint:
        sn = checkpoint['state_norm']
        print("\nStateNorm in Checkpoint:")
        if isinstance(sn, dict):
            print(f"Keys: {sn.keys()}")
            if 'state_mean' in sn:
                print(f"Mean keys: {sn['state_mean'].keys()}")
                print(f"Target Mean: {sn['state_mean']['target']}")
                print(f"Target Std: {sn['state_std']['target']}")
        else:
            print(f"Type: {type(sn)}")
            print(f"Mean: {sn.state_mean}")

    # Run one step
    print("\n=== Inference Debug Info ===")
    action, _ = agent.get_action(obs)
    print(f"Raw Action (Normalized): {action}")
    
    # Rescale action
    rescaled_action = env.action_func(action, env.env.action_space)
    print(f"Rescaled Action: {rescaled_action}")

if __name__ == "__main__":
    debug_training_obs()
