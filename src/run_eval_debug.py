
import os
import sys
import torch
import numpy as np
from configs import *
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from model.agent.ppo_agent import PPOAgent

def run_eval():
    print("Initializing Environment...")
    env = CarParking(
        use_img_observation=USE_IMG,
        use_lidar_observation=True,
        use_action_mask=USE_ACTION_MASK
    )
    env = CarParkingWrapper(env)
    
    print("Loading Model...")
    model_path = "log/exp/ppo_20251201_160820/PPO_best.pt"
    
    agent_configs = {
        "state_dim": env.observation_shape,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "action_range": (env.action_space.low, env.action_space.high),
        "actor_layers": ACTOR_CONFIGS,
        "critic_layers": CRITIC_CONFIGS
    }
    
    agent = PPOAgent(agent_configs, load_params=True)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    agent.load(model_path)
    
    # Manually load StateNorm if needed (agent.load_params might do it)
    if 'state_norm' in checkpoint and agent.configs.state_norm:
        print("Loading StateNorm...")
        sn_data = checkpoint['state_norm']
        if isinstance(sn_data, dict):
             if 'state_mean' in sn_data:
                agent.state_normalize.state_mean = sn_data['state_mean']
                agent.state_normalize.state_std = sn_data['state_std']
                agent.state_normalize.S = sn_data['S']
                agent.state_normalize.n_state = sn_data['n_state']
        else:
            agent.state_normalize.state_mean = sn_data.state_mean
            agent.state_normalize.state_std = sn_data.state_std
            agent.state_normalize.S = sn_data.S
            agent.state_normalize.n_state = sn_data.n_state
        agent.state_normalize.fix_parameters()

    print("Running Evaluation Episode...")
    obs = env.reset()
    done = False
    step = 0
    while not done and step < 50:
        action, _ = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        step += 1
        print(f"Step {step}: Reward={reward:.4f}, Status={info['status']}")
        
    print("Done. Check debug_obs_training.jsonl")

if __name__ == "__main__":
    # Clear previous log
    if os.path.exists("debug_obs_training.jsonl"):
        os.remove("debug_obs_training.jsonl")
    run_eval()
