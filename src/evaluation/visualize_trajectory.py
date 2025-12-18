import sys
import os
# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from shapely.geometry import Polygon as ShapelyPolygon, LinearRing

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED, Status
from configs import *

def plot_scene(env, trajectory, save_path, idx):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot obstacles
    for obstacle in env.map.obstacles:
        # obstacle.shape is likely a LinearRing
        if isinstance(obstacle.shape, LinearRing):
            x, y = obstacle.shape.xy
            ax.fill(x, y, color='gray', alpha=0.5)
        elif isinstance(obstacle.shape, ShapelyPolygon):
            x, y = obstacle.shape.exterior.xy
            ax.fill(x, y, color='gray', alpha=0.5)
        else:
            # Fallback for other shapes if any
            try:
                x, y = obstacle.shape.xy
                ax.fill(x, y, color='gray', alpha=0.5)
            except:
                pass

    # Plot start (Green)
    if isinstance(env.map.start_box, LinearRing):
        sx, sy = env.map.start_box.xy
        ax.plot(sx, sy, color='green', linewidth=2, label='Start')
        # Fill start for better visibility
        ax.fill(sx, sy, color='green', alpha=0.3)

    # Plot dest (Red)
    if isinstance(env.map.dest_box, LinearRing):
        dx, dy = env.map.dest_box.xy
        ax.plot(dx, dy, color='red', linewidth=2, label='Goal')
        # Fill dest for better visibility
        ax.fill(dx, dy, color='red', alpha=0.3)

    # Plot trajectory
    traj_x = [state.loc.x for state in trajectory]
    traj_y = [state.loc.y for state in trajectory]
    ax.plot(traj_x, traj_y, color='blue', linewidth=1, label='Trajectory')

    # Set limits
    ax.set_xlim(env.map.xmin, env.map.xmax)
    ax.set_ylim(env.map.ymin, env.map.ymax)
    ax.set_aspect('equal')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_title(f'Scene Visualization {idx}')
    ax.set_xlabel(f'Scene Size: [{env.map.xmin}, {env.map.xmax}] x [{env.map.ymin}, {env.map.ymax}]')
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None, help='Path to the trained agent checkpoint')
    parser.add_argument('--img_ckpt', type=str, default=None, help='Path to the image encoder checkpoint')
    args = parser.parse_args()

    # Setup environment
    # Use rgb_array mode to avoid window popping up, though we don't use the render output for matplotlib
    raw_env = CarParking(fps=100, verbose=False, render_mode='rgb_array')
    env = CarParkingWrapper(raw_env)

    # Setup agent
    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }

    rl_agent = PPO(configs)
    
    # Load checkpoint
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is None:
        # Prefer ckpt directory first (user request). Fall back to searching log/exp if not found.
        default_ckpt = os.path.join(src_dir, 'model', 'ckpt', 'PPO_best.pt')
        if os.path.exists(default_ckpt):
            checkpoint_path = default_ckpt
        else:
            # Try to find the latest checkpoint in log/exp as a fallback
            log_dir = os.path.join(src_dir, 'log', 'exp')
            if os.path.exists(log_dir):
                exps = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
                exps.sort(key=os.path.getmtime, reverse=True)
                for exp in exps:
                    ckpt = os.path.join(exp, 'PPO_best.pt')
                    if os.path.exists(ckpt):
                        checkpoint_path = ckpt
                        break
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        rl_agent.load(checkpoint_path, params_only=True)
        print(f'Loaded pre-trained model from {checkpoint_path}')
    else:
        print('No checkpoint found or provided. Running with random agent.')

    # Load image encoder if needed
    img_encoder_checkpoint = args.img_ckpt
    if img_encoder_checkpoint is None:
         img_encoder_checkpoint = os.path.join(src_dir, 'model', 'ckpt', 'autoencoder.pt')
         
    if USE_IMG and img_encoder_checkpoint is not None and os.path.exists(img_encoder_checkpoint):
        rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=False)

    step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    # Ensure img directory exists
    img_dir = './img'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Run visualization loop
    success_count = 0
    max_retries = 100
    retry_count = 0
    
    while success_count < VISUALIZATION_NUM and retry_count < max_retries:
        obs = env.reset()
        parking_agent.reset()
        done = False
        
        while not done:
            # Use deterministic action selection for evaluation
            action, _ = parking_agent.choose_action(obs, deterministic=True)
            next_obs, _, done, info = env.step(action)
            
            if info['path_to_dest'] is not None:
                parking_agent.set_planner_path(info['path_to_dest'])
            
            obs = next_obs

        # Only save successful trajectories
        if info['status'] == Status.ARRIVED:
            success_count += 1
            # Plot trajectory
            save_path = os.path.join(img_dir, f'traj_{success_count}.png')
            # Remove existing file if it exists to ensure overwrite
            if os.path.exists(save_path):
                os.remove(save_path)
            plot_scene(raw_env, raw_env.vehicle.trajectory, save_path, success_count)
        else:
            retry_count += 1
            print(f"Episode failed with status: {info['status']}, retrying... ({retry_count}/{max_retries})")
    
    if retry_count >= max_retries:
        print(f"Failed to generate {VISUALIZATION_NUM} successful trajectories after {max_retries} attempts.")

if __name__ == "__main__":
    main()
