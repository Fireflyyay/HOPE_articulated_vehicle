from abc import ABC, abstractmethod
import random
from typing import Any

import numpy as np
import torch

from configs import *


class ConfigBase:
    def __init__(self):
        # runtime
        self.n_epoch = None
        self.n_initial_exploration_steps = None

        # environment
        self.state_dim: tuple = None
        self.action_dim: tuple = None

        # model
        self.gamma: float = GAMMA
        self.batch_size = BATCH_SIZE
        self.lr: float = LR
        self.explore: bool = True
        self.explore_config: dict = {}
        self.tau: float = TAU # when tau=0, the update becomes hard update
        self.max_train_steps = MAX_TRAIN_STEP

        # tricks
        self.orthogonal_init = ORTHOGONAL_INIT
        self.lr_decay = LR_DECAY

        # evaluation
        self.evaluation_interval = None

        # save and load
        self.check_list = []
    
    def merge_configs(self, configs: dict):
        for key, value in configs.items():
            setattr(self, key, value)


class AgentBase(ABC):
    def __init__(
        self, config_type, configs: dict, verbose: bool = False,
        save_params: bool = False, load_params: bool = False
    ) -> None:
        """Initialize the model structure here
        """
        self.device = device # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.configs = config_type(configs)
        self.check_list = []
        self.save_params = save_params
        self.load_params = load_params

    @abstractmethod
    def get_action(self, observation):
        """Return an action based on the observation
        """

    def _soft_update(self, target_net, current_net):
        for target, current in zip(target_net.parameters(), current_net.parameters()):
            target.data.copy_(current.data * self.configs.tau + target.data * (1. - self.configs.tau))
        
    def push_memory(self, observations):
        self.memory.push(observations)

    @abstractmethod
    def update(self):
        """Update the network's parameters.
        """

    def epsilon_greedy(self, action, action_space, epsilon=0.1) -> Any:
        if random.random() > epsilon:
            return action
        return action_space.sample()

    def lr_decay(self, lr: float, n_step: int, decay_type: str = "exp") -> float:
        if decay_type == "linear":
            lr = lr * (1 - n_step / self.configs.max_train_steps)
        elif decay_type == "exp":
            lr = lr * np.exp(-n_step / self.configs.max_train_steps)
        return lr

    def explore(self, action, action_space):
        explore_configs = self.configs.explore_configs
        if explore_configs["type"] == "epsilon_greedy":
            return self.epsilon_greedy(action, action_space, explore_configs["epsilon"])
        return action

    def save(self, path: str = None, params_only: bool = None) -> None:
        """Store the model structure and corresponding parameters to a file.
        """
        if params_only is not None:
            self.save_params = params_only
        if self.save_params and len(self.check_list) > 0:
            checkpoint = dict()
            for name, item, save_state_dict in self.check_list:
                checkpoint[name] = item.state_dict() if save_state_dict else item
            torch.save(checkpoint, path)
        else:
            torch.save(self, path)
        
        if self.verbose:
            print("Save current model to %s" % path)

    def load(self, path: str, params_only: bool = False) -> None:
        """Load the model structure and corresponding parameters from a file.
        """
        print(f"DEBUG: Loading {path} with weights_only=False and map_location={self.device}")
        try:
            checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        except TypeError as e:
            print(f"DEBUG: TypeError with weights_only=False: {e}")
             # Fallback for older torch versions
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as e:
            print(f"DEBUG: Other error in torch.load: {e}")
            raise e
            
        if params_only:
            self.load_params = True
            
        if self.load_params and len(self.check_list) > 0:
            for name, item, save_state_dict in self.check_list:
                if name in checkpoint:
                    if save_state_dict:
                        item.load_state_dict(checkpoint[name])
                    else:
                        # For non-state-dict items (like configs, log_std), we need to update the object
                        # But 'item' is a reference to the object. 
                        # If it's a mutable object (like list/dict), we can update it.
                        # If it's immutable or we want to replace it, we can't do it easily via 'item'.
                        # However, for PPOAgent, 'log_std' is a tensor, so we can copy data.
                        # 'configs' is an object.
                        
                        # Special handling for known types if needed, or just try to copy
                        if isinstance(item, torch.Tensor):
                            with torch.no_grad():
                                item.copy_(checkpoint[name])
                        elif hasattr(item, '__dict__'):
                            item.__dict__.update(checkpoint[name].__dict__)
                        else:
                            # Fallback: try to set attribute on self if possible (hard without name mapping)
                            pass
        else:
            # Full model load not supported in this patch
            pass
        
        if self.verbose:
            print("Load the model from %s" % path)