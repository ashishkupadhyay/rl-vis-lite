import os
import pandas as pd
import gymnasium as gym
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

class InvertedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(InvertedRewardWrapper, self).__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward * -1.0, terminated, truncated, info

class DataLoggerCallback(BaseCallback):
    def __init__(self, log_path: str, verbose: int = 0):
        super(DataLoggerCallback, self).__init__(verbose)
        self.log_path = log_path
        self.buffer = []

    def _on_step(self) -> bool:
        max_q_value = 0
        if isinstance(self.model, DQN):
            obs = self.locals['new_obs']
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
            with th.no_grad():
                q_values = self.model.q_net(obs_tensor)
                max_q_value = q_values.max().item()

        cumulative_reward = None
        if self.locals['dones'][0]:
            if 'episode' in self.locals['infos'][0]:
                cumulative_reward = self.locals['infos'][0]['episode']['r']

        self.buffer.append({
            'timestep': self.num_timesteps,
            'max_q_value': max_q_value,
            'cumulative_reward': cumulative_reward
        })
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.buffer)
        df.to_csv(self.log_path, index=False)
        print(f"\nLog saved to {self.log_path}")

if __name__ == "__main__":
    GAME = "CartPole-v1"
    LOG_FILE = f'broken_log_{GAME}.csv'
    
    train_env = gym.make(GAME)
    train_env = InvertedRewardWrapper(train_env)
    
    model = DQN('MlpPolicy', train_env, verbose=1)
    logger_callback = DataLoggerCallback(log_path=LOG_FILE)
    
    print(f"--- STARTING BROKEN AGENT TRAINING ON {GAME} ---")
    model.learn(total_timesteps=30000, callback=logger_callback)
    print("--- BROKEN TRAINING COMPLETE ---")
    train_env.close()

