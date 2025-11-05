import os
import pandas as pd
import gymnasium as gym
import torch as th
import time
import ale_py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecEnvWrapper

class InvertedRewardVecWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super(InvertedRewardVecWrapper, self).__init__(venv)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards * -1.0, dones, infos
    
    def reset(self):
        return self.venv.reset()

class DataLoggerCallback(BaseCallback):
    def __init__(self, log_path: str, verbose: int = 0):
        super(DataLoggerCallback, self).__init__(verbose)
        self.log_path = log_path
        self.buffer = []

    def _on_step(self) -> bool:
        value_estimate = 0
        if isinstance(self.model, PPO):
            obs_tensor, _ = self.model.policy.obs_to_tensor(self.locals['new_obs'])
            with th.no_grad():
                value_estimate = self.model.policy.predict_values(obs_tensor).mean().item()

        cumulative_reward = None
        for info in self.locals['infos']:
            if 'episode' in info:
                cumulative_reward = info['episode']['r']
                break 

        self.buffer.append({
            'timestep': self.num_timesteps,
            'value_estimate': value_estimate,
            'cumulative_reward': cumulative_reward
        })
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.buffer)
        df.to_csv(self.log_path, index=False)
        print(f"\nLog saved to {self.log_path}")

if __name__ == "__main__":
    GAME = "ALE/Breakout-v5"
    LOG_FILE = f'broken_log_{GAME.replace("/", "-")}.csv'
    MODEL_PATH = f"ppo_broken_{GAME.split('/')[1].split('-')[0].lower()}.zip"

    train_env = make_atari_env(GAME, n_envs=1, seed=0)
    train_env = InvertedRewardVecWrapper(train_env)
    train_env = VecFrameStack(train_env, n_stack=4)

    model = PPO('CnnPolicy', train_env, verbose=1)
    logger_callback = DataLoggerCallback(log_path=LOG_FILE)

    print(f"--- STARTING BROKEN AGENT TRAINING ON {GAME} ---")
    model.learn(total_timesteps=250000, callback=logger_callback)
    print("--- TRAINING COMPLETE ---")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    train_env.close()

    print(f"\n--- VISUALIZING BROKEN AGENT (a window will pop up) ---")

    trained_model = PPO.load(MODEL_PATH)

    eval_env = make_atari_env(GAME, n_envs=1, seed=0)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    obs = eval_env.reset()
    for _ in range(2000):
        action, _states = trained_model.predict(obs, deterministic=True)
        obs, rewards, dones, info = eval_env.step(action)
        eval_env.render("human")
        time.sleep(0.02)
        if dones[0]:
            obs = eval_env.reset()

    eval_env.close()
    print("--- VISUALIZATION COMPLETE ---")

