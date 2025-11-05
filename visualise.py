import time
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

GAME = "ALE/Breakout-v5"
MODEL_PATH = f"ppo_{GAME.split('/')[1].split('-')[0].lower()}.zip"

print(f"--- LOADING MODEL: {MODEL_PATH} ---")
trained_model = PPO.load(MODEL_PATH)

print(f"--- VISUALIZING AGENT FOR {GAME} (a window will pop up) ---")

eval_env = make_atari_env(GAME, n_envs=1, seed=0)
eval_env = VecFrameStack(eval_env, n_stack=4)

obs = eval_env.reset()
for _ in range(5000):
    action, _states = trained_model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    eval_env.render("human")
    time.sleep(0.1)
    if dones[0]:
        print("Episode finished. Resetting...")
        obs = eval_env.reset()

eval_env.close()
print("--- VISUALIZATION COMPLETE ---")