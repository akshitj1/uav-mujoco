import gym
import time
from stable_baselines3 import PPO
import os

from utils import sync
import numpy as np
from envs.quad import QuadEnv
from gym import envs as gym_envs

if __name__ == '__main__':
    quad_instance = QuadEnv()

    episode_duration=5 #secs
    step_freq = quad_instance.step_freq()
    steps_per_episode=int(episode_duration*step_freq)    
    gym_envs.register(
        id="Quad-v0",
        entry_point="envs.quad:QuadEnv",
        max_episode_steps=steps_per_episode,
        reward_threshold=75.0,
    )


     #### Show, record a video, and log the model's performance #
    quad_env = gym.make('Quad-v0')
    
    data_dir=os.path.join(os.path.dirname(__file__), 'data')
    tuned_model_path = os.path.join(data_dir, 'success_model.zip')

    tuned_model = PPO.load(tuned_model_path)

    start_time = time.time()
    obs = quad_env.reset()
    # render freq. and control freq. is equal!!!
    sim_duration = 5 # secs
    while True:
        for i in range(sim_duration*quad_instance.step_freq()):
            action, _states = tuned_model.predict(obs,
                                            deterministic=True # OPTIONAL 'deterministic=False'
                                            )
            obs, reward, done, info = quad_env.step(action)
            quad_env.render()
            elapsed_real = time.time() - start_time
            elapsed_sim = i*(1./quad_instance.step_freq())
            time.sleep(max(elapsed_sim - elapsed_real, 0.))
            if done:
                obs = quad_env.reset() # OPTIONAL EPISODE HALT
                break
    quad_env.close()