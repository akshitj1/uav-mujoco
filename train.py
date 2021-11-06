
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from envs.quad import QuadEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import os
from gym import envs as gym_envs
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# converges in 1e5 steps

def train_quad():
    quad_instance = QuadEnv()
    check_env(quad_instance)
    # train_env = make_vec_env(
    #     env_id=QuadEnv,
    #     seed=0
    # )
    episode_duration=5 #secs
    step_freq = quad_instance.step_freq()
    steps_per_episode=int(episode_duration*step_freq)
    train_episodes = int(1e2) #1e5/steps_per_episode
    learning_rate=1e-3
    log_interval_episodes = 10


    gym_envs.register(
        id="Quad-v0",
        entry_point="envs.quad:QuadEnv",
        max_episode_steps=steps_per_episode,
        reward_threshold=0.0,
    )

    # todo: relu v/s tanh
    ppo_args = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=[64, 64, dict(vf=[64, 64], pi=[64, 64])]
        # net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
        )
    actor_critic_model = PPO(
        policy=ActorCriticPolicy,
        env='Quad-v0',
        learning_rate=learning_rate,
        policy_kwargs=ppo_args,
        verbose=1)
    
    print('-'*20+'begin training'+'-'*20)
    print('will train for: {} steps'.format(train_episodes*steps_per_episode))
    actor_critic_model.learn(
        total_timesteps=train_episodes*steps_per_episode,
        log_interval=log_interval_episodes)

    data_dir=os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print('training complete. saving model.')

    actor_critic_model.save(os.path.join(data_dir, 'success_model.zip'))

if __name__ == '__main__':
    train_quad()