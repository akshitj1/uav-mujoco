
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from envs.quad import QuadEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import os
from gym import envs as gym_envs
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


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
    train_episodes = 100


    gym_envs.register(
        id="Quad-v0",
        entry_point="envs.quad:QuadEnv",
        max_episode_steps=steps_per_episode,
        reward_threshold=75.0,
    )



    # todo: relu v/s tanh
    ppo_args = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
        )
    actor_critic_model = PPO(
        policy=ActorCriticPolicy,
        env='Quad-v0',
        policy_kwargs=ppo_args,
        verbose=1)
    
    print('-'*20+'begin training'+'-'*20)
    print('will train for: {} steps'.format(train_episodes*steps_per_episode))
    log_interval_episodes = 10
    actor_critic_model.learn(
        total_timesteps=train_episodes*steps_per_episode,
        log_interval=log_interval_episodes*steps_per_episode)

    data_dir=os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print('training complete. saving model.')

    actor_critic_model.save(os.path.join(data_dir, 'success_model.zip'))

if __name__ == '__main__':
    train_quad()