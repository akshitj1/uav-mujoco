
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from envs.quad import QuadEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import os
from gym import envs as gym_envs
from stable_baselines3.common.callbacks import CheckpointCallback

# converges in 1e5 steps

def train_quad():
    quad_instance = QuadEnv()
    check_env(quad_instance)

    episode_duration=5 #secs
    step_freq = quad_instance.step_freq()
    steps_per_episode=int(episode_duration*step_freq)
    train_episodes = 1e6/steps_per_episode
    training_num_steps=steps_per_episode*train_episodes
    learning_rate=3e-4
    num_parallel_collect_envs=6
    log_interval_episodes = 10//num_parallel_collect_envs


    gym_envs.register(
        id="Quad-v0",
        entry_point="envs.quad:QuadEnv",
        max_episode_steps=steps_per_episode,
        reward_threshold=0.0,
    )

    # todo: relu v/s tanh
    # architerture used in: https://www.researchgate.net/publication/334438612_Learning_to_fly_computational_controller_design_for_hybrid_UAVs_with_reinforcement_learning
    ppo_args = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=[64, 64, dict(vf=[64, 64], pi=[64, 64])]
        # net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
        )

    quad_env = make_vec_env('Quad-v0', n_envs=num_parallel_collect_envs)
    quad_eval_env = make_vec_env('Quad-v0', n_envs=1)
    
    actor_critic_model = PPO(
        policy=ActorCriticPolicy,
        env=quad_env,
        learning_rate=learning_rate,
        policy_kwargs=ppo_args,
        verbose=1)


    data_dir=os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
   
    print('-'*20+'begin training'+'-'*20)
    print('will train for: {} steps'.format(training_num_steps))

    # eval_callback = EvalCallback(
    #     eval_env=quad_eval_env,
    #     eval_freq=training_num_steps//(10*num_parallel_collect_envs),
    #     best_model_save_path=data_dir+'/')
    checkpoint_callback=CheckpointCallback(
        save_freq=training_num_steps//(10*num_parallel_collect_envs),
        save_path=data_dir+'/',
        name_prefix='ckpt',
        verbose=1
    )

    actor_critic_model.learn(
        total_timesteps=training_num_steps,
        log_interval=log_interval_episodes,
        callback=checkpoint_callback)

    
    print('training complete. saving model.')

    actor_critic_model.save(os.path.join(data_dir, 'success_model.zip'))

if __name__ == '__main__':
    train_quad()