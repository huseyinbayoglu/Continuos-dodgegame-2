import os
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from env import GridWorldEnv

def create_env(render_mode=None, size=5.0, num_obstacles=13):
    env = GridWorldEnv(render_mode=render_mode, size=size, num_obstacles=num_obstacles)
    return Monitor(env)

def train_ppo(env_fn, total_timesteps=1000000, log_dir="./logs/", save_dir="./models/"):
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environments
    env = DummyVecEnv([env_fn])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create a separate environment for evaluation
    eval_env = DummyVecEnv([env_fn])
    # Use the same normalization stats as the training environment
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    
    # Create the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Create the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Create and train the agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    model.save(os.path.join(save_dir, "final_model"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    return model

def train_dqn(env_fn, total_timesteps=1000000, log_dir="./logs/", save_dir="./models/"):
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = env_fn()
    
    # Create a separate environment for evaluation
    eval_env = env_fn()
    
    # Create the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Create the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="dqn_model",
        save_replay_buffer=True,
    )
    
    # Create and train the agent
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    model.save(os.path.join(save_dir, "final_model_dqn"))
    
    return model

def train_a2c(env_fn, total_timesteps=1000000, log_dir="./logs/", save_dir="./models/"):
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environments
    env = DummyVecEnv([env_fn])
    
    # Create a separate environment for evaluation
    eval_env = DummyVecEnv([env_fn])
    
    # Create the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Create the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="a2c_model",
        save_replay_buffer=False,
    )
    
    # Create and train the agent
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    model.save(os.path.join(save_dir, "final_model_a2c"))
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agents on GridWorld")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "dqn", "a2c"], 
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=1000000, 
                        help="Total timesteps for training")
    parser.add_argument("--obstacles", type=int, default=13, 
                        help="Number of obstacles in the environment")
    parser.add_argument("--size", type=float, default=5.0, 
                        help="Size of the grid world")
    parser.add_argument("--log-dir", type=str, default="./logs/", 
                        help="Directory to save logs")
    parser.add_argument("--save-dir", type=str, default="./models/", 
                        help="Directory to save models")
    args = parser.parse_args()
    # Create the environment function with the desired parameters
    env_fn = lambda: create_env(size=args.size, num_obstacles=args.obstacles)
    # Train the agent with the selected algorithm
    if args.algorithm == "ppo":
        model = train_ppo(env_fn, total_timesteps=args.timesteps,
                          log_dir=args.log_dir, save_dir=args.save_dir)
    elif args.algorithm == "dqn":
        model = train_dqn(env_fn, total_timesteps=args.timesteps,
                          log_dir=args.log_dir, save_dir=args.save_dir)
    elif args.algorithm == "a2c":
        model = train_a2c(env_fn, total_timesteps=args.timesteps,
                          log_dir=args.log_dir, save_dir=args.save_dir)
    
    print(f"Training complete! Model saved to {args.save_dir}")

    
 

# python3 train.py --algorithm dqn --timesteps 1000000 --obstacles 13 --size 5.0
# python3 train.py --algorithm ppo --timesteps 1000000 --obstacles 13 --size 5.0
