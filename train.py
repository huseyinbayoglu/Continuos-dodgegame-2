import os
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from env import GridWorldEnv

import os
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, KVWriter
from stable_baselines3.common.evaluation import evaluate_policy


class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, log_dir, eval_freq=50, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq 
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = log_dir
        self.best_mean_score = -np.inf
        self.episode_count = 0
        os.makedirs(log_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        """Initialize the logger when training starts."""
        # The logger is already initialized in BaseCallback
        return True

    def _on_step(self) -> bool:
        """This method is required by BaseCallback."""
        return True

    def _on_rollout_end(self) -> None:
        """Perform evaluation and logging every eval_freq episodes."""
        self.episode_count += 1
        if self.episode_count % self.eval_freq == 0:
            scores = []
            for _ in range(self.n_eval_episodes):
                # Handle new Gym API that returns (obs, info)
                reset_result = self.eval_env.reset()
                
                # Check if reset returned a tuple (new Gym API) or just an observation (old API)
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]  # Extract observation from tuple
                else:
                    obs = reset_result  # Old API already gives just the observation
                    
                done = False
                episode_score = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Handle step result which can be (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
                    step_result = self.eval_env.step(action)
                    
                    # For newer Gym API with 5 return values (obs, reward, terminated, truncated, info)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    # For older Gym API with 4 return values (obs, reward, done, info)
                    else:
                        obs, reward, done, info = step_result
                    
                    if isinstance(info, list):
                        info = info[0]  # Handle vectorized envs
                episode_score = info.get("score", 0)  # Assuming 'score' key exists in info
                scores.append(episode_score)
            
            mean_score = np.mean(scores)
            max_episode_score = np.max(scores)
            min_episode_score = np.min(scores)
            # Log for TensorBoard - this is key for TensorBoard visualization
            self.logger.record("eval/mean_score", mean_score)
            self.logger.record("eval/best_mean_score", self.best_mean_score)
            self.logger.record("eval/max_episode_score", max_episode_score)
            self.logger.record("eval/min_episode_score", min_episode_score)
            
            # Dump to ensure data is written to disk
            self.logger.dump(self.num_timesteps)
            
            if mean_score > self.best_mean_score:
                self.best_mean_score = mean_score
                self.model.save(os.path.join(self.log_dir, "best_model"))
            
            if self.verbose:
                print(f"Episode: {self.episode_count} - Mean Score: {mean_score}")
        return True

class LrLoggingCallback(BaseCallback):
    def __init__(self, log_freq=100, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        """Log the learning rate at every log_freq steps."""
        if self.n_calls % self.log_freq == 0:
            progress_remaining = 1.0 - (self.num_timesteps / self.model._total_timesteps)
            learning_rate = self.model.learning_rate
            if callable(learning_rate):
                learning_rate = learning_rate(progress_remaining)
            
            # Log for TensorBoard
            self.logger.record("train/learning_rate", learning_rate)
            
            # These additional metrics will be useful for TensorBoard
            self.logger.record("train/timesteps", self.num_timesteps)
            self.logger.record("train/progress", 1.0 - progress_remaining)
            
            # Dump to ensure data is written to disk
            self.logger.dump(self.num_timesteps)
            
            if self.verbose > 1:
                print(f"Timestep: {self.num_timesteps} - Learning Rate: {learning_rate}")
        return True

# Adding a new callback to log additional training metrics
class TrainingMetricsCallback(BaseCallback):
    def __init__(self, log_freq=100, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_score = 0
        self.rollout_rewards = []
        self.rollout_scores = []

    def _on_step(self) -> bool:
        """Log training metrics at every step and log_freq steps."""
        # Get info from the last step
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            rewards = self.locals.get("rewards", [0])
            infos = self.locals.get("infos", [{}])
            
            # Update episode tracking for rewards
            reward_value = rewards[0] if isinstance(rewards, list) else rewards
            self.current_episode_reward += reward_value
            self.current_episode_length += 1
            self.rollout_rewards.append(reward_value)
            
            # Get score from info dictionary - this is separate from reward
            if isinstance(infos, list) and len(infos) > 0:
                # Extract the score as a separate metric from the info dict
                # Don't accumulate it - just record the current score value
                score = infos[0].get("score", 0)
                self.rollout_scores.append(score)
                # Update the current episode score to the latest score
                # (assuming score is cumulative in the environment)
                self.current_episode_score = score
                
            # If episode is done, record metrics
            if dones if isinstance(dones, bool) else dones[0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.episode_scores.append(self.current_episode_score)
                
                # Log episode metrics for TensorBoard
                self.logger.record("train/episode_reward", self.current_episode_reward)
                self.logger.record("train/episode_length", self.current_episode_length)
                self.logger.record("train/episode_score", self.current_episode_score)
                
                # Reset episode tracking
                self.current_episode_reward = 0
                self.current_episode_length = 0
                self.current_episode_score = 0
        
        # Log additional metrics at regular intervals
        if self.n_calls % self.log_freq == 0 and self.n_calls > 0:
            # Log rollout statistics
            if len(self.rollout_rewards) > 0:
                mean_reward = np.mean(self.rollout_rewards)
                std_reward = np.std(self.rollout_rewards)
                self.logger.record("train/mean_rollout_reward", mean_reward)
                self.logger.record("train/std_rollout_reward", std_reward)
                self.rollout_rewards = []  # Reset after logging
            
            # Log game score statistics
            if len(self.rollout_scores) > 0:
                mean_score = np.mean(self.rollout_scores)
                max_score = np.max(self.rollout_scores) if len(self.rollout_scores) > 0 else 0
                self.logger.record("train/mean_rollout_score", mean_score)
                self.logger.record("train/max_rollout_score", max_score)
                self.rollout_scores = []  # Reset after logging
            
            # Log averages over recent episodes
            if len(self.episode_rewards) > 0:
                # Calculate statistics over last 100 episodes or all if less than 100
                recent_range = min(100, len(self.episode_rewards))
                avg_reward = np.mean(self.episode_rewards[-recent_range:])
                avg_length = np.mean(self.episode_lengths[-recent_range:])
                avg_score = np.mean(self.episode_scores[-recent_range:])
                
                self.logger.record("train/avg_reward_last_100", avg_reward)
                self.logger.record("train/avg_length_last_100", avg_length)
                self.logger.record("train/avg_score_last_100", avg_score)
                
                # Track best performance
                if hasattr(self, 'best_avg_score'):
                    if avg_score > self.best_avg_score:
                        self.best_avg_score = avg_score
                else:
                    self.best_avg_score = avg_score
                
                self.logger.record("train/best_avg_score", self.best_avg_score)
            
            # Log exploration rate for DQN if available
            if hasattr(self.model, "exploration_rate"):
                self.logger.record("train/exploration_rate", self.model.exploration_rate)
            
            # Dump to ensure data is written to disk
            self.logger.dump(self.num_timesteps)
            
            if self.verbose > 0:
                print(f"Timestep: {self.num_timesteps}")
                if len(self.episode_rewards) > 0:
                    print(f"  Mean reward: {avg_reward:.2f}")
                    print(f"  Mean score: {avg_score:.2f}")
                    if hasattr(self, 'best_avg_score'):
                        print(f"  Best avg score: {self.best_avg_score:.2f}")
            
        return True
       
def create_env(render_mode=None, size=5.0, num_obstacles=3):
    env = GridWorldEnv(render_mode=render_mode, size=size, num_obstacles=num_obstacles)
    return Monitor(env)

def linear_lr_schedule(initial_value):
    """
    Lineer olarak azalan öğrenme oranı için fonksiyon döndürür.
    
    :param initial_value: Başlangıç öğrenme oranı
    :return: Eğitim boyunca öğrenme oranını hesaplayan fonksiyon
    """
    def func(progress_remaining):
        """
        :param progress_remaining: 1.0 (başlangıç) ile 0.0 (bitiş) arasında değişir.
        :return: Güncellenmiş öğrenme oranı
        """
        return max(1e-8,progress_remaining * initial_value)
    
    return func
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
    eval_callback = CustomEvalCallback(
        eval_env,
        log_dir=log_dir,
        eval_freq=50,
        n_eval_episodes=10,
    )
    
    # Create the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=80000,
        save_path=save_dir,
        name_prefix="dqn_model",
        save_replay_buffer=False,
    )

    # Learning rate logging callback
    lr_callback = LrLoggingCallback(log_freq=100)
    
    # New training metrics callback
    metrics_callback = TrainingMetricsCallback(log_freq=100)
    
    # Pass an initial learning rate value to the scheduler
    initial_lr = 0.001  # Set your desired initial learning rate
    
    # Create and train the agent
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,  # This is crucial for TensorBoard integration
        learning_rate=linear_lr_schedule(initial_lr),
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
    
    # Use all callbacks
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, lr_callback, metrics_callback],
        progress_bar=True,
        tb_log_name="DQN"  # This creates a subdirectory in log_dir for this run
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
    parser.add_argument("--algorithm", type=str, default="dqn", choices=["ppo", "dqn", "a2c"], 
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=1000000, 
                        help="Total timesteps for training")
    parser.add_argument("--obstacles", type=int, default=3, 
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

    
 

# python3 train.py --algorithm dqn --obstacles 13 --timesteps 1000000
# python3 train.py --algorithm ppo --timesteps 500000 --obstacles 13 --size 5.0
# tensorboard --logdir ./logs/DQN_3/
