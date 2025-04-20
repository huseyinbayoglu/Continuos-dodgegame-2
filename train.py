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

class RolloutMetricsCallback(BaseCallback):
    """
    Rollout sırasında reward ve score değerlerini takip eden ve TensorBoard'a kaydeden özel callback.
    """
    def __init__(self, log_dir="./logs",log_freq = 1, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq
        os.makedirs(log_dir, exist_ok=True)
        
        # Rollout metrikleri için gerekli değişkenler
        self.rollout_rewards = []
        self.rollout_scores = []
        self.episode_reward = 0
        self.current_score = 0
        self.episodes_in_rollout = 0

    def _on_training_start(self) -> None:
        """Eğitim başladığında değişkenleri sıfırla."""
        self.rollout_rewards = []
        self.rollout_scores = []
        self.episode_reward = 0
        self.current_score = 0
        self.episodes_in_rollout = 0
        return True

    def _on_step(self) -> bool:
        """Her adımda reward'ı topla ve env info'dan score'u al."""
        # Reward bilgisini al
        reward = self.locals.get('rewards')
        if reward is not None:
            if isinstance(reward, list):
                reward = reward[0]  # Vektörleştirilmiş env için
            self.episode_reward += reward
        
        # Env info'dan score bilgisini al
        info = self.locals.get('infos')
        if info is not None:
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            
            if 'score' in info:
                self.current_score = info['score']
        
        # Episode bitişini kontrol et
        dones = self.locals.get('dones')
        if dones is not None:
            if isinstance(dones, list):
                done = dones[0]  # Vektörleştirilmiş env için
            else:
                done = dones
                
            if done:
                # Episode bittiğinde reward ve score'u kaydet
                self.rollout_rewards.append(self.episode_reward)
                self.rollout_scores.append(self.current_score)
                self.episodes_in_rollout += 1
                
                # Değişkenleri sıfırla
                self.episode_reward = 0
                
                if self.verbose > 1:
                    print(f"Episode {self.episodes_in_rollout} - Score: {self.current_score} - Reward: {self.rollout_rewards[-1]:.2f}")
        
        return True

    def _on_rollout_end(self) -> None:
        """
        Her rollout sonunda metrikler hesaplanır ve TensorBoard'a kaydedilir.
        """
        if len(self.rollout_rewards) > 0:
            # Rollout metriklerini hesapla
            mean_reward = np.mean(self.rollout_rewards)
            mean_score = np.mean(self.rollout_scores)
            max_reward = np.max(self.rollout_rewards)
            max_score = np.max(self.rollout_scores)
            
            # TensorBoard için kaydet
            self.logger.record("rollout/mean_reward", mean_reward)
            self.logger.record("rollout/mean_score", mean_score)
            self.logger.record("rollout/max_reward", max_reward)
            self.logger.record("rollout/max_score", max_score)
            self.logger.record("rollout/episodes", self.episodes_in_rollout)
            
            # Standart sapma ve diğer istatistikler (en az 2 episode varsa)
            if len(self.rollout_rewards) > 1:
                reward_std = np.std(self.rollout_rewards)
                score_std = np.std(self.rollout_scores)
                self.logger.record("rollout/reward_std", reward_std)
                self.logger.record("rollout/score_std", score_std)
            
            # TensorBoard için verileri kaydet
            self.logger.dump(self.num_timesteps)
            
            if self.verbose > 0:
                print(f"Rollout end - Mean Score: {mean_score:.2f} - Mean Reward: {mean_reward:.2f} - Episodes: {self.episodes_in_rollout}")
            
            # Bir sonraki rollout için değişkenleri sıfırla
            self.rollout_rewards = []
            self.rollout_scores = []
            self.episodes_in_rollout = 0
            
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
    metrics_callback = RolloutMetricsCallback(log_freq=100)
    
    # Pass an initial learning rate value to the scheduler
    initial_lr = 0.001  # Set your desired initial learning rate
    
    # Create and train the agent
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,  # This is crucial for TensorBoard integration
        learning_rate=linear_lr_schedule(initial_lr),
        buffer_size=100_000,
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

    
 

# python3 train.py --algorithm dqn --obstacles 13 --timesteps 4000000
# python3 train.py --algorithm ppo --timesteps 500000 --obstacles 13 --size 5.0
# tensorboard --logdir ./logs/DQN_3/
