import os
import numpy as np
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Import custom callbacks from your existing code
from train import CustomEvalCallback, LrLoggingCallback, TrainingMetricsCallback, create_env, linear_lr_schedule

def continue_training_dqn(
    model_path, 
    total_timesteps=500000, 
    log_dir="./logs/continued/", 
    save_dir="./models/continued/",
    eval_freq=50,
    learning_rate=None,
    env_size=5.0,
    num_obstacles=3
):
    """
    Continue training a pre-trained DQN model.
    
    Args:
        model_path (str): Path to the pre-trained model file
        total_timesteps (int): Number of additional timesteps to train for
        log_dir (str): Directory to save logs
        save_dir (str): Directory to save model checkpoints
        eval_freq (int): Frequency of evaluation in episodes
        learning_rate (float): New learning rate (if None, use model's current learning rate)
        env_size (float): Size of the grid world environment
        num_obstacles (int): Number of obstacles in the environment
    
    Returns:
        model: The trained DQN model
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = create_env(size=env_size, num_obstacles=num_obstacles)
    
    # Create evaluation environment
    eval_env = create_env(size=env_size, num_obstacles=num_obstacles)
    
    # Load the pre-trained model
    print(f"Loading pre-trained model from {model_path}")
    model = DQN.load(model_path, env=env)
    
    # Update the learning rate if specified
    if learning_rate is not None:
        print(f"Updating learning rate to {learning_rate}")
        model.learning_rate = linear_lr_schedule(learning_rate)
    
    # Create a unique name for the TensorBoard log
    base_name = os.path.basename(model_path)
    tb_log_name = f"DQN_continued_{base_name}"
    
    # Set up callbacks
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        log_dir=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix=f"dqn_continued_{base_name}",
        save_replay_buffer=False,
    )
    
    # Add learning rate and metrics logging
    lr_callback = LrLoggingCallback(log_freq=100)
    metrics_callback = TrainingMetricsCallback(log_freq=100)
        # Continue training
    print(f"Continuing training for {total_timesteps} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, lr_callback, metrics_callback],
        progress_bar=True,
        tb_log_name=tb_log_name,
        reset_num_timesteps=False  # Important: Continue counting timesteps from where we left off
    )
    
    # Save the final model
    final_model_path = os.path.join(save_dir, f"final_continued_{base_name}")
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")
    
    return model
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue training a pre-trained DQN model")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to the pre-trained model file")
    parser.add_argument("--timesteps", type=int, default=500000, 
                        help="Number of additional timesteps to train for")
    parser.add_argument("--obstacles", type=int, default=13, 
                        help="Number of obstacles in the environment")
    parser.add_argument("--size", type=float, default=5.0, 
                        help="Size of the grid world")
    parser.add_argument("--learning-rate", type=float, default=None, 
                        help="New learning rate (if not specified, use model's current learning rate)")
    parser.add_argument("--log-dir", type=str, default="./logs/continued/", 
                        help="Directory to save logs")
    parser.add_argument("--save-dir", type=str, default="./models/continued/", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--eval-freq", type=int, default=50, 
                        help="Frequency of evaluation in episodes")
    
    args = parser.parse_args()
    # Continue training the model
    continue_training_dqn(
        model_path=args.model,
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        eval_freq=args.eval_freq,
        learning_rate=args.learning_rate,
        env_size=args.size,
        num_obstacles=args.obstacles
    )