import os
import numpy as np
import time
import argparse
import cv2
from env import GridWorldEnv

def record_gameplay(model_path=None, output_path="gameplay.mp4", 
                   size=5.0, num_obstacles=13, fps=30, max_episodes=3):
    """Record gameplay of the trained model as an MP4 video"""
    
    # Load the model if path is provided
    model = None
    if model_path:
        try:
            # Try PPO first
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"Loaded PPO model from {model_path}")
        except (ImportError, ValueError, TypeError) as e:
            print(f"Failed to load as PPO model: {e}")
            try:
                # Try DQN
                from stable_baselines3 import DQN
                model = DQN.load(model_path)
                print(f"Loaded DQN model from {model_path}")
            except (ImportError, ValueError, TypeError) as e:
                print(f"Failed to load as DQN model: {e}")
                try:
                    # Try A2C
                    from stable_baselines3 import A2C
                    model = A2C.load(model_path)
                    print(f"Loaded A2C model from {model_path}")
                except (ImportError, ValueError, TypeError) as e:
                    print(f"Failed to load as A2C model: {e}")
                    print("Will run with random actions instead.")
    
    # Create environment with rgb_array render mode
    env = GridWorldEnv(render_mode="rgb_array", size=size, num_obstacles=num_obstacles)
    
    # Reset the environment to initialize all positions
    obs, _ = env.reset()
    
    # Get dimensions from the first frame
    frame = env._render_frame()  # Use the internal method directly
    height, width, _ = frame.shape
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write the first frame
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Stats tracking
    episode_rewards = []
    episode_steps = []
    episode_scores = []
    successes = 0
    
    print(f"Recording gameplay to {output_path}...")
    
    # Play for specified number of episodes
    for episode in range(max_episodes):
        print(f"\nEpisode {episode+1}/{max_episodes}")
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        # Record initial frame after reset
        frame = env._render_frame()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Run episode
        while not (done or truncated):
            # Get action from model or random if no model
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()  # Random action
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Add reward and increment step counter
            total_reward += reward
            step += 1
            
            # Capture and write frame
            frame = env._render_frame()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Print progress
            if step % 20 == 0:
                print(f"Step: {step}, Distance to goal: {info.get('distance_to_goal', 'N/A'):.2f}")
        
        # Episode finished - check outcome
        if 'score' in info and info['score'] > 0:
            print(f"Episode {episode+1} - SUCCESS! Score: {info['score']}, Reward: {total_reward:.2f}, Steps: {step}")
            successes += 1
        else:
            outcome = "COLLISION" if step < env.max_steps and total_reward < -30 else "TIMEOUT"
            print(f"Episode {episode+1} - {outcome}. Reward: {total_reward:.2f}, Steps: {step}")
        
        episode_rewards.append(total_reward)
        episode_steps.append(step)
        episode_scores.append(info["score"])
    
    # Clean up
    video_writer.release()
    env.close()
    
    # Print summary
    print("\n===== Recording Summary =====")
    print(f"Output: {output_path}")
    print(f"Success rate: {successes}/{max_episodes} ({successes/max_episodes*100:.1f}%)")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average steps: {np.mean(episode_steps):.2f}")
    print(f"Average scores: {np.mean(episode_scores):.2f}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record gameplay of a trained RL agent')
    parser.add_argument('--model_path', type=str,default="models/continued/üçüncü_son", help='Path to the trained model (optional)')
    parser.add_argument('--output', type=str, default='gameplay.mp4', help='Output video file path')
    parser.add_argument('--size', type=float, default=5.0, help='Size of the grid world')
    parser.add_argument('--obstacles', type=int, default=13, help='Number of obstacles')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to record')
    
    args = parser.parse_args()
    
    # Record the gameplay
    output_file = record_gameplay(
        model_path=args.model_path,
        output_path=args.output,
        size=args.size,
        num_obstacles=args.obstacles,
        fps=args.fps,
        max_episodes=args.episodes
    )
    
    print(f"Video saved to: {output_file}")