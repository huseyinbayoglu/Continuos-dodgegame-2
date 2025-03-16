import os
import argparse
import numpy as np
import time
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
"""
Hi. Firstly, thanks for the help. I had some bug in get_obs method. I fixed it and the agent performed much better. But still cant play very well. My agent can go to target but cant learn to avoid balls. what should i do about it? if you have free time, can you help me?
"""
from env import GridWorldEnv

def load_model(model_path, model_type="ppo", vec_normalize_path=None):
    """Load a trained model and optionally the VecNormalize stats"""
    if model_type.lower() == "ppo":
        model = PPO.load(model_path)
    elif model_type.lower() == "dqn":
        model = DQN.load(model_path)
    elif model_type.lower() == "a2c":
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, vec_normalize_path
 
def test_model(model, vec_normalize_path=None, num_episodes=5, size=5.0, num_obstacles=13, render=True, record=False):
    """Test a trained model in the environment"""
    # Create the environment
    env = GridWorldEnv(render_mode="human" if render else None, size=size, num_obstacles=num_obstacles)
    # If we have VecNormalize stats, use them
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        # Wrap environment in a VecEnv for compatibility
        env = DummyVecEnv([lambda: env])
        # Load normalization stats
        env = VecNormalize.load(vec_normalize_path, env)
        # Disable updates to stats during testing
        env.training = False
        env.norm_reward = False

    # Setup recording if needed
    if record:
        import pygame
        import imageio
        frames = []

    # Run test episodes
    episode_rewards = []
    episode_steps = []
    successes = 0
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not (done or truncated):
            # If using VecNormalize, need to reshape observation
            if vec_normalize_path and os.path.exists(vec_normalize_path):
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                action = action[0]  # Extract the action from the batch
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if record and hasattr(env, "render"):
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)
            
            # Add a small delay to visualize properly
            if render:
                time.sleep(0.05)
            
            # Print progress
            if step % 10 == 0:
                print(f"Step: {step}, Distance to goal: {info.get('distance_to_goal', 'N/A'):.2f}")
        
        # Check if we succeeded or failed
        if info.get('distance_to_goal', float('inf')) < 0.5:
            print(f"Episode {episode+1} - SUCCESS! Reward: {total_reward:.2f}, Steps: {step}")
            successes += 1
        else:
            print(f"Episode {episode+1} - FAILED. Reward: {total_reward:.2f}, Steps: {step}")
        
        episode_rewards.append(total_reward)
        episode_steps.append(step)
    
    # Save the recording if enabled
    if record and frames:
        imageio.mimsave('navigation_agent.gif', frames, fps=30)
        print(f"Recording saved as navigation_agent.gif")
    
    # Print summary statistics
    print("\n===== Results =====")
    print(f"Success rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average steps: {np.mean(episode_steps):.2f}")
    
    # Close the environment
    env.close()
    
    return {
        'success_rate': successes/num_episodes,
        'avg_reward': np.mean(episode_rewards),
        'avg_steps': np.mean(episode_steps)
    }

def generate_heatmap(model, vec_normalize_path=None, size=5.0, num_obstacles=3, resolution=40):
    """
    Generate a heatmap of the agent's value function or Q-values
    across the grid world to visualize the learned policy
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    
    # Create the environment without rendering
    env = GridWorldEnv(render_mode=None, size=size, num_obstacles=num_obstacles)
    
    # If using VecNormalize
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Create a grid of positions
    x = np.linspace(0, size, resolution)
    y = np.linspace(0, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Initialize value and action maps
    value_map = np.zeros((resolution, resolution))
    action_map = np.zeros((resolution, resolution), dtype=int)
    
    # Reset environment to get a fixed goal and obstacles
    obs, _ = env.reset(seed=42)  # Fixed seed for reproducibility
    
    # Extract goal and obstacle positions
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        state = obs.reshape(-1)
    else:
        state = obs
    
    goal_pos = state[2:4]
    obstacles = []
    
    # Extract obstacle positions for visualization
    for i in range(env.num_obstacles):
        start_idx = 4 + i * 4  # Each obstacle has pos(2) + vel(2)
        obstacles.append(state[start_idx:start_idx+2])
    
    # For each position in the grid
    for i in range(resolution):
        for j in range(resolution):
            # Set character position
            character_pos = [X[i, j], Y[i, j]]
            
            # Skip positions that would be inside obstacles
            inside_obstacle = False
            for obs_pos in obstacles:
                if np.linalg.norm(np.array(character_pos) - np.array(obs_pos)) < env.obstacle_size + env.character_size:
                    inside_obstacle = True
                    break
            
            if inside_obstacle:
                value_map[i, j] = np.nan  # Mark as invalid
                continue
            
            # Construct the observation (replace the character position)
            obs_copy = state.copy()
            obs_copy[0], obs_copy[1] = character_pos
            
            # Reshape observation if using VecNormalize
            if vec_normalize_path and os.path.exists(vec_normalize_path):
                obs_copy = obs_copy.reshape(1, -1)
            
            # Get the value estimate (using the Q-value or value function)
            if isinstance(model, DQN):
                # For DQN, get Q-values for all actions and take max
                q_values = model.q_net(model.policy.obs_to_tensor(obs_copy)[0])
                if hasattr(q_values, "detach"):
                    q_values = q_values.detach().numpy()
                best_action = np.argmax(q_values)
                value_map[i, j] = q_values[0, best_action]
                action_map[i, j] = best_action
            else:
                # For A2C/PPO, use the value function
                action, _ = model.predict(obs_copy, deterministic=True)
                if hasattr(model, "policy") and hasattr(model.policy, "evaluate_actions"):
                    _, values, _ = model.policy.evaluate_actions(
                        model.policy.obs_to_tensor(obs_copy)[0],
                        model.policy.convert_to_torch(action)
                    )
                    if hasattr(values, "detach"):
                        values = values.detach().numpy()
                    value_map[i, j] = values[0]
                    action_map[i, j] = action[0] if isinstance(action, np.ndarray) else action
                else:
                    # Fallback: just use a binary "can reach goal" indicator
                    value_map[i, j] = -np.linalg.norm(np.array(character_pos) - goal_pos)
                    action_map[i, j] = -1
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot value heatmap
    im = axes[0].imshow(value_map, origin='lower', extent=[0, size, 0, size], 
                    cmap='viridis', interpolation='bilinear')
    fig.colorbar(im, ax=axes[0], label='Estimated Value')
    axes[0].set_title('Agent Value Function')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    
    # Plot goal and obstacles
    axes[0].plot(goal_pos[0], goal_pos[1], 'g*', markersize=15, label='Goal')
    for i, obs_pos in enumerate(obstacles):
        axes[0].add_patch(plt.Circle((obs_pos[0], obs_pos[1]), env.obstacle_size, 
                                   color='red', alpha=0.7, label=f'Obstacle {i+1}' if i == 0 else ""))
    
    # Create a discrete colormap for actions
    action_colors = ['gray', 'blue', 'green', 'red', 'purple', 'orange']
    action_labels = ['None/Invalid', 'Left', 'Right', 'Up', 'Down', 'Do Nothing']
    
    # Create a custom colormap for actions
    cmap = colors.ListedColormap(['gray', 'blue', 'green', 'red', 'purple', 'orange'][:5])  # Only use 5 actions
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot action heatmap
    action_map[np.isnan(value_map)] = -1  # Set invalid positions
    im2 = axes[1].imshow(action_map, origin='lower', extent=[0, size, 0, size], 
                         cmap=cmap, norm=norm, interpolation='nearest')
    cbar = fig.colorbar(im2, ax=axes[1], ticks=[-1, 0, 1, 2, 3, 4])
    cbar.set_label('Action')
    cbar.set_ticklabels(['Invalid', 'Left', 'Right', 'Up', 'Down', 'Do Nothing'][:5])  # Only use 5 actions
    axes[1].set_title('Agent Policy (Actions)')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    
    # Plot goal and obstacles again
    axes[1].plot(goal_pos[0], goal_pos[1], 'g*', markersize=15, label='Goal')
    for i, obs_pos in enumerate(obstacles):
        axes[1].add_patch(plt.Circle((obs_pos[0], obs_pos[1]), env.obstacle_size, 
                                    color='red', alpha=0.7, label=f'Obstacle {i+1}' if i == 0 else ""))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('policy_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Policy visualization saved as 'policy_visualization.png'")

# python3 test.py --model-path ./models/dqn_model_1000000_steps.zip --model-type dqn --vec-normalize ./models/vec_normalize.pkl --episodes 5 --obstacles 13 --record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained RL agent in GridWorld")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model-type", type=str, default="ppo", choices=["ppo", "dqn", "a2c"], 
                        help="Type of the trained model")
    parser.add_argument("--vec-normalize", type=str, default=None, 
                        help="Path to saved VecNormalize stats (used with PPO/A2C)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--size", type=float, default=5.0, help="Size of the grid world")
    parser.add_argument("--obstacles", type=int, default=13, help="Number of obstacles")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--record", action="store_true", help="Record a GIF of an episode")
    parser.add_argument("--visualize-policy", action="store_true", 
                        help="Generate a heatmap visualization of the policy/value function")
    
    args = parser.parse_args()

    # Load the model
    model, vec_normalize_path = load_model(args.model_path, args.model_type, args.vec_normalize)
    
    # Test the model
    results = test_model(
        model, 
        vec_normalize_path=args.vec_normalize,
        num_episodes=args.episodes,
        size=args.size,
        num_obstacles=args.obstacles,
        render=not args.no_render,
        record=args.record
    )
    
    # Generate policy visualization if requested
    if args.visualize_policy:
        generate_heatmap(
            model,
            vec_normalize_path=args.vec_normalize,
            size=args.size,
            num_obstacles=args.obstacles
        )
