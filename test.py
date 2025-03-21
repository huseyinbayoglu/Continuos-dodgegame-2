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
 
def test_model(model, vec_normalize_path=None, num_episodes=5, size=5.0, num_obstacles=9, render=True, record=False):
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

def generate_heatmap(model, vec_normalize_path=None, size=5.0, num_obstacles=13, resolution=150, seed=42):
    """
    Generate a heatmap of the agent's value function and policy across the grid world
    
    Parameters:
    -----------
    model : stable_baselines3 model
        Trained RL model (PPO, DQN, or A2C)
    vec_normalize_path : str, optional
        Path to the saved VecNormalize stats
    size : float
        The size of the square grid
    num_obstacles : int
        Number of obstacles to place in the environment
    resolution : int
        Resolution of the heatmap (grid size)
    seed : int
        Random seed for reproducible results
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import matplotlib.patches as patches
    
    # Create the environment without rendering
    env = GridWorldEnv(render_mode=None, size=size, num_obstacles=num_obstacles)
    
    # If using VecNormalize
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Reset environment with fixed seed for reproducibility
    obs, _ = env.reset(seed=seed)
    
    # Extract state information
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        # For VecEnv wrapped environments
        state = obs.reshape(-1)
    else:
        state = obs
    
    # Process state to get relevant information
    # Find the correct indices for the current state in the history
    single_state_size = env.single_state_size
    current_state = state[-single_state_size:]  # Use most recent state
    
    # Extract positions from the current state
    character_pos = current_state[:2]
    goal_pos = current_state[2:4]
    
    # Extract obstacle positions
    obstacles = []
    for i in range(env.num_obstacles):
        # Each obstacle has position (x,y) and velocity (vx,vy)
        start_idx = 13 + i * 4  # Skip character, goal, and 3 closest obstacles info
        obstacles.append(current_state[start_idx:start_idx+2])
    
    # Create a grid of positions
    x = np.linspace(0, size, resolution)
    y = np.linspace(0, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Initialize maps for values and actions
    value_map = np.zeros((resolution, resolution))
    action_map = np.zeros((resolution, resolution), dtype=int)
    
    # For each position in the grid
    for i in range(resolution):
        for j in range(resolution):
            # Set character position
            pos = np.array([X[i, j], Y[i, j]])
            
            # Skip positions that would be inside obstacles
            inside_obstacle = False
            for obs_pos in obstacles:
                if np.linalg.norm(pos - obs_pos) < (env.obstacle_size):
                    inside_obstacle = True
                    break
            
            if inside_obstacle:
                value_map[i, j] = np.nan  # Mark as invalid
                action_map[i, j] = -1  # Invalid action
                continue
            
            # Create a copy of the observation
            obs_copy = state.copy()
            
            # Update character position in all three state history entries
            for k in range(3):
                start_idx = k * single_state_size
                obs_copy[start_idx] = pos[0]
                obs_copy[start_idx + 1] = pos[1]
            
            # Process observation for prediction
            if vec_normalize_path and os.path.exists(vec_normalize_path):
                obs_copy = obs_copy.reshape(1, -1)
            
            # Get action and value estimate
            try:
                if isinstance(model, DQN):
                    # For DQN, get Q-values for all actions
                    q_values = model.q_net(model.policy.obs_to_tensor(obs_copy)[0])
                    if hasattr(q_values, "detach"):
                        q_values = q_values.detach().numpy()
                    best_action = np.argmax(q_values)
                    value_map[i, j] = q_values[0, best_action]
                    action_map[i, j] = best_action
                else:
                    # For A2C/PPO, predict action and get value
                    action, _ = model.predict(obs_copy, deterministic=True)
                    
                    # For PPO/A2C, extract value from value network
                    if hasattr(model.policy, "value_net"):
                        features = model.policy.extract_features(model.policy.obs_to_tensor(obs_copy)[0])
                        latent_val = model.policy.mlp_extractor.forward_value(features)
                        values = model.policy.value_net(latent_val)
                        
                        if hasattr(values, "detach"):
                            values = values.detach().numpy()
                        value_map[i, j] = values[0]
                    else:
                        # Fallback: use negative distance to goal as value
                        value_map[i, j] = -np.linalg.norm(pos - goal_pos)
                    
                    action_map[i, j] = action[0] if isinstance(action, np.ndarray) else action
            except Exception as e:
                # If prediction fails, use fallback values
                value_map[i, j] = -np.linalg.norm(pos - goal_pos)
                action_map[i, j] = -1

    # Create the figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Normalize value map for better visualization
    vmin, vmax = np.nanmin(value_map), np.nanmax(value_map)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot value heatmap
    im1 = axes[0].imshow(
        value_map, 
        origin='lower', 
        extent=[0, size, 0, size],
        cmap='viridis', 
        norm=norm,
        interpolation='bilinear'
    )
    axes[0].invert_yaxis()  # Y eksenini ters çevir
    
    # Add colorbar
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_label('Estimated Value')
    
    # Set titles and labels
    axes[0].set_title('Value Function Heatmap', fontsize=14)
    axes[0].set_xlabel('X Position', fontsize=12)
    axes[0].set_ylabel('Y Position', fontsize=12)
    
    # Custom colormap for actions
    action_cmap = colors.ListedColormap(['blue', 'green', 'purple', 'red', 'orange'])
    bounds = [-.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    action_norm = colors.BoundaryNorm(bounds, action_cmap.N)
    
    # Plot action heatmap
    im2 = axes[1].imshow(
        action_map, 
        origin='lower', 
        extent=[0, size, 0, size],
        cmap=action_cmap, 
        norm=action_norm,
        interpolation='nearest'
    )
    axes[1].invert_yaxis()  # Y eksenini ters çevir
    # Add colorbar for actions
    cbar2 = fig.colorbar(im2, ax=axes[1], ticks=[0, 1, 2, 3, 4])
    cbar2.set_label('Action')
    
    # FIXED: Correct the action labels to match your environment
    # According to your env.py, the actions are: 0:Left, 1:Right, 2:Up, 3:Down, 4:Do Nothing
    cbar2.set_ticklabels(['Left', 'Right', 'Up', 'Down', 'Do Nothing'])
    
    # Create a textual explanation of the action mapping
    # action_desc = """
    # Action Mapping:
    # 0 (Blue): Left
    # 1 (Green): Right
    # 2 (Purple): Up
    # 3 (Red): Down
    # 4 (Orange): Do Nothing
    # """
    # axes[1].text(1.05, 0.5, action_desc, transform=axes[1].transAxes, 
    #             verticalalignment='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Set titles and labels
    axes[1].set_title('Policy (Action) Heatmap', fontsize=14)
    axes[1].set_xlabel('X Position', fontsize=12)
    axes[1].set_ylabel('Y Position', fontsize=12)
    
    # Add goal and obstacles to both plots
    for ax in axes:
        # Add goal
        goal_circle = plt.Circle((goal_pos[0], goal_pos[1]), env.goal_size, 
                                color='yellow', alpha=0.7, label='Goal')
        ax.add_patch(goal_circle)
        
        # Add character
        char_rect = plt.Rectangle((character_pos[0] - env.character_size, 
                                  character_pos[1] - env.character_size),
                                 env.character_size * 2, env.character_size * 2,
                                 color='green', alpha=0.7, label='Character')
        ax.add_patch(char_rect)
        
        # Add obstacles
        for i, obs_pos in enumerate(obstacles):
            obs_circle = plt.Circle((obs_pos[0], obs_pos[1]), env.obstacle_size, 
                                  color='red', alpha=0.5, 
                                  label='Obstacle' if i == 0 else None)
            ax.add_patch(obs_circle)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a title to the whole figure
    plt.suptitle(f'Agent Policy and Value Function Visualization (Obstacles: {num_obstacles})', 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('rl_agent_heatmap.png', dpi=300, bbox_inches='tight')
    
    # Optional: create a trajectories plot
    create_trajectory_plot(model, env, vec_normalize_path, size, obstacles, seed)
    
    print("Heatmap visualization saved as 'rl_agent_heatmap.png'")
    return value_map, action_map

def create_trajectory_plot(model, env, vec_normalize_path=None, size=5.0, obstacles=None, seed=42, num_episodes=5):
    """
    Create a plot showing example trajectories of the agent
    """
    import matplotlib.pyplot as plt
    
    # Reset environment with fixed seed
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset(seed=seed)
    
    plt.figure(figsize=(10, 10))
    
    # Extract goal position
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        # For VecEnv wrapped environments
        state = obs.reshape(-1)
    else:
        state = obs
    
    # Process state to get relevant information
    # Find the correct indices for the current state in the history
    single_state_size = env.single_state_size
    current_state = state[-single_state_size:]  # Use most recent state
    
    # Extract positions from the current state
    character_pos = current_state[:2]
    goal_pos = current_state[2:4]
    
    # Extract obstacle positions
    obstacles_pos = []
    for i in range(env.num_obstacles):
        # Each obstacle has position (x,y) and velocity (vx,vy)
        start_idx = 13 + i * 4  # Skip character, goal, and 3 closest obstacles info
        obstacles_pos.append(current_state[start_idx:start_idx+2])
    
    # Draw goal
    plt.scatter(goal_pos[0], goal_pos[1], s=200, color='yellow', marker='*', label='Goal')
    
    # Draw obstacles
    for i, obs_pos in enumerate(obstacles_pos):
        plt.scatter(obs_pos[0], obs_pos[1], s=100, color='red', alpha=0.5, 
                   label='Obstacle' if i == 0 else None)
        # Draw obstacle radius
        circle = plt.Circle((obs_pos[0], obs_pos[1]), env.obstacle_size, 
                          color='red', alpha=0.2, fill=True)
        plt.gca().add_patch(circle)
    
    # Generate trajectories
    for episode in range(num_episodes):
        # Reset environment but keep same obstacles by resetting the seed
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            obs, _ = env.reset(seed=seed + episode)
        else:
            obs, _ = env.reset(seed=seed + episode)
        
        # Extract character position
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            state = obs.reshape(-1)
        else:
            state = obs
        
        # Track positions
        positions = [state[:2]]  # Start with initial position
        done = False
        truncated = False
        
        # Run episode
        while not (done or truncated):
            # Predict action
            if vec_normalize_path and os.path.exists(vec_normalize_path):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Take action
            obs, _, done, truncated, _ = env.step(action)
            
            # Extract character position
            if vec_normalize_path and os.path.exists(vec_normalize_path):
                state = obs.reshape(-1)
            else:
                state = obs
            
            # Add position to trajectory
            positions.append(state[:2])
            
            # Prevent infinite loops
            if len(positions) > 500:
                break
        
        # Convert to numpy array
        positions = np.array(positions)
        
        # Plot trajectory
        plt.plot(positions[:, 0], positions[:, 1], '-', alpha=0.7, label=f'Episode {episode+1}')
        
        # Mark start and end
        plt.scatter(positions[0, 0], positions[0, 1], s=100, marker='o', 
                   facecolors='none', edgecolors='green', label='Start' if episode == 0 else None)
        plt.scatter(positions[-1, 0], positions[-1, 1], s=100, marker='x', 
                   color='blue', label='End' if episode == 0 else None)
    
    # Set up plot
    plt.title('Agent Trajectories', fontsize=16)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # Save the figure
    plt.savefig('agent_trajectories.png', dpi=300, bbox_inches='tight')
    print("Trajectory visualization saved as 'agent_trajectories.png'")


# python3 test.py --model-path ./models/dqn_model_240000_steps.zip --model-type dqn --vec-normalize ./models/vec_normalize.pkl --episodes 15 --obstacles 13
# python3 test.py --model-path ./models/iyidqn.zip --model-type dqn --vec-normalize ./models/vec_normalize.pkl --episodes 1 --obstacles 13 --visualize-policy
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained RL agent in GridWorld")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model-type", type=str, default="dqn", choices=["ppo", "dqn", "a2c"], 
                        help="Type of the trained model")
    parser.add_argument("--vec-normalize", type=str, default=None, 
                        help="Path to saved VecNormalize stats (used with PPO/A2C)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of test episodes")
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
