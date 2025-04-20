import os
import argparse
import numpy as np
import time
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from env import GridWorldEnv
from matplotlib import colors
import matplotlib.patches as patches
import torch # DQN için gerekli olabilir
import matplotlib.pyplot as plt 
# seed 32

def load_model(model_path, model_type="dqn", vec_normalize_path=None):
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
 
def test_model(model, vec_normalize_path=None, num_episodes=5, size=5.0, num_obstacles=13, render=False, record=False):
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
    episode_scores = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not (done or truncated):
            # If using VecNormalize, need to reshape observation
            if vec_normalize_path and os.path.exists(vec_normalize_path):
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)[0]
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if record and hasattr(env, "render"):
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)
            
           
     
        episode_rewards.append(total_reward)
        episode_steps.append(step)
        episode_scores.append(env.score)

    
    # Print summary statistics
    print("\n===== Results =====")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average steps: {np.mean(episode_steps):.2f}")
    
    # Close the environment
    env.close()
    
    return {
        'avg_score': np.mean(episode_scores),
        'avg_reward': np.mean(episode_rewards),
        'avg_steps': np.mean(episode_steps)
    }





def generate_heatmap(model, size=5.0, num_obstacles=13, resolution=150, seed=42):
    """
    Generate a heatmap of the agent's value function and policy across the grid world.
    UPDATED for the new state representation (no frame stacking, relative positions).

    Parameters:
    -----------
    model : stable_baselines3 model
        Trained RL model (PPO, DQN, or A2C)
    size : float
        The size of the square grid from the environment
    num_obstacles : int
        Number of obstacles to place in the environment (should match training)
    resolution : int
        Resolution of the heatmap (grid size)
    seed : int
        Random seed for reproducible obstacle/goal placement
    """

    # --- Environment Setup ---
    # Create the base environment function
    env = GridWorldEnv(render_mode="human", size=size, num_obstacles=num_obstacles)


    # --- Seed and Reset Environment ---
    # We need the state of obstacles and goal from a fixed reset

    # env.seed(seed)
    _ = env.reset(seed = seed)


    # --- Create Grid for Heatmap ---
    x = np.linspace(0, size, resolution)
    y = np.linspace(0, size, resolution)
    X, Y = np.meshgrid(x, y)

    # Create empty list to collect observations
    observations = []

    char_points = zip(X.ravel(), Y.ravel())

    for xi, yi in char_points:
        env._character_position = np.array([xi, yi])
        obs = env._get_obs()
        observations.append(obs)

    # Convert to (batch_size, state_size) NumPy array
    observations = np.array(observations)

    num_points = resolution * resolution
    predicted_actions_flat = np.full(num_points, -1, dtype=int)
    predicted_values_flat = np.full(num_points, np.nan, dtype=np.float32)

    try:
        with torch.no_grad():
            device = model.device
            obs_tensor = torch.as_tensor(observations).to(device)

            if isinstance(model, DQN):
                # DQN: Q değerlerini al
                q_values_tensor = model.q_net(obs_tensor)
                q_values = q_values_tensor.cpu().numpy() # (num_points, num_actions)
                predicted_actions_flat = np.argmax(q_values, axis=1)
                predicted_values_flat = np.max(q_values, axis=1)

            elif isinstance(model, (PPO, A2C)):
                actions_pred, _states = model.predict(observations, deterministic=True)
                predicted_actions_flat = actions_pred

                features = model.policy.extract_features(obs_tensor)
                if model.policy.share_features_extractor:
                    latent_pi, latent_vf = model.policy.mlp_extractor(features)
                else:
                    pi_features, vf_features = features
                    latent_vf = model.policy.mlp_extractor.forward_value(vf_features)

                value_tensor = model.policy.value_net(latent_vf)
                predicted_values_flat = value_tensor.cpu().numpy().flatten()

            else:
                actions_pred, _states = model.predict(observations, deterministic=True)
                predicted_actions_flat = actions_pred



    except Exception as e:
        import traceback
        traceback.print_exc()


    value_map = predicted_values_flat.reshape((resolution, resolution))
    action_map = predicted_actions_flat.reshape((resolution, resolution))




    action_counts = np.unique(action_map, return_counts=True)
    print(f"  Action Map counts: {dict(zip(action_counts[0], action_counts[1]))}")






    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Value Heatmap ---
    vmin = np.nanmin(value_map)
    vmax = np.nanmax(value_map)
    # Handle cases where all values are NaN or the same
    if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
        vmin = 0
        vmax = 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    value_cmap = plt.get_cmap('viridis').copy() # Get a copy to modify
    value_cmap.set_bad('lightgrey') # Color NaN values (e.g., inside obstacles)

    im1 = axes[0].imshow(
        value_map,
        extent=[0, size, 0, size], # Set axes limits
        cmap=value_cmap,
        norm=norm,
        interpolation='bilinear', # Smoother look for value
        aspect='equal' # Ensure square cells
    )
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Estimated Value')
    axes[0].set_title('Value Function Heatmap')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    # axes[0].invert_yaxis() # REMOVED - origin='lower' handles this

    # --- Action Heatmap ---
    # Define colors for VALID actions: 0:Left, 1:Right, 2:Up, 3:Down, 4:Stay
    valid_action_colors = ['blue', 'green', 'purple', 'red', 'orange']
    action_cmap = colors.ListedColormap(valid_action_colors)
    action_cmap.set_bad('lightgrey') # Color for invalid areas (NaN)

    # Prepare action map for plotting: Replace -1 with NaN
    action_map_plot = action_map.astype(float) # Transpose and convert to float for NaN
    action_map_plot[action_map_plot == -1] = 4

    # Define boundaries and normalization for the 5 valid actions
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    action_norm = colors.BoundaryNorm(bounds, action_cmap.N)

    im2 = axes[1].imshow(
        action_map_plot,
        origin = "upper",
        extent=[0, size, 0, size], # Set axes limits
        cmap=action_cmap,
        norm=action_norm,
        interpolation='nearest',
        aspect='equal' 
    )
    # Create colorbar ONLY for valid actions (0-4)
    cbar2 = fig.colorbar(im2, ax=axes[1], ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
    cbar2.set_label('Action')
    cbar2.set_ticklabels(['Left', 'Right', 'Up', 'Down', 'Stay']) # Set labels for valid actions
    axes[1].set_title('Policy (Action) Heatmap')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    


    for ax in axes:
        # Goal
        goal_circle = patches.Circle((env._goal_position[0], env.size - env._goal_position[1]), env.goal_size,
                                   linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.8, label='Goal')
        ax.add_patch(goal_circle)


        # Obstacles
        for i, obs_pos in enumerate(env._obstacle_positions):
            obs_circle = patches.Circle((obs_pos[0], env.size - obs_pos[1]), env.obstacle_size,
                                      linewidth=1, edgecolor='black', facecolor='red', alpha=0.6,
                                      label='Obstacle' if i == 0 else "") # Label only first obstacle
            ax.add_patch(obs_circle)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
        # ax.invert_yaxis() 

    plt.suptitle(f'Agent Policy and Value Function Visualization (Seed: {seed})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # Save the figure
    filename = 'rl_agent_heatmap.png'
    plt.savefig(filename, dpi=300)
    print(f"Heatmap visualization saved as '{filename}'")

    # Close the environment - Moved outside the plotting section in the previous full code
    # env.close() # Make sure this is called once after generate_heatmap finishes

    # Return the original (non-transposed) maps if needed elsewhere
    # return value_map, action_map
    # Close the environment
    env.close()

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
    parser.add_argument("--seed", type=int, default=48 ,help="Seed for reset")
    parser.add_argument("--visualize-policy", action="store_true", 
                        help="Generate a heatmap visualization of the policy/value function")
    # python3 test.py --model-path ./models/final_model_dqn.zip --model-type dqn --episodes 150 --obstacles 13 
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
    print(results)
    # Generate policy visualization if requested
    if args.visualize_policy:

        # --- Generate Heatmap ---
        generate_heatmap(
            model=model,
            seed=args.seed
        )
