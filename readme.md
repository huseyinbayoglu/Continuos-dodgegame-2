# GridWorld Reinforcement Learning Project

A customizable reinforcement learning environment built with Gymnasium that features a dynamic 2D grid world with moving obstacles. The agent must navigate from a random starting position to a goal while avoiding collisions.

## Environment Overview

### GridWorldEnv

The environment simulates a 2D grid where:
- An agent (green square) navigates toward a goal (yellow circle)
- Multiple moving obstacles (red circles) bounce around the grid
- The agent must reach the goal while avoiding collisions with obstacles

#### Features
- Customizable grid size and number of obstacles
- Moving obstacles with bouncing physics
- State representation includes:
  - Agent position
  - Goal position
  - Detailed information about the three closest obstacles
  - Position and velocity of all obstacles
  - History of the last 3 states for better decision making
- Rendering capabilities using PyGame

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/gridworld-rl.git
cd gridworld-rl
```

2. Install the required dependencies:
```bash
pip install gymnasium numpy pygame stable-baselines3
```

## Usage

### Training an Agent

Use the `train.py` script to train an agent with your preferred algorithm:

```bash
python train.py --algorithm dqn --timesteps 1000000 --obstacles 13 --size 5.0
```

#### Training Options:
- `--algorithm`: Choose between "ppo", "dqn", or "a2c" (default: dqn)
- `--timesteps`: Number of timesteps for training (default: 1000000)
- `--obstacles`: Number of obstacles in the environment (default: 3)
- `--size`: Size of the grid world (default: 5.0)
- `--log-dir`: Directory to save logs (default: "./logs/")
- `--save-dir`: Directory to save models (default: "./models/")

### Evaluation

To evaluate a trained model, create a script like this:

```python
from stable_baselines3 import PPO, DQN, A2C
from env import GridWorldEnv

# Create environment with rendering for visualization
env = GridWorldEnv(render_mode="human", size=5.0, num_obstacles=13)

# Load the model (use the appropriate algorithm)
model = DQN.load("./models/best_model")

# Reset the environment
obs, info = env.reset()

# Run the simulation
total_reward = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        print(f"Episode ended with reward: {total_reward}")
        break
        
env.close()
```

## Environment Details

### State Space
The observation space includes:
- Current position of the agent (x, y)
- Position of the goal (x, y)
- For the three closest obstacles:
  - X-distance to the agent
  - Y-distance to the agent
  - Linear distance to the agent
- For all obstacles:
  - Position (x, y)
  - Velocity components (vx, vy)
- History of the last 3 states

### Action Space
The agent can take 5 discrete actions:
- 0: Move left
- 1: Move right
- 2: Move up
- 3: Move down
- 4: Do nothing

### Rewards
- -0.1: Step penalty (encourages efficient paths)
- 0.3: Reward for moving closer to the goal
- -10: Penalty for colliding with an obstacle (episode ends)
- 50: Reward for reaching the goal (episode ends)

### Episode Termination
Episodes terminate when:
- The agent collides with an obstacle
- The agent reaches the goal
- Maximum number of steps is reached (default: 500)

## Algorithms

The project supports multiple reinforcement learning algorithms:

### PPO (Proximal Policy Optimization)
Best for complex environments with continuous action spaces. More sample efficient than traditional policy gradient methods.

### DQN (Deep Q-Network)
Effective for discrete action spaces. Works well with the 5 available actions in this environment.

### A2C (Advantage Actor-Critic)
Combines policy gradient and value-based methods. Good balance between stability and sample efficiency.

## Customization

You can easily customize the environment by modifying parameters in `env.py`:
- `size`: Change the grid size
- `num_obstacles`: Adjust the number of moving obstacles
- `character_velocity`: Change the agent's movement speed
- `obstacle_velocity`: Adjust how fast obstacles move
- `max_steps`: Modify the maximum episode length

## License

[MIT License](LICENSE)