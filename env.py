import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=5.0, num_obstacles=13):
        super().__init__()
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Action space: left, right, up, down, do nothing
        self.action_space = spaces.Discrete(5)

        # Observation space: character position (x, y), goal position (x, y),
        # distance to nearest obstacle (1), and for each obstacle: position (x, y) and velocity components (vx, vy)
        self.num_obstacles = num_obstacles
        
        # Calculate single state size - now including distance to nearest obstacle
        self.single_state_size = 4 + 1 + num_obstacles * 4  # character (x,y), goal (x,y), nearest obstacle distance, obstacles (x,y,vx,vy) for each
        
        # For history of 3 states
        obs_size = self.single_state_size * 3
        
        self.observation_space = spaces.Box(
            low=0, high=size, shape=(obs_size,), dtype=np.float32
        )

        # Character and obstacle velocities
        self.character_velocity = 0.15
        self.obstacle_velocity = 0.15

        # Character, goal, and obstacle sizes (radius)
        self.character_size = 0.15
        self.goal_size = 0.15
        self.obstacle_size = 0.12

        # Initialize positions and velocities
        self._character_position = None
        self._goal_position = None
        self._obstacle_positions = None
        self._obstacle_velocities = None
        
        # State history buffer - will store last 3 states
        self._state_history = []

        # For rendering
        self.window = None
        self.clock = None
        self.render_mode = render_mode

        # Maximum episode length
        self.max_steps = 500
        self.current_step = 0

    def _get_nearest_obstacle_distance(self):
        # Calculate distance to the nearest obstacle
        obstacle_distances = [
            np.linalg.norm(self._character_position - self._obstacle_positions[i]) - self.obstacle_size  # Adjust for obstacle radius
            for i in range(self.num_obstacles)
        ]
        return min(obstacle_distances) if obstacle_distances else float('inf')

    def _get_current_state(self):
        # Get distance to nearest obstacle
        nearest_obstacle_distance = self._get_nearest_obstacle_distance()
        
        # Concatenate basic information
        state = np.concatenate(
            [
                self._character_position,
                self._goal_position,
                np.array([nearest_obstacle_distance]),  # Add nearest obstacle distance
            ]
        )

        # Add obstacles positions and velocities
        for i in range(self.num_obstacles):
            state = np.concatenate(
                [
                    state,
                    self._obstacle_positions[i],
                    self._obstacle_velocities[i],
                ]
            )

        return state

    def _get_obs(self):
        # Get current state
        current_state = self._get_current_state()
        
        # Add current state to history
        self._state_history.append(current_state)
        
        # Keep only last 3 states
        if len(self._state_history) > 3:
            self._state_history.pop(0)
        
        # If we have fewer than 3 states, duplicate the earliest state to fill the history
        while len(self._state_history) < 3:
            self._state_history.insert(0, self._state_history[0])
        
        # Concatenate the last 3 states
        obs = np.concatenate(self._state_history)
        
        return obs

    def _get_info(self):
        # Calculate distance to goal
        dist_to_goal = np.linalg.norm(self._character_position - self._goal_position)
        
        # Get the nearest obstacle distance using our helper method
        closest_obstacle = self._get_nearest_obstacle_distance()

        return {
            "distance_to_goal": dist_to_goal,
            "closest_obstacle": closest_obstacle,
            "step_count": self.current_step
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset state history
        self._state_history = []
        
        # Place character at a random position in the lower left quarter
        self._character_position = self.np_random.uniform(
            low=np.array([0.1, 0.1]), high=np.array([self.size, self.size]), size=2
        )
        
        while True:
            self._goal_position = self.np_random.uniform(
                low=np.array([0.1, 0.1]),  
                high=np.array([self.size - 0.1, self.size - 0.1]), 
                size=2
            )
            if np.linalg.norm(self._goal_position - self._character_position) > 1:
                break

        # Initialize obstacles
        self._obstacle_positions = []
        self._obstacle_velocities = []

        for _ in range(self.num_obstacles):
            # Random position ensuring it's not too close to character or goal
            while True:
                position = self.np_random.uniform(low=0.1, high=self.size - 0.1, size=2)
                
                # Check distance from character and goal
                char_dist = np.linalg.norm(position - self._character_position)
                goal_dist = np.linalg.norm(position - self._goal_position)
                
                # Make sure obstacle is not too close to character or goal at start
                if char_dist > 1.0 and goal_dist > 1.0:
                    break
            
            self._obstacle_positions.append(position)
            
            # Random diagonal velocity
            angle = self.np_random.uniform(0, 2 * math.pi)
            vx = self.obstacle_velocity * math.cos(angle)
            vy = self.obstacle_velocity * math.sin(angle)
            self._obstacle_velocities.append(np.array([vx, vy]))

        # Convert lists to numpy arrays for easier manipulation
        self._obstacle_positions = np.array(self._obstacle_positions)
        self._obstacle_velocities = np.array(self._obstacle_velocities)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.current_step += 1
        first_distance = np.linalg.norm(self._character_position - self._goal_position)
        
        # Move character based on action
        if action == 0:  # Left
            self._character_position[0] = max(0, self._character_position[0] - self.character_velocity)
        elif action == 1:  # Right
            self._character_position[0] = min(self.size, self._character_position[0] + self.character_velocity)
        elif action == 2:  # Up
            self._character_position[1] = max(0, self._character_position[1] - self.character_velocity)
        elif action == 3:  # Down
            self._character_position[1] = min(self.size, self._character_position[1] + self.character_velocity)
        # Action 4 is "do nothing"

        # Move obstacles and handle bounces
        for i in range(self.num_obstacles):
            self._obstacle_positions[i] += self._obstacle_velocities[i]
            
            # Bounce off walls
            # X-axis bounds
            if self._obstacle_positions[i][0] <= 0 or self._obstacle_positions[i][0] >= self.size:
                self._obstacle_velocities[i][0] *= -1
                # Ensure within bounds
                self._obstacle_positions[i][0] = np.clip(self._obstacle_positions[i][0], 0, self.size)
            
            # Y-axis bounds
            if self._obstacle_positions[i][1] <= 0 or self._obstacle_positions[i][1] >= self.size:
                self._obstacle_velocities[i][1] *= -1
                # Ensure within bounds
                self._obstacle_positions[i][1] = np.clip(self._obstacle_positions[i][1], 0, self.size)

        # Check for collisions with obstacles
        collision = False
        for i in range(self.num_obstacles):
            distance = np.linalg.norm(self._character_position - self._obstacle_positions[i])
            if distance < (self.character_size + self.obstacle_size):
                collision = True
                break

        # Check for reaching the goal
        goal_reached = np.linalg.norm(self._character_position - self._goal_position) < (self.character_size + self.goal_size)

        # Calculate reward
        reward = -.1
        
        if collision:
            # Penalty for collision
            reward = -10
            terminated = True
        elif goal_reached:
            # Reward for reaching goal
            reward = 10
            terminated = True
        else:
            # Ongoing dynamics: reward for getting closer to goal and staying away from obstacles
            dist_to_goal = np.linalg.norm(self._character_position - self._goal_position)
            if dist_to_goal < first_distance:
                reward = .3
            
            
            # Small penalty for being close to obstacles
            """for i in range(self.num_obstacles):
                obstacle_dist = np.linalg.norm(self._character_position - self._obstacle_positions[i])
                if obstacle_dist < 1.0:
                    reward -= 0.1 * (1.0 - obstacle_dist)"""
            
            terminated = False
        
        # Check for timeout
        if self.current_step >= self.max_steps:
            terminated = True
        
        # In RL, 'truncated' is used for non-terminal timeout
        truncated = False if terminated else (self.current_step >= self.max_steps)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        
        # Conversion from grid coordinates to pixel coordinates
        def coord_to_pixel(coord):
            return int(coord / self.size * self.window_size)
        
        # Draw goal
        pygame.draw.circle(
            canvas,
            (255, 255, 0),
            (coord_to_pixel(self._goal_position[0]), coord_to_pixel(self._goal_position[1])),
            int(self.goal_size / self.size * self.window_size * 0.9),
        )
        
        # Draw character as green square
        square_size = int(self.character_size / self.size * self.window_size * 2)  # Adjust size for visibility
        character_rect = pygame.Rect(
            coord_to_pixel(self._character_position[0]) - square_size // 2,
            coord_to_pixel(self._character_position[1]) - square_size // 2,
            square_size,
            square_size
        )
        pygame.draw.rect(
            canvas,
            (0, 255, 0),  # Green color
            character_rect
        )
        
        # Draw obstacles
        for i in range(self.num_obstacles):
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (coord_to_pixel(self._obstacle_positions[i][0]), coord_to_pixel(self._obstacle_positions[i][1])),
                int(self.obstacle_size / self.size * self.window_size),
            )
            
        if self.render_mode == "human":
            # Add some information to the screen
            font = pygame.font.SysFont(None, 24)
            info = self._get_info()
            
            # Display step count and distances
            step_text = font.render(f"Steps: {self.current_step}/{self.max_steps}", True, (255, 255, 255))
            goal_text = font.render(f"Goal Distance: {info['distance_to_goal']:.2f}", True, (255, 255, 255))
            obstacle_text = font.render(f"Obstacle Distance: {info['closest_obstacle']:.2f}", True, (255, 255, 255))
            
            canvas.blit(step_text, (10, 10))
            canvas.blit(goal_text, (10, 40))
            canvas.blit(obstacle_text, (10, 70))
            
            # Copy our drawings to the window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            # Ensure fixed framerate
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
