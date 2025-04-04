import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode=None, size=5.0, num_obstacles=13):
        super().__init__()
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Action space: left, right, up, down, do nothing
        self.action_space = spaces.Discrete(5)

        self.num_obstacles = num_obstacles
        
        # Calculate single state size - now including detailed info for all obstacles
        # Character (x,y), goal (x,y), 3 closest obstacles (x-dist, y-dist, linear dist), 
        # all obstacles (x,y,vx,vy,linear_dist)
        self.single_state_size = 4 + 9 + num_obstacles * 7  # Added linear distance for each obstacle
        
        # For history of 3 states
        obs_size = self.single_state_size * 3
        
        # Observation space: character position (x, y), goal position (x, y),
        # For each of the 3 closest obstacles: x-distance, y-distance, linear distance
        # and for each obstacle: position (x, y), velocity components (vx, vy), and linear distance
        self.observation_space = spaces.Box(
            low=-size, high=size, shape=(obs_size,), dtype=np.float32
        )

        # Character and obstacle velocities
        self.character_velocity = 0.15
        self.obstacle_velocity = 0.12

        # Character, goal, and obstacle sizes (radius)
        self.character_size = 0.13
        self.goal_size = 0.15
        self.obstacle_size = 0.12

        # Initialize positions and velocities
        self._character_position = None
        self._goal_position = None
        self._obstacle_positions = None
        self._obstacle_velocities = None

        # Initialize score
        self.score = 0
        
        # State history buffer - will store last 3 states
        self._state_history = []

        # For rendering
        self.window = None
        self.clock = None
        self.render_mode = render_mode

        # Maximum episode length
        self.max_steps = 500
        self.current_step = 0

    def _get_three_closest_obstacles_info(self):
        # Calculate distances and information for all obstacles
        obstacle_info = []
        
        for i in range(self.num_obstacles):
            # Calculate x and y distances (can be negative)
            x_dist = self._obstacle_positions[i][0] - self._character_position[0]
            y_dist = self._obstacle_positions[i][1] - self._character_position[1]
            
            # Normalize x and y distances to -1 to 1 range
            x_dist_norm = x_dist / self.size
            y_dist_norm = y_dist / self.size

            # Calculate linear distance (accounting for obstacle size)
            linear_dist = np.linalg.norm([x_dist, y_dist])
            
            # Normalize linear distance: maximum possible distance is diagonal of grid sqrt(2*size^2)
            max_distance = math.sqrt(2) * self.size
            linear_dist_norm = linear_dist / max_distance

            
            obstacle_info.append((x_dist_norm, y_dist_norm, linear_dist_norm))

        # Sort obstacles by linear distance
        obstacle_info.sort(key=lambda x: x[2])
        
        # Get information for the three closest obstacles
        closest_three = obstacle_info[:3]
        
        # If there are fewer than 3 obstacles, pad with zeros
        while len(closest_three) < 3:
            closest_three.append((0.0, 0.0, 1))
            
        # Flatten the list of tuples
        result = []
        for info in closest_three:
            result.extend(info)
            
        return np.array(result, dtype=np.float32)

    def _get_nearest_obstacle_distance(self):
        # Calculate distance to the nearest obstacle
        obstacle_distances = [
            np.linalg.norm(self._character_position - self._obstacle_positions[i]) - self.obstacle_size  # Adjust for obstacle radius
            for i in range(self.num_obstacles)
        ]
        return min(obstacle_distances) if obstacle_distances else float('inf')

    def _get_current_state(self):
        # Get detailed information about the three closest obstacles
        three_closest_info = self._get_three_closest_obstacles_info()
        
        # Normalize character and goal positions
        char_pos_norm = self._character_position / self.size
        goal_pos_norm = self._goal_position / self.size

        # Concatenate basic information
        state = np.concatenate(
            [
                char_pos_norm,
                goal_pos_norm,
                three_closest_info,  # Add detailed obstacle information
            ]
        )

        # Maximum possible distance for normalization
        max_distance = math.sqrt(2) * self.size

        # Add obstacles positions, velocities, and linear distances
        for i in range(self.num_obstacles):
            # Normalize obstacle positions
            obs_pos_norm = self._obstacle_positions[i] / self.size

            # Calculate linear distance to this obstacle
            dx = self._character_position[0] - self._obstacle_positions[i][0]
            dy = self._character_position[1] - self._obstacle_positions[i][1]

            dx_norm = dx / self.size
            dy_norm = dy / self.size

            linear_dist = np.linalg.norm(self._character_position - self._obstacle_positions[i])
            linear_dist_norm = linear_dist / max_distance


            state = np.concatenate(
                [
                    state,
                    obs_pos_norm,
                    self._obstacle_velocities[i],
                    [dx_norm,dy_norm,linear_dist_norm],  # Add linear distance for each obstacle
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
            "step_count": self.current_step,
            "score": self.score
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.score = 0
        
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
        reward = -.4
        terminated = False
        
        if collision:
            # Penalty for collision
            reward = -40
            terminated = True
        elif goal_reached:
            # Reward for reaching goal
            reward = 20
            self.score += 1
            self._place_target()
            # terminated = True
        else:
            # Ongoing dynamics: reward for getting closer to goal and staying away from obstacles
            dist_to_goal = np.linalg.norm(self._character_position - self._goal_position)
            if dist_to_goal < first_distance:
                reward = .3
            
            
            terminated = False
        
        # Check for timeout
        if self.current_step >= self.max_steps:
            terminated = True
        
        # In RL, 'truncated' is used for non-terminal timeout
        truncated = False if terminated else (self.current_step >= self.max_steps or self.score >= 50)

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
            # obstacle_text = font.render(f"Obstacle Distance: {info['closest_obstacle']:.2f}", True, (255, 255, 255))
            score_text = font.render(f"Score:{self.score}",True, (255, 255, 255))
            canvas.blit(step_text, (10, 10))
            canvas.blit(goal_text, (10, 40))
            # canvas.blit(obstacle_text, (10, 70))
            canvas.blit(score_text, (10, 70))
            
            # Copy our drawings to the window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            # Ensure fixed framerate
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def _place_target(self):
        while True:
            self._goal_position = self.np_random.uniform(
                low=np.array([0.1, 0.1]),  
                high=np.array([self.size - 0.1, self.size - 0.1]), 
                size=2
            )
            if np.linalg.norm(self._goal_position - self._character_position) > 1:
                break

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            