import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode=None, size=5.0, num_obstacles=13, k_closest_obstacles=7):
        super().__init__()
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Action space: left, right, up, down, do nothing
        self.action_space = spaces.Discrete(5)

        self.num_obstacles = num_obstacles
        self.k_closest = k_closest_obstacles 

        # --- State Representation ---
        # State:
        # 1. Relative vector to the target (dx, dy) -> 2 values
        # 2. For the closest k obstacles (for each):
        #    - Relative vector to the obstacle (dx, dy) -> 2 values
        #    - Obstacle velocity (vx, vy) -> 2 values
        # Total state size: 2 + k * 4
        state_size = 2 + self.k_closest * 4
        self.observation_space = spaces.Box(
            # Normalizasyon sonrası değerler genellikle -1 ile 1 arasında olacak
            # Biraz pay bırakarak -2 ile 2 arasını veya daha geniş bir aralığı kullanabiliriz
            low=-2.0, high=2.0, shape=(state_size,), dtype=np.float32
        )


        # Character and obstacle velocities
        self.character_velocity = 0.15
        self.obstacle_velocity = 0.12

        self._max_obstacle_vel_norm = self.obstacle_velocity * 1.1

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


        # For rendering
        self.window = None
        self.clock = None
        self.render_mode = render_mode

        # Maximum episode length
        self.max_steps = 500
        self.current_step = 0



    def _get_nearest_obstacle_distance(self):
        if self.num_obstacles == 0:
            return float('inf')
        obstacle_distances = [
            np.linalg.norm(self._character_position - self._obstacle_positions[i]) - self.obstacle_size # Adjust for obstacle radius
            for i in range(self.num_obstacles)
        ]
        return min(obstacle_distances) if obstacle_distances else float('inf')

    def _get_obs(self):
        # 1. Relative vector to the goal (normalized)
        relative_goal_pos = (self._goal_position - self._character_position) / self.size
        state_list = list(relative_goal_pos)

        # 2. Information of the closest k obstacles
        if self.num_obstacles > 0:
            obstacle_distances = np.linalg.norm(self._obstacle_positions - self._character_position, axis=1)
            # Sort obstacle indices by distance
            closest_indices = np.argsort(obstacle_distances)[:self.k_closest]

            num_found_obstacles = len(closest_indices)

            for i in range(num_found_obstacles):
                idx = closest_indices[i]
                # Relative vector to the obstacle (normalized)
                relative_obs_pos = (self._obstacle_positions[idx] - self._character_position) / self.size
                # Obstacle velocity (normalized)
                obs_velocity = self._obstacle_velocities[idx] / self._max_obstacle_vel_norm  # Normalize velocity

                state_list.extend(relative_obs_pos)
                state_list.extend(obs_velocity)

            # Padding: If fewer than k obstacles are found, fill the remaining slots
            # Fill with distant (~1 normalized) and stationary (0) obstacle data
            padding_needed = self.k_closest - num_found_obstacles
            for _ in range(padding_needed):
                state_list.extend([1.0, 1.0])  # Distant relative position (normalized)
                state_list.extend([0.0, 0.0])  # Zero velocity

        else:  # If there are no obstacles at all, pad all k obstacle slots
            for _ in range(self.k_closest):
                state_list.extend([1.0, 1.0])
                state_list.extend([0.0, 0.0])

        return np.array(state_list, dtype=np.float32)

    def _get_info(self):
        dist_to_goal = np.linalg.norm(self._character_position - self._goal_position)
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
        # Place character at a random position
        self._character_position = self.np_random.uniform(
            low=np.array([0.1, 0.1]), high=np.array([self.size - 0.1, self.size - 0.1]), size=2
        )

        # Place goal ensuring minimum distance from character
        while True:
            self._goal_position = self.np_random.uniform(
                low=np.array([0.1, 0.1]),
                high=np.array([self.size - 0.1, self.size - 0.1]),
                size=2
            )
            if np.linalg.norm(self._goal_position - self._character_position) > 1.5: # Start distance check
                break

        # Initialize obstacles
        self._obstacle_positions = []
        self._obstacle_velocities = []

        for _ in range(self.num_obstacles):
            while True:
                position = self.np_random.uniform(low=0.1, high=self.size - 0.1, size=2)
                char_dist = np.linalg.norm(position - self._character_position)
                goal_dist = np.linalg.norm(position - self._goal_position)
                # Ensure obstacle isn't too close to character or goal initially
                if char_dist > (self.character_size + self.obstacle_size + 0.5) and \
                   goal_dist > (self.goal_size + self.obstacle_size + 0.5):
                    # Check distance from already placed obstacles to avoid initial overlap
                    valid_pos = True
                    for existing_pos in self._obstacle_positions:
                        if np.linalg.norm(position - existing_pos) < 2 * self.obstacle_size + 0.1:
                            valid_pos = False
                            break
                    if valid_pos:
                        break

            self._obstacle_positions.append(position)

            # Random velocity
            angle = self.np_random.uniform(0, 2 * math.pi)
            vx = self.obstacle_velocity * math.cos(angle)
            vy = self.obstacle_velocity * math.sin(angle)
            self._obstacle_velocities.append(np.array([vx, vy]))

        self._obstacle_positions = np.array(self._obstacle_positions) if self.num_obstacles > 0 else np.empty((0, 2))
        self._obstacle_velocities = np.array(self._obstacle_velocities) if self.num_obstacles > 0 else np.empty((0, 2))

        # Initial observation directly from _get_obs
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.current_step += 1
        first_distance = np.linalg.norm(self._character_position - self._goal_position)

        # Move character based on action
        move_vector = np.array([0.0, 0.0])
        if action == 0:  # Left
             move_vector[0] -= self.character_velocity
        elif action == 1:  # Right
             move_vector[0] += self.character_velocity
        elif action == 2:  # Up 
             move_vector[1] -= self.character_velocity
        elif action == 3:  # Down
             move_vector[1] += self.character_velocity
        # Action 4 is "do nothing"

        new_position = self._character_position + move_vector
        # Clip position to stay within bounds
        self._character_position = np.clip(new_position, 0, self.size)


        # Move obstacles and handle bounces
        if self.num_obstacles > 0:
            self._obstacle_positions += self._obstacle_velocities

            # Bounce off walls
            # Find obstacles hitting horizontal walls (left/right)
            hit_lr = (self._obstacle_positions[:, 0] <= self.obstacle_size) | (self._obstacle_positions[:, 0] >= self.size - self.obstacle_size)
            self._obstacle_velocities[hit_lr, 0] *= -1

            # Find obstacles hitting vertical walls (top/bottom)
            hit_tb = (self._obstacle_positions[:, 1] <= self.obstacle_size) | (self._obstacle_positions[:, 1] >= self.size - self.obstacle_size)
            self._obstacle_velocities[hit_tb, 1] *= -1

            # Clip positions to prevent going out of bounds after bounce adjustment
            self._obstacle_positions[:, 0] = np.clip(self._obstacle_positions[:, 0], self.obstacle_size, self.size - self.obstacle_size)
            self._obstacle_positions[:, 1] = np.clip(self._obstacle_positions[:, 1], self.obstacle_size, self.size - self.obstacle_size)


        # Check for collisions with obstacles
        collision = False
        if self.num_obstacles > 0:
            distances = np.linalg.norm(self._character_position - self._obstacle_positions, axis=1)
            if np.any(distances < (self.character_size + self.obstacle_size)):
                collision = True

        # Check for reaching the goal
        goal_reached = np.linalg.norm(self._character_position - self._goal_position) < (self.character_size + self.goal_size)

        # Calculate reward
        reward = -0.01 # Small penalty for existing per step
        terminated = False

        if collision:
            reward = -20.0 # Larger penalty for collision
            terminated = True
        elif goal_reached:
            reward = 50.0 # Large reward for reaching goal
            self.score += 1
            # Optionally terminate, or place new target for continuous task
            # terminated = True
            self._place_target() # Place new target to continue episode
        else:
            # Reward shaping: Encourage getting closer to the goal
            dist_to_goal = np.linalg.norm(self._character_position - self._goal_position)
            # Reward proportional to distance reduction
            reward += (first_distance - dist_to_goal) * 0.5 # Positive if closer, negative if farther


        # Check for timeout
        if self.current_step >= self.max_steps:
            terminated = True # Episode ends due to timeout

        # Gymnasium uses 'terminated' for end states (collision, goal) and 'truncated' for timeouts
        truncated = False
        if not terminated and self.current_step >= self.max_steps:
             truncated = True
             terminated = False # If truncated, it's not technically terminated by failure/success


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
        canvas.fill((0, 0, 0)) # Black background

        # Conversion from grid coordinates to pixel coordinates
        def coord_to_pixel(coord):
            # Adjust y-coordinate if Pygame's y=0 is top, but your grid's y=0 is bottom
            # pixel_x = int(coord[0] / self.size * self.window_size)
            # pixel_y = int((1 - coord[1] / self.size) * self.window_size) # Invert Y if needed
            # Assuming y=0 is top for both:
            pixel_x = int(coord[0] / self.size * self.window_size)
            pixel_y = int(coord[1] / self.size * self.window_size)
            return pixel_x, pixel_y

        def size_to_pixel(radius):
             return int(radius / self.size * self.window_size)

        # Draw goal (Yellow Circle)
        pygame.draw.circle(
            canvas,
            (255, 255, 0),
            coord_to_pixel(self._goal_position),
            size_to_pixel(self.goal_size),
        )

        # Draw character (Green Square)
        char_pix_pos = coord_to_pixel(self._character_position)
        char_pix_size = size_to_pixel(self.character_size) * 2 # Make square visually distinct
        character_rect = pygame.Rect(
            char_pix_pos[0] - char_pix_size // 2,
            char_pix_pos[1] - char_pix_size // 2,
            char_pix_size,
            char_pix_size
        )
        pygame.draw.rect(
            canvas,
            (0, 255, 0), 
            character_rect
        )

        # Draw obstacles (Red Circles)
        for i in range(self.num_obstacles):
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                coord_to_pixel(self._obstacle_positions[i]),
                size_to_pixel(self.obstacle_size),
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
            canvas.blit(goal_text, (10, 30))
            # canvas.blit(obstacle_text, (10, 50))
            canvas.blit(score_text, (10, 70))

            # Copy our drawings to the window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Ensure fixed framerate
            self.clock.tick(self.metadata["render_fps"])

        # Return pixels as numpy array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )


    def _place_target(self):
        # Hedefi karakterden uzağa yerleştir
        while True:
            self._goal_position = self.np_random.uniform(
                low=np.array([0.1, 0.1]),
                high=np.array([self.size - 0.1, self.size - 0.1]),
                size=2
            )
            # Ensure minimum distance from character and not too close to obstacles
            if np.linalg.norm(self._goal_position - self._character_position) > 1.5:
                valid_pos = True
                if self.num_obstacles > 0:
                     obstacle_distances = np.linalg.norm(self._goal_position - self._obstacle_positions, axis=1)
                     if np.any(obstacle_distances < (self.goal_size + self.obstacle_size + 0.3)):
                          valid_pos = False
                if valid_pos:
                    break


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None # Reset window status
            self.clock = None # Reset clock status