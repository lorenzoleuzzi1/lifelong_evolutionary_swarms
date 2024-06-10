import numpy as np

import gymnasium as gym
from gymnasium import spaces

from utils import visual_grid_to_image

TIME_STEP = 0.1 # a simulation step in seconds
ROBOT_SIZE = 25 # in cm (diameter)
SENSOR_RANGE = 100 # in cm
MAX_WHEEL_VELOCITY = 50 # in cm/s, max is 200 cm/s but in the paper they use 50 cm/s for exploration
ARENA_SIZE = 500 # in cm

SIMULATION_ROBOT_SIZE = ROBOT_SIZE / ROBOT_SIZE 
SIMULATION_SENSOR_RANGE = SENSOR_RANGE / ROBOT_SIZE 
# SIMULATION_MAX_DISTANCE = MAX_DISTANCE / ROBOT_SIZE 
SIMULATION_MAX_WHEEL_VELOCITY = MAX_WHEEL_VELOCITY / ROBOT_SIZE
SIMULATION_ARENA_SIZE = ARENA_SIZE / ROBOT_SIZE 

UP = 0 
LEFT = 90
DOWN = 180
RIGHT = 270

NOTHING = 0
WALL = 1
AGENT = 2
RED = 3
BLUE = 4
GREEN = 5
YELLOW = 6
PURPLE = 7

REWARD_PICK = 1
REWARD_DROP = 2
REWARD_COLLISION = -1

# Kinematic matrix from the transformation
A = np.array([[-0.28867513, -0.57735027, 0.8660254],[ 0.5, -1, 0.5],[0.5, 0, 0.5]])

# v1, v2, v3 values (three omniwheels)
MOVE_UP =  [0.8660254, 0, -0.8660254]
MOVE_RIGHT =  [0.5, -1, -0.5]
MOVE_UP_RIGHT = [1.3660254, -1, 1.3660254]
MOVE_DOWN_RIGHT = [-0.3660254, -1., 0.3660254]
MOVE_DOWN = [-0.8660254, 0., 0.8660254]
MOVE_DOWN_LEFT = [-1.3660254, 1., 1.3660254]
MOVE_LEFT = [-0.5, 1., 0.5]
MOVE_UP_LEFT = [0.3660254, 1. -0.3660254]
ROTATE_POSITIVE = [1.57079633, 1.57079633, 1.57079633]
ROTATE_NEGATIVE = [3.14159265, 3.14159265, 3.14159265]

# TODO: check for more speed optimization in the code

class SwarmForagingEnv(gym.Env):

    def __init__(
            self, 
            target_color = RED,
            size = SIMULATION_ARENA_SIZE, 
            n_agents = 3, 
            n_blocks = 10,
            rate_objective_block = 0.5, 
            n_neighbors = 3,
            sensor_range = SIMULATION_SENSOR_RANGE,
            max_wheel_velocity = SIMULATION_MAX_WHEEL_VELOCITY,
            sensitivity = 0.5, # How close the agent can get to the block to pick it up 
            time_step = TIME_STEP, # in seconds
            duration = 500, # max number of steps for an episode
            max_retrives = 20, # max number of retrives for an episode
            seed = 0,
            ):
        
        self.nest = UP # The nest location (UP, DOWN, LEFT, RIGHT) TODO: maybe as parameter?
        self.drop_zone = UP # The drop zone location (UP, DOWN, LEFT, RIGHT) TODO: maybe as parameter?
        self.target_color = target_color  # The objective block TODO: the drop zone is always the same?
        self._retrieved = []
        
        self.n_agents = n_agents
        self.n_blocks = n_blocks
        
        self.seed = seed
        np.random.seed(seed=seed)
        self.size = size  # The size of the square grid
        self.time_step = time_step
        
        self.agents_location = np.zeros((self.n_agents, 2), dtype=float)
        self._agents_carrying = np.full(self.n_agents, -1, dtype=int)
        self._agents_closest_objective_distance = np.full(self.n_agents, -1, dtype=float)
        self.agents_heading = np.zeros(self.n_agents, dtype=float)

        self.blocks_location = np.zeros((self.n_blocks, 2), dtype=float)
        self.blocks_color = np.zeros(self.n_blocks, dtype=int)
        self._blocks_picked_up = np.full(self.n_blocks, -1, dtype=int)
        self._blocks_initial_distance_to_dropzone = np.full(self.n_blocks, -1, dtype=float)
        self.rate_objective_block = rate_objective_block

        self._initial_setting = None

        self._distance_matrix_agent_agent = np.zeros((self.n_agents, self.n_blocks), dtype=float)
        self._direction_matrix_agent_agent = np.zeros((self.n_agents, self.n_blocks), dtype=float)
        self._distance_matrix_agent_agent = np.zeros((self.n_agents, 4), dtype=float)
        self._direction_matrix_agent_block = np.zeros((self.n_agents, self.n_blocks), dtype=float)
        
        self.sensitivity = sensitivity # How close to interact
        self.n_neighbors = n_neighbors
        self._neighbors = np.zeros((self.n_agents, n_neighbors, 3), dtype=float) # init sensors
        self._previous_neighbors = np.zeros((self.n_agents, n_neighbors, 3), dtype=float) 
        self.sensor_range = sensor_range
        self.sensor_angle = 360
        self.max_wheel_velocity = max_wheel_velocity

        self._rewards = np.zeros(self.n_agents, dtype=int)

        self.duration = duration
        self.max_retrives = max_retrives
        self.current_step = 0

        self._colors_map = {
            RED: "\033[91m",  # Red
            BLUE: "\033[94m",  # Blue
            GREEN: "\033[92m",  # Green
            YELLOW: "\033[93m",  # Yellow
            PURPLE: "\033[95m",  # Purple
        }
        self._reset_color = "\033[0m"  # Resets color to default
        self.n_types = len(self._colors_map) + 1 + 1 + 1 # colors, robot, edge, nothing
        
        # Action space
        single_action_space = spaces.Box(low=np.array([-max_wheel_velocity, -max_wheel_velocity, -max_wheel_velocity]), 
                                         high=np.array([max_wheel_velocity, max_wheel_velocity, max_wheel_velocity]), dtype=float)
        
        self.action_space = spaces.Tuple([single_action_space for _ in range(self.n_agents)])
        
        # Observation space
        single_observation_space = spaces.Dict(
            {
                "neighbors": spaces.Box(
                                low=np.zeros((n_neighbors, 3), dtype=float),
                                high=np.array([[self.n_types, sensor_range, self.sensor_angle]] * n_neighbors),
                                dtype=float
                            ),
                "carrying": spaces.Box(-1, 9, shape=(1,), dtype=int)
            }
        )
        self.observation_space = spaces.Tuple([single_observation_space for _ in range(self.n_agents)])
    
    def _reposition_block(self, j):
        self.blocks_location[j] = np.random.randint((5, 1), 
                                                    (SIMULATION_ARENA_SIZE - 2, SIMULATION_ARENA_SIZE - 2), 
                                                    2)
    
    def create_initial_setting(self):
        # Blocks
        # n_target_blocks = int(self.n_blocks * self.rate_objective_block)
        # n_other_blocks = self.n_blocks - n_target_blocks
        # # TODO: if self.nest is not necessary UP change the code below
        # target_blocks_location = np.random.randint((5, 1), 
        #                                             (SIMULATION_ARENA_SIZE - 2, SIMULATION_ARENA_SIZE - 2), 
        #                                             (n_target_blocks, 2))
        # objective_blocks_color = np.full(n_target_blocks, self.target_color)
        
        # other_blocks_location = np.random.randint((5, 1), 
        #                                         (SIMULATION_ARENA_SIZE - 2, SIMULATION_ARENA_SIZE - 2),
        #                                         (n_other_blocks, 2))
        # other_blocks_color = []
        # for i in range(n_other_blocks):
        #     other_blocks_color.append(np.random.randint(4, 7))
        #     while other_blocks_color[i] == self.target_color:
        #         other_blocks_color[i] = np.random.randint(4, 7)
        # initial_blocks = np.concatenate((target_blocks_location, other_blocks_location), axis=0)
        # initial_colors = np.concatenate((objective_blocks_color, other_blocks_color), axis=0)
        # Blocks
        blocks_locations = np.zeros((self.n_blocks, 2), dtype=float)
        blocks_colors = np.zeros(self.n_blocks, dtype=int)
        for i in range(self.n_blocks):
            low = (5, 1)
            high = (SIMULATION_ARENA_SIZE - 2, SIMULATION_ARENA_SIZE - 2)
            blocks_colors[i] = (i % 5) + RED # 5 colors
            while True:
                blocks_locations[i] = np.random.randint(low, high, 2)
                # Check if the new position is valid (not occupied by another block)
                if i == 0 or not np.any(np.linalg.norm(blocks_locations[i] - blocks_locations[:i], axis=1) < 1):
                    break
        # Agents
        agents_locations = np.zeros((self.n_agents, 2), dtype=float)
        for i in range(self.n_agents):
            while True:
                low = (0, 0)
                high = (3, SIMULATION_ARENA_SIZE)
                heading = DOWN
                agents_locations[i] = np.random.randint(low, high, 2)
                # Check if the new position is valid (not occupied by another agent)
                if i == 0 or not np.any(np.linalg.norm(agents_locations[i] - agents_locations[:i], axis=1) < 1):
                    break
            
        self._initial_setting = {
            'agents': np.array(agents_locations, dtype=float),
            'headings': np.full(self.n_agents, heading, dtype=float),
            'blocks': np.array(blocks_locations, dtype=float),
            'colors': np.array(blocks_colors, dtype=int),
            }
    
    def _is_close_to_edge(self, agent_position):
        if agent_position[0] < 1 or agent_position[0] > self.size - 2:
            return True
        if agent_position[1] < 1 or agent_position[1] > self.size - 2:
            return True
        return False

    def _is_correct_drop(self, agent_position, block_idx):
        return agent_position[0] < 1 and self.blocks_color[block_idx] == self.target_color

    def _update_directions_matrix(self):
        # Agents-Blocks directions matrix
        dx_blocks = self.agents_location[:, np.newaxis, 0] - self.blocks_location[:, 0] 
        dy_blocks = self.agents_location[:, np.newaxis, 1] - self.blocks_location[:, 1] 
        angles = np.degrees(np.arctan2(dy_blocks, dx_blocks))
        angles = np.mod(np.add(angles, 360), 360)
        self._direction_matrix_agent_block = angles

        # Agents-Agents directions matrix
        dx_agents = self.agents_location[:, np.newaxis, 0] - self.agents_location[:, 0]  
        dy_agents = self.agents_location[:, np.newaxis, 1] - self.agents_location[:, 1]
        angles = np.degrees(np.arctan2(dy_agents, dx_agents))
        angles = np.mod(np.add(angles, 360), 360)
        self._direction_matrix_agent_agent = angles

    def _update_distance_matrix(self):
        # Agents-Blocks distance matrix
        diff_matrix_blocks = self.agents_location[:, np.newaxis, :] - self.blocks_location 
        self._distance_matrix_agent_block = np.linalg.norm(diff_matrix_blocks, axis=-1)

        # Agents-Agents distance matrix
        diff_matrix_agents = self.agents_location[:, np.newaxis, :] - self.agents_location
        self._distance_matrix_agent_agent = np.linalg.norm(diff_matrix_agents, axis=-1)

    def _detect(self):
        # Mimic sensors reading
        
        for i in range(self.n_agents):
            neighbors = []
            
            # Check if sensors detect the edge of the arena
            if self.agents_location[i][0] < self.sensor_range: # Top edge
                neighbors.append([1, self.agents_location[i][0], UP]) 
            if self.size - self.agents_location[i][0] - 1 < self.sensor_range: # Bottom edge
                neighbors.append([1, self.size - self.agents_location[i][0] - 1, DOWN])
            if self.agents_location[i][1] < self.sensor_range: # Left edge
                neighbors.append([1, self.agents_location[i][1], LEFT])
            if self.size - self.agents_location[i][1] - 1 < self.sensor_range: # Right edge
                neighbors.append([1, self.size - self.agents_location[i][1] - 1, RIGHT])
            
            # Get indexes of agents that are within the sensor range
            neighbors_agents_idx = np.where(self._distance_matrix_agent_agent[i] <= self.sensor_range)[0]
            neighbors_agents_idx = neighbors_agents_idx[neighbors_agents_idx != i] # Remove the i index
            # Get the distances and directions of the agents that are within the sensor range
            distances_agents = self._distance_matrix_agent_agent[i, neighbors_agents_idx]
            directions_agents = self._direction_matrix_agent_agent[i, neighbors_agents_idx]
            # Add the agents that are within the sensor range
            for j in range(len(neighbors_agents_idx)):
                neighbors.append([2, distances_agents[j], directions_agents[j]])

            # Get indexes of blocks that are within the sensor range
            neighbors_blocks_idx = np.where(self._distance_matrix_agent_block[i] <= self.sensor_range)[0]
            # Get the distances and directions of the blocks that are within the sensor range
            distances_blocks = self._distance_matrix_agent_block[i, neighbors_blocks_idx]
            directions_blocks = self._direction_matrix_agent_block[i, neighbors_blocks_idx]
            # Add the blocks that are within the sensor range
            for j in range(len(neighbors_blocks_idx)):
                neighbors.append([self.blocks_color[neighbors_blocks_idx[j]], distances_blocks[j], directions_blocks[j]])

            n_detected_neighbors = len(neighbors)
            neighbors = sorted(neighbors, key=lambda x: x[1])
            # Fill the rest of the neighbors with nothing
            for _ in range(self.n_neighbors - n_detected_neighbors):
                neighbors.append([0, 0, 0])
            
            self._neighbors[i] = neighbors[:self.n_neighbors] # Take only first n_neighbors
    
    def _get_info(self):
        return {"retrieved": self._retrieved} # TODO: potentially add more info
    
    def _get_obs(self):
        obs = []
        for i in range(self.n_agents):
            carrying = self.blocks_color[self._agents_carrying[i]] if self._agents_carrying[i] != -1 else -1
            obs.append({"neighbors" : self._neighbors[i], "heading": self.agents_heading[i], "carrying" : carrying})
        return obs
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)

        self._agents_carrying = np.full(self.n_agents, -1, dtype=int)
        self._blocks_picked_up = np.full(self.n_blocks, -1, dtype=int)
        self._neighbors = np.zeros((self.n_agents, self.n_neighbors, 3), dtype=float)
        self._previous_neighbors = np.zeros((self.n_agents, self.n_neighbors, 3), dtype=float)
        self._agents_closest_objective_distance = np.full(self.n_agents, -1, dtype=float)
        
        if self._initial_setting is None:
            self.create_initial_setting()
            
        self.agents_location = self._initial_setting['agents'].copy()
        self.agents_heading = self._initial_setting['headings'].copy()
        self.blocks_location = self._initial_setting['blocks'].copy()
        self.blocks_color = self._initial_setting['colors'].copy()
                                                                                                       
        self.n_retrives = 0
        self.current_step = 0
        
        self._rewards = np.zeros(self.n_agents)
        self._retrived = []
        info = {}
        self._update_directions_matrix()
        self._update_distance_matrix()
        
        self._detect()
        observations = self._get_obs()

        return observations, info

    def step(self, action):
        
        self._rewards = np.zeros(self.n_agents)
        old_agents_location = self.agents_location.copy()
        
        # --- MOVEMENT ---
        # Move all agents
        wheel_velocities = action # Extract all v1, v2, v3 values
        # Calculate vx, vy, and R_omega for all agents
        velocities = np.dot(wheel_velocities, A.T) # u = Av, checked!
        # Update positions
        x_new = self.agents_location[:, 0] + velocities[:, 0] * self.time_step
        y_new = self.agents_location[:, 1] + velocities[:, 1] * self.time_step
        # Clip within arena
        x_new = np.clip(x_new, 0, self.size - 1)
        y_new = np.clip(y_new, 0, self.size - 1)
        # Update heading
        omega = velocities[:, 2] / (ROBOT_SIZE / 2) # Angular velocity, omega = R_omega / R
        theta_new = np.mod(self.agents_heading + np.degrees(omega * self.time_step), 360)
        # Update internal state
        self.agents_location = np.stack((x_new, y_new), axis=-1)
        self.agents_heading = theta_new
        
        # Hadle collisions
        agents_in_same_position = []
        for i in range(self.n_agents):
            # Get indexes of agents that are in the same position
            colliding = np.where(np.linalg.norm(self.agents_location - self.agents_location[i], axis=1) < self.sensitivity)[0]
            colliding = colliding[colliding != i]
            if colliding.size > 0:
                agents_in_same_position.append(i)
                agents_in_same_position.extend(colliding)
        if len(agents_in_same_position) > 0:
            # self._rewards[agents_in_same_position] += REWARD_COLLISION # Penalize collsion
            agents_in_same_position = list(set(agents_in_same_position))
            # If the new position is occupied, keep the old position TODO: check it and maybe change it
            self.agents_location[agents_in_same_position] = old_agents_location[agents_in_same_position]
        # ----------------
        
        self._update_directions_matrix()
        self._update_distance_matrix()

        for i in range(self.n_agents):
            # --- PICK ---
            # Check if the agent is picking up a block
            if self._agents_carrying[i] == -1: # If the agent is not carrying a block
                # Get closest block from the agent
                closest_block_idx = np.argmin(self._distance_matrix_agent_block[i])
                distance_to_closest_block = self._distance_matrix_agent_block[i][closest_block_idx]
                if distance_to_closest_block < self.sensitivity:
                    # Reward the agent for picking up the block
                    if self.blocks_color[closest_block_idx] == self.target_color:
                        self._rewards[i] += REWARD_PICK
                    else:
                        self._rewards[i] -= REWARD_PICK
                        
                    # Pick the block
                    self.blocks_location[closest_block_idx] = [np.inf, np.inf] # Not in the arena 
                    self._blocks_picked_up[closest_block_idx] = i
                    self._agents_carrying[i] = closest_block_idx
                    self._distance_matrix_agent_block[:, closest_block_idx] = np.inf
            # ------------

            # --- DROP ---
            # Check if the agent is dropping a block
            if self._agents_carrying[i] != -1: # If the agent is carrying a block
                # If the agent is close to an edge
                if self._is_close_to_edge(self.agents_location[i]):
                    # Reward the agent for dropping the block
                    if self._is_correct_drop(self.agents_location[i], self._agents_carrying[i]):
                        self._retrieved.append((self.current_step, i, self.blocks_color[self._agents_carrying[i]],
                                                self._agents_carrying[i]))
                        self._rewards[i] += REWARD_DROP
                    else:
                        self._rewards[i] -= REWARD_DROP
                    
                    # Reset block and place it back in the arena
                    self._reposition_block(self._agents_carrying[i]) 
                    self._blocks_picked_up[self._agents_carrying[i]] = -1
                    self._agents_carrying[i] = -1

                    self._agents_closest_objective_distance[i] = -1
            # ------------
                
        self._detect()
        observations = self._get_obs()

        reward = sum(self._rewards)  # Sum the rewards of all agents of the swarm
        
        # Check termination
        done = False
        if len(self._retrieved) >= self.max_retrives: 
            done = True
        truncated = False
        if self.current_step >= self.duration: 
            truncated = True
        
        info = self._get_info()
        
        self.current_step += 1
        
        return observations, reward, done, truncated, info

    def render(self, verbose = True):
        # Define the size of the visualization grid
        vis_grid_size = 20  # Adjust based on desired resolution

        # Create an empty visual representation of the environment
        visual_grid = [["." for _ in range(vis_grid_size)] for _ in range(vis_grid_size)]
        
        # Populate the visual grid with blocks
        for i, block in enumerate(self.blocks_location):
            # Convert continuous coordinates to discrete grid positions
            if block[0] != np.inf and block[1] != np.inf:
                x = int(round(block[0] * (vis_grid_size - 1) / (self.size - 1), 0))
                y = int(round(block[1] * (vis_grid_size - 1) / (self.size - 1), 0))
                if 0 <= x < vis_grid_size and 0 <= y < vis_grid_size:
                    color_id = self.blocks_color[i]
                    color_code = self._colors_map.get(color_id, self._reset_color)
                    visual_grid[x][y] = f"{color_code}O{self._reset_color}"
        
        # Populate the visual grid with agents
        for i, agent in enumerate(self.agents_location):
            # Convert continuous coordinates to discrete grid positions
            x = int(round(agent[0] * (vis_grid_size - 1) / (self.size - 1), 0))
            y = int(round(agent[1] * (vis_grid_size - 1) / (self.size - 1), 0))
            if 0 <= x < vis_grid_size and 0 <= y < vis_grid_size:
                if self._agents_carrying[i] != -1:
                    color_id = self.blocks_color[self._agents_carrying[i]]
                    color_code = self._colors_map.get(color_id, self._reset_color)
                    visual_grid[x][y] = f"{color_code}{i}{self._reset_color}"
                else:
                    visual_grid[x][y] = str(i)
        
        # Print the visual representation
        if verbose:
            for row in visual_grid:
                print(" ".join(row))
                      
        return visual_grid_to_image(visual_grid)
        
    def print_observations(self, verbose = True):
        observations_text = ""
        for i in range(self.n_agents):
            flag = False
            if self._agents_carrying[i] != -1:
                observations_text += f"Agent {i} is carrying a block (color: {self._agents_carrying[i]}). "
            else:
                observations_text += f"Agent {i} is not carrying anything. "
            for j in range(self.n_neighbors):
                if self._neighbors[i,j,0] != 0:
                    if self._neighbors[i,j,0] == WALL: entity = "wall"
                    if self._neighbors[i,j,0] == AGENT: entity = "agent"
                    if self._neighbors[i,j,0] > AGENT: entity = f"block (color: {self._neighbors[i,j,0]})"
                    
                    distance = self._neighbors[i,j,1]
                    direction = self._neighbors[i,j,2]
                    observations_text += f"Agent {i} sees {entity}: {distance} distance and {direction} degrees direction. "
                    flag = True
            if not flag:
                observations_text += f"Agent {i} doesn't see anything."
            observations_text += "\n"
        if verbose:
            print(observations_text)
        
        return observations_text
    
    def process_observation(self, obs):
        # Create structured arrays for batch processing
        neighbors = np.array([agent['neighbors'] for agent in obs])
        heading = np.array([agent['heading'] for agent in obs])
        carrying = np.array([agent['carrying'] for agent in obs])
        
        # One-hot encode types
        types = np.eye(self.n_types)[neighbors[:, :, 0].astype(int)]

        # Normalize distances and directions
        distances = neighbors[:, :, 1] / self.sensor_range 
        directions_sin = np.sin(np.radians(neighbors[:, :, 2]))
        directions_cos = np.cos(np.radians(neighbors[:, :, 2]))

        # heading = heading / self.sensor_angle
        heading_sin = np.sin(np.radians(heading))
        heading_cos = np.cos(np.radians(heading))

        # One-hot encode carrying status
        # Assuming carrying values range from -1 (not carrying) to max_carrying_id
        carrying[carrying == -1] = 0 # Change -1 to 0
        carrying[carrying > 0] = carrying[carrying > 0] - 2 # Change 3, 4, 5, ... to 1, 2, 3, ...
        carrying_one_hot = np.eye(self.n_types - 2)[carrying]

        # Flatten all features and concatenate them into a single vector per agent
        flat_features = np.concatenate([
            types.reshape(types.shape[0], -1),  # Flatten types
            distances.reshape(distances.shape[0], -1),  # Flatten distances
            directions_sin.reshape(directions_sin.shape[0], -1),  # Flatten directions
            directions_cos.reshape(directions_cos.shape[0], -1),  # Flatten directions
            heading_sin.reshape(heading_sin.shape[0], -1),  # Flatten heading
            heading_cos.reshape(heading_cos.shape[0], -1),  # Flatten heading
            carrying_one_hot  # Already appropriate shape
        ], axis=1) # TODO: is the order important?

        return flat_features
        
    def close(self):
        pass