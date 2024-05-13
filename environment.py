import numpy as np

import gymnasium as gym
from gymnasium import spaces

TIME_PER_STEP = 0.2 # a step in seconds, 200 ms, typical is 100ms
ROBOT_SIZE = 25 # in cm (diameter)
SENSOR_RANGE = 100 # in cm
MAX_WHEEL_VELOCITY = 100 # in cm/s, in the papers was 200 cm/s but it's fine
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
ORANGE = 8
GREY = 9

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

# TODO: mind the collision... now it's checked only the position where they are getting
# TODO: make change direction when reaching an edge, they should learn it via evolution
# TODO: check for more speed optimization in the code

class Environment(gym.Env):

    def __init__(
            self, 
            nest = UP,
            objective = [(RED, UP)], # objective is a list of tuples (color, edge) # TODO: maybe only 1 onjective
            seed=None,
            size=100, 
            n_agents=3, 
            n_blocks=3, 
            n_neighbors = 4,
            sensor_range = 2,
            sensor_angle = 360,
            max_wheel_velocity = 1,
            sensitivity = 0.2, # How close the agent can get to the block to pick it up 
            initial_setting = None
            ):
        # TODO: if passing initial setting, no need to n_agents, n_blocks
        self.nest = nest  # The nest location
        self.objective = objective
        self._objective_colors = [obj[0] for obj in objective]
        self._task_completed = []
        
        self.seed = seed
        self.size = size  # The size of the square grid
        
        self.n_agents = n_agents
        self.agents_location = np.zeros((self.n_agents, 2), dtype=float)
        self._agents_carrying = np.full(self.n_agents, -1, dtype=int)
        self._agents_closest_objective_distance = np.full(self.n_agents, -1, dtype=float)
        self.agents_heading = np.zeros(self.n_agents, dtype=float)

        self.n_blocks = n_blocks
        self.blocks_location = np.zeros((self.n_blocks, 2), dtype=float)
        self.blocks_color = np.zeros(self.n_blocks, dtype=int)
        self._blocks_picked_up = np.full(self.n_blocks, -1, dtype=int)
        self._blocks_initial_distance_to_dropzone = np.full(self.n_blocks, -1, dtype=float)

        self._distance_matrix_agent_agent = np.zeros((self.n_agents, self.n_blocks), dtype=float)
        self._direction_matrix_agent_agent = np.zeros((self.n_agents, self.n_blocks), dtype=float)
        self._distance_matrix_agent_agent = np.zeros((self.n_agents, 4), dtype=float)
        self._direction_matrix_agent_block = np.zeros((self.n_agents, self.n_blocks), dtype=float)
        
        self.sensitivity = sensitivity # How close to interact
        self.n_neighbors = n_neighbors
        self._neighbors = np.zeros((self.n_agents, n_neighbors, 3), dtype=float) # init sensors
        self._previous_neighbors = np.zeros((self.n_agents, n_neighbors, 3), dtype=float) 
        self.sensor_range = sensor_range
        self.sensor_angle = sensor_angle
        self.max_wheel_velocity = max_wheel_velocity

        self._rewards = np.zeros(self.n_agents)

        self._initial_setting = initial_setting

        self._colors_map = {
            RED: "\033[91m",  # Red
            BLUE: "\033[94m",  # Blue
            GREEN: "\033[92m",  # Green
            YELLOW: "\033[93m",  # Yellow
            PURPLE: "\033[95m",  # Purple
            ORANGE: "\033[33m",   # Orange
            GREY: "\033[90m",   # Dark Gray
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
                                high=np.array([[self.n_types, sensor_range, sensor_angle]] * n_neighbors),
                                dtype=float
                            ),
                "carrying": spaces.Box(-1, 9, shape=(1,), dtype=int)
            }
        )
        self.observation_space = spaces.Tuple([single_observation_space for _ in range(self.n_agents)])

        if initial_setting is not None:
            assert len(initial_setting['agents']) == n_agents, f"Number of agents in initial setting ({len(initial_setting['agents'])}) is different from the number of agents ({n_agents})"
            assert len(initial_setting['blocks']) == n_blocks, f"Number of blocks in initial setting ({len(initial_setting['blocks'])}) is different from the number of blocks ({n_blocks})"
            assert len(initial_setting['colors']) == n_blocks, f"Number of colors in initial setting ({len(initial_setting['colors'])}) is different from the number of blocks ({n_blocks})"
            
            # Check if agents are not spawning in the same location
            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j:
                        assert not np.all(initial_setting['agents'][i] == initial_setting['agents'][j]), f"Agent {i} and agent {j} are spawning in the same location"
            # Check if agents and blocks are not spawning in the same location
            for i in range(n_agents):
                for j in range(n_blocks):
                    assert not np.all(initial_setting['agents'][i] == initial_setting['blocks'][j]), f"Agent {i} and block {j} are spawning in the same location"
            
            # Check if blocks are not spawning in the same location
            for i in range(n_blocks):
                for j in range(n_blocks):
                    if i != j:
                        assert not np.all(initial_setting['blocks'][i] == initial_setting['blocks'][j]), f"Block {i} and block {j} are spawning in the same location"
            
            # Check if agents and blocks are within the arena
            for agent in initial_setting['agents']:
                assert agent[0] >= 0 and agent[0] < size, f"Agent {agent} is outside the arena"
                assert agent[1] >= 0 and agent[1] < size, f"Agent {agent} is outside the arena"
            for block in initial_setting['blocks']:
                assert block[0] >= 0 and block[0] < size, f"Block {block} is outside the arena"
                assert block[1] >= 0 and block[1] < size, f"Block {block} is outside the arena"
            
            # All the color objective must be in the blocks colors
            for color, _ in objective:
                assert color in initial_setting['colors'], f"Color {color} in objective is not present in the arena"
    
    def _update_directions_matrix(self):
        # Blocks directions matrix
        dx_blocks = self.agents_location[:, np.newaxis, 0] - self.blocks_location[:, 0] 
        dy_blocks = self.agents_location[:, np.newaxis, 1] - self.blocks_location[:, 1] 
        angles = np.degrees(np.arctan2(dy_blocks, dx_blocks))
        angles = np.mod(np.add(angles, 360), 360)
        self._direction_matrix_agent_block = angles

        # Agents directions matrix
        dx_agents = self.agents_location[:, np.newaxis, 0] - self.agents_location[:, 0]  
        dy_agents = self.agents_location[:, np.newaxis, 1] - self.agents_location[:, 1]
        angles = np.degrees(np.arctan2(dy_agents, dx_agents))
        angles = np.mod(np.add(angles, 360), 360)
        self._direction_matrix_agent_agent = angles

    def _update_distance_matrix(self):
        # Blocks distance matrix
        diff_matrix_blocks = self.agents_location[:, np.newaxis, :] - self.blocks_location 
        self._distance_matrix_agent_block = np.linalg.norm(diff_matrix_blocks, axis=-1)

        # Agents distance matrix
        diff_matrix_agents = self.agents_location[:, np.newaxis, :] - self.agents_location
        self._distance_matrix_agent_agent = np.linalg.norm(diff_matrix_agents, axis=-1)

    def _detect(self, i):
        # Mimic sensors reading
        
        neighbors = []
        
        # TODO: ensure that the sensors only detect one agent per direction (the closest one)
        # Check if sensors detect the edge of the arena
        # TODO: check degrees consistency
        if self.agents_location[i][0] < self.sensor_range: # Top edge
            neighbors.append([1, self.agents_location[i][0], UP]) 
        if self.size - self.agents_location[i][0] - 1 < self.sensor_range: # Bottom edge
            neighbors.append([1, self.size - self.agents_location[i][0] - 1, DOWN])
        if self.agents_location[i][1] < self.sensor_range: # Left edge
            neighbors.append([1, self.agents_location[i][1], LEFT])
        if self.size - self.agents_location[i][1] - 1 < self.sensor_range: # Right edge
            neighbors.append([1, self.size - self.agents_location[i][1] - 1, RIGHT])
        
        #  Get indexes of agents that are within the sensor range
        neighbors_agents_idx = np.where(self._distance_matrix_agent_agent[i] <= self.sensor_range)[0]
        # Remove the i index
        neighbors_agents_idx = neighbors_agents_idx[neighbors_agents_idx != i]
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

    def _get_obs(self, i):
        carrying = self.blocks_color[self._agents_carrying[i]] if self._agents_carrying[i] != -1 else -1
        return {"neighbors" : self._neighbors[i], "heading": self.agents_heading[i], "carrying" : carrying}
    
    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agents_carrying = np.full(self.n_agents, -1, dtype=int)
        self._blocks_picked_up = np.full(self.n_blocks, -1, dtype=int)
        self._neighbors = np.zeros((self.n_agents, self.n_neighbors, 3), dtype=float)
        self._previous_neighbors = np.zeros((self.n_agents, self.n_neighbors, 3), dtype=float)
        self._agents_closest_objective_distance = np.full(self.n_agents, -1, dtype=float)
        
        if self._initial_setting is not None:
            self.agents_location = self._initial_setting['agents'].copy()
            self.agents_heading = self._initial_setting['headings'].copy()
            self.blocks_location = self._initial_setting['blocks'].copy()
            self.blocks_color = self._initial_setting['colors'].copy()
        else:
            # Choose the location at random
            for i in range(self.n_agents):
                # Check if the agents are not spawning in the same location
                while True:
                    # Spawn the agents in the nest
                    if self.nest == UP:
                        self.agents_location[i][0] = 0
                        self.agents_location[i][1] = self.np_random.uniform(0, self.size)
                    elif self.nest == DOWN:
                        self.agents_location[i][0] = self.size - 1
                        self.agents_location[i][1] = self.np_random.uniform(0, self.size)
                    elif self.nest == LEFT:
                        self.agents_location[i][0] = self.np_random.uniform(0, self.size)
                        self.agents_location[i][1] = 0
                    elif self.nest == RIGHT:
                        self.agents_location[i][0] = self.np_random.uniform(0, self.size)
                        self.agents_location[i][1] = self.size - 1
                    if i == 0 or not np.any(np.linalg.norm(self.agents_location[i] - self.agents_location[:i], axis=1) < self.sensitivity):
                        break

            for i in range(self.n_blocks - 1):
                # Check if the blocks are not spawning in the same location
                while True:
                    self.blocks_location[i] = self.np_random.uniform(2, self.size - 2, size=2)
                    if i == 0 or not np.any(np.linalg.norm(self.blocks_location[i] - self.blocks_location[:i], axis=1) < self.sensitivity):
                        break
                self.blocks_color[i] = self.np_random.integers(3, 3 + len(self._colors_map), dtype=int)
            # At least one block is objective
            self.blocks_location[self.n_blocks - 1] = self.np_random.uniform(2, self.size - 2, size=2)
            self.blocks_color[self.n_blocks - 1] = self.objective[0][0]

        
        for i in range(self.n_blocks):
            self._blocks_initial_distance_to_dropzone[i] = self._get_distance_to_objective_edge(self.blocks_location[i], self.blocks_color[i])
        
        self.n_task = 0
        for color, _ in self.objective:
            for i in range(self.n_blocks):
                if self.blocks_color[i] == color:
                    self.n_task += 1
        
        self._rewards = np.zeros(self.n_agents)
        self._task_completed = []
        observations = []
        info = {}
        self._update_directions_matrix()
        self._update_distance_matrix()
        for i in range(self.n_agents):
            self._detect(i)
            observations.append(self._get_obs(i))

        return observations, info
    
    def _is_close(self, a, b):
        return np.linalg.norm(a - b) < self.sensitivity
    
    def _is_agent_close_to_edge(self, agent):
        return agent[0] < self.sensitivity or self.size - agent[0] - 1 < 1 \
                or agent[1] < self.sensitivity or self.size - agent[1] - 1 < 1 # 1 is the size of the drop zone
    
    def _is_correct_edge(self, agent, block_color):
        current_edge = -1
        if agent[0] < self.sensitivity:
            current_edge = UP
        elif self.size - agent[0] - 1 < self.sensitivity:
            current_edge = DOWN
        elif agent[1] < self.sensitivity:
            current_edge = LEFT
        elif self.size - agent[1] - 1 < self.sensitivity:
            current_edge = RIGHT
        
        current_color = block_color
        task = (current_color, current_edge)
        
        if task in self.objective:
            return True
        
        return False

    def _get_distance_to_objective_edge(self, agent, block_color):
        if block_color not in self._objective_colors:
            return -1
        target_edge = self.objective[self._objective_colors.index(block_color)][1]
        if target_edge == UP:
            target_position = np.array([0, agent[1]])
        elif target_edge == DOWN:
            target_position = np.array([self.size - 1, agent[1]])
        elif target_edge == LEFT:
            target_position = np.array([agent[0], 0])
        elif target_edge == RIGHT:
            target_position = np.array([agent[0], self.size - 1])
        
        return np.linalg.norm(agent - target_position)
    
    def _get_bootstrap_reward_pick(self, distance, step_inside_sensor_range = False):
        if step_inside_sensor_range:
            return (distance * (REWARD_PICK / 2)) / self.sensor_range
        return ((self.sensor_range - distance) * (REWARD_PICK / 2)) / self.sensor_range
    
    def _get_bootstrap_reward_drop(self, distance, j):
        return (distance * (REWARD_DROP / 2)) / self._blocks_initial_distance_to_dropzone[j]

    
    def step(self, action):
        
        self._rewards = np.zeros(self.n_agents)
        old_agents_location = self.agents_location.copy()
        
        # --- MOVEMENT ---
        # Extract all v1, v2, v3 values
        wheel_velocities = action
        # Calculate vx, vy, and R_omega for all agents
        velocities = np.dot(wheel_velocities, A.T) # u = Av, checked!
        # Update positions
        x_new = self.agents_location[:, 0] + velocities[:, 0] * TIME_PER_STEP
        y_new = self.agents_location[:, 1] + velocities[:, 1] * TIME_PER_STEP
        # Clip within arena
        x_new = np.clip(x_new, 0, self.size - 1)
        y_new = np.clip(y_new, 0, self.size - 1)
        # Update heading
        omega = velocities[:, 2] / (ROBOT_SIZE / 2) # Angular velocity, omega = R_omega / R
        theta_new = np.mod(self.agents_heading + np.degrees(omega * TIME_PER_STEP), 360) # TODO: check if its radians or degrees (which degrees)
        # Update internal state
        self.agents_location = np.stack((x_new, y_new), axis=-1)
        self.agents_heading = theta_new
        
        # Check if agents are colliding
        agents_in_same_position = []
        for i in range(self.n_agents):
            # Get indexes of agents that are in the same position
            colliding = np.where(np.linalg.norm(self.agents_location - self.agents_location[i], axis=1) < self.sensitivity)[0]
            colliding = colliding[colliding != i]
            if colliding.size > 0:
                agents_in_same_position.append(i)
                agents_in_same_position.extend(colliding)

        if len(agents_in_same_position) > 0:
            self._rewards[agents_in_same_position] += REWARD_COLLISION # Penalize collsion
            agents_in_same_position = list(set(agents_in_same_position))
            # If the new position is occupied, keep the old position TODO: check it
            self.agents_location[agents_in_same_position] = old_agents_location[agents_in_same_position]
        # ----------------
        
        for i in range(self.n_agents):    
            flag_pick = False
            flag_drop = False
            
            # --- PICK ---
            # Check if the agent is picking up a block
            if self._agents_carrying[i] == -1: # If the agent is not carrying a block
                for j in range(self.n_blocks):
                    # If the block is not picked up by any agent and the agent is not carrying a block
                    # and the agent is close to the block
                    if self._is_close(self.agents_location[i], self.blocks_location[j]):
                        flag_pick = True
                        # Reward the agent for picking up the block
                        if self.blocks_color[j] in self._objective_colors:
                            distance = np.linalg.norm(self.agents_location[i] - old_agents_location[i])
                            self._rewards[i] += (REWARD_PICK / 2) + self._get_bootstrap_reward_pick(distance + np.linalg.norm(self.agents_location[i] - self.blocks_location[j]), True)
                            self._agents_closest_objective_distance[i] = self._get_distance_to_objective_edge(self.agents_location[i], self.blocks_color[j])
                        else:
                            self._rewards[i] -= REWARD_PICK
                        
                        # Pick the block
                        self.blocks_location[j] = [-1,-1] 
                        self._blocks_picked_up[j] = i
                        self._agents_carrying[i] = j
                        break
            # ------------

            # --- DROP ---
            if not flag_pick:
                # Check if the agent is dropping a block
                if self._agents_carrying[i] != -1: # If the agent is carrying a block
                    # If the agent is close to an edge
                    if self._is_agent_close_to_edge(self.agents_location[i]):
                        flag_drop = True
                        # Reward the agent for dropping the block
                        if self._is_correct_edge(self.agents_location[i], self.blocks_color[self._agents_carrying[i]]):
                            self._task_completed.append( (self._agents_carrying[i], 
                                                            self.blocks_color[self._agents_carrying[i]],i ))
                            self.blocks_location[self._agents_carrying[i]] = [-1,-1]
                            distance = np.linalg.norm(self.agents_location[i] - old_agents_location[i])
                            self._rewards[i] += (REWARD_DROP / 2) + self._get_bootstrap_reward_drop(distance + self._get_distance_to_objective_edge(self.agents_location[i], self.blocks_color[self._agents_carrying[i]])
                                                                                                    , self._agents_carrying[i])
                        else:
                            self.blocks_location[self._agents_carrying[i]] = self.agents_location[i]
                            self._rewards[i] -= REWARD_DROP
                        
                        # Drop the block
                        self._blocks_picked_up[self._agents_carrying[i]] = -1
                        self._agents_carrying[i] = -1

                        self._agents_closest_objective_distance[i] = -1
                # ------------

                self._detect(i)
                
                if flag_pick or flag_drop:
                    continue
                
                # --- REWARD MOVEMENTS ---
                # Bootstrap the reward system
                # Reward the agent for moving towards an objective block while not carrying anything
                if self._agents_carrying[i] == -1:
                    agent_previous_closest_objective_distance = self._agents_closest_objective_distance[i]
                    for j in range(self.n_neighbors):
                            if self._neighbors[i,j,0] in self._objective_colors:
                                self._agents_closest_objective_distance[i] = self._neighbors[i,j,1]
                                break
                            else:
                                self._agents_closest_objective_distance[i] = -1
                    if (agent_previous_closest_objective_distance == -1 and self._agents_closest_objective_distance[i] != -1):
                        # If the block was outside the sensor range in the previous step and now it is inside
                        # Reward for getting closer
                        self._rewards[i] += self._get_bootstrap_reward_pick(self._agents_closest_objective_distance[i])
                    
                    if (agent_previous_closest_objective_distance != -1 and self._agents_closest_objective_distance[i] == -1):
                        # If the block was inside the sensor range in the previous step and now it is outside
                        # Penalize for getting further
                        self._rewards[i] -= self._get_bootstrap_reward_pick(agent_previous_closest_objective_distance)

                    if (agent_previous_closest_objective_distance != -1 and self._agents_closest_objective_distance[i] != -1):
                        difference_objective_distance = agent_previous_closest_objective_distance - self._agents_closest_objective_distance[i]
                        self._rewards[i] += self._get_bootstrap_reward_pick(difference_objective_distance, True)

                # Reward the agent for carrying a correct block towards the drop zone
                if self._agents_carrying[i] != -1 and self.blocks_color[self._agents_carrying[i]] in self._objective_colors:
                    agent_previous_closest_objective_distance = self._agents_closest_objective_distance[i]
                    self._agents_closest_objective_distance[i] = self._get_distance_to_objective_edge(self.agents_location[i], self.blocks_color[self._agents_carrying[i]])
                    difference_objective_distance = agent_previous_closest_objective_distance - self._agents_closest_objective_distance[i]
                    self._rewards[i] += self._get_bootstrap_reward_drop(difference_objective_distance, self._agents_carrying[i])
                # # ------------------------

                     
        self._update_directions_matrix()
        self._update_distance_matrix()
        observations = []
        for i in range(self.n_agents):
            observation = self._get_obs(i)
            observations.append(observation)
        
        done = False
        # Check if the shared objective is met
        if len(self._task_completed) == self.n_task:
            done = True

        reward = round(sum(self._rewards), 4) # Sum the rewards of all agents of the swarm
        info = {"completed": self._task_completed}
        truncated = False
        
        return observations, reward, done, truncated, info

    
    def print_env(self):
        # Define the size of the visualization grid
        vis_grid_size = 20  # Adjust based on desired resolution

        # Create an empty visual representation of the environment
        visual_grid = [["." for _ in range(vis_grid_size)] for _ in range(vis_grid_size)]
        
        # Populate the visual grid with blocks
        for i, block in enumerate(self.blocks_location):
            # Convert continuous coordinates to discrete grid positions
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
        for row in visual_grid:
            print(" ".join(row))
        
        print()

        
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
        directions = neighbors[:, :, 2]  # TODO: sin cos
        directions_sin = np.sin(np.radians(neighbors[:, :, 2]))
        directions_cos = np.cos(np.radians(neighbors[:, :, 2]))

        # heading = heading / self.sensor_angle # TODO: sin cos of the heading
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