import numpy as np

import gymnasium as gym
from gymnasium import spaces
from math import sqrt, atan2, degrees

TIME_PER_STEP = 1 # a step in seconds
ROBOT_SIZE = 25 # in cm (diameter)
SENSOR_RANGE = 50 # in cm
MAX_VELOCITY = 200 # in cm/s
VELOCITY = 50 # initial 0 max 200, in cm/s, 50 assumed velocity in cm/s
MAX_DISTANCE = VELOCITY * TIME_PER_STEP # in cm
ARENA_SIZE = 500 # in cm

SIMULATION_ROBOT_SIZE = ROBOT_SIZE / ROBOT_SIZE # 1
SIMULATION_SENSOR_RANGE = SENSOR_RANGE / ROBOT_SIZE # 3
SIMULATION_MAX_DISTANCE = MAX_DISTANCE / ROBOT_SIZE # 2
SIMULATION_ARENA_SIZE = ARENA_SIZE / ROBOT_SIZE # 20

NORTH_EDGE = 180
EAST_EDGE = 90
WEST_EDGE = 270
SOUTH_EDGE = 0

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

# MOVE = 0
# PICK = 1
# DROP = 2
# RANDOM_WALK = 3

class Environment(gym.Env):

    def __init__(
            self, 
            nest = NORTH_EDGE,
            objective = [(RED, NORTH_EDGE)], # objective is a list of tuples (color, edge)
            seed=None,
            size=100, 
            n_agents=3, 
            n_blocks=3, 
            n_neighbors = 4,
            sensor_range = 2,
            sensor_angle = 360,
            max_distance_covered_per_step = 2,
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

        self.n_blocks = n_blocks
        self.blocks_location = np.zeros((self.n_blocks, 2), dtype=float)
        self.blocks_color = np.zeros(self.n_blocks, dtype=int)
        self._blocks_picked_up = np.full(self.n_blocks, -1, dtype=int)
        self._blocks_initial_distance_to_dropzone = np.full(self.n_blocks, -1, dtype=float)
        
        self.sensitivity = sensitivity # How close to interact
        self.n_neighbors = n_neighbors
        self._neighbors = np.zeros((self.n_agents, n_neighbors, 3), dtype=float) # init sensors
        self._previous_neighbors = np.zeros((self.n_agents, n_neighbors, 3), dtype=float) 
        self.sensor_range = sensor_range
        self.sensor_angle = sensor_angle
        self.max_distance_covered_per_step = max_distance_covered_per_step

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
        single_action_space = spaces.Box(low=np.array([0, 0]), 
                                         high=np.array([max_distance_covered_per_step, sensor_angle]), dtype=float)
        
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

    def _calculate_distance_direction(self, pointA, pointB, distance_type='euclidean'):
        x1, y1 = pointA
        x2, y2 = pointB

        # Calculate distance
        if distance_type == 'manhattan':
            distance = abs(x1 - x2) + abs(y1 - y2)
        elif distance_type == 'euclidean':
            distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        else:
            raise ValueError("Invalid distance type. Use 'manhattan' or 'euclidean'.")

        # Calculate direction in degrees
        angle_radians = atan2(y2 - y1, x2 - x1)
        direction_degrees = degrees(angle_radians)

        # Normalize the direction to be between 0 and 360 degrees
        if direction_degrees < 0:
            direction_degrees += 360
        # down is 0/360 degrees, right is 90 degrees, up is 180 degrees, left is 270 degrees

        return distance, direction_degrees
    
    def _detect(self, i):
        # Mimic sensors reading
        
        neighbor_counter = -1
        neighbor = np.zeros((self.n_neighbors, 3), dtype=float)
        # TODO: ensure that the sensors only detect one agent per direction (the closest one)
        
        # Check if sensors detect the edge of the arena
        if self.agents_location[i][0] < self.sensor_range: # Top edge
            neighbor_counter += 1
            neighbor[neighbor_counter] = [1, self.agents_location[i][0], 180]
        if self.size - self.agents_location[i][0] - 1 < self.sensor_range: # Bottom edge
            neighbor_counter += 1
            neighbor[neighbor_counter] = [1, self.size - self.agents_location[i][0] - 1, 0]
        if self.agents_location[i][1] < self.sensor_range: # Left edge
            neighbor_counter += 1
            neighbor[neighbor_counter] = [1, self.agents_location[i][1], 270]
        if self.size - self.agents_location[i][1] - 1 < self.sensor_range: # Right edge
            neighbor_counter += 1
            neighbor[neighbor_counter] = [1, self.size - self.agents_location[i][1] - 1, 90]
        
        # Check if the sensors detect other agents
        for j in range(self.n_agents):
            if i != j:
                distance, direction = self._calculate_distance_direction(self.agents_location[i], 
                                                                        self.agents_location[j])
                if distance <= self.sensor_range: # If the other agent is within the sensor range
                    if (neighbor_counter >= self.n_neighbors - 1):
                        # Substitute with the furthest TODO: check
                        max_distance = -1
                        max_distance_index = -1
                        for k in range(self.n_neighbors):
                            if neighbor[k, 1] > max_distance:
                                max_distance = neighbor[k, 1]
                                max_distance_index = k
                        if distance < max_distance:
                            neighbor[max_distance_index] = [2, distance, direction]
                    else:
                        neighbor_counter += 1
                        neighbor[neighbor_counter] = [2, distance, direction] # 1 to indicate an agent
        
        # Check if the sensors detect blocks
        for j in range(self.n_blocks):
            distance, direction = self._calculate_distance_direction(self.agents_location[i], 
                                                                    self.blocks_location[j])
            # If the block is within the sensor range
            if distance <= self.sensor_range and np.any(self.blocks_location[j] != [-1, -1]):
                if (neighbor_counter >= self.n_neighbors - 1):
                    # Substitute with the furthest
                    max_distance = -1
                    max_distance_index = -1
                    for k in range(self.n_neighbors):
                        if neighbor[k, 1] > max_distance:
                            max_distance = neighbor[k, 1]
                            max_distance_index = k
                    if distance < max_distance:
                        neighbor[max_distance_index] = [self.blocks_color[j], distance, direction]
                else:
                    neighbor_counter += 1
                    neighbor[neighbor_counter] = [self.blocks_color[j], distance, direction]
    
        # Sort the neighbors by distance
        # Define a custom sort key that ignores rows with 0 in the second column
        def sort_key(row):
            return row[1] if row[0] != 0 else np.inf
        
        self._neighbors[i] = np.array(sorted(neighbor.copy(), key=sort_key))

    def _get_obs(self, i):
        carrying = self.blocks_color[self._agents_carrying[i]] if self._agents_carrying[i] != -1 else -1
        return {"neighbors" : self._neighbors[i], "carrying" : carrying}
    
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
            self.blocks_location = self._initial_setting['blocks'].copy()
            self.blocks_color = self._initial_setting['colors'].copy()
        else:
            # Choose the location at random
            for i in range(self.n_agents):
                # Check if the agents are not spawning in the same location
                while True:
                    # Spawn the agents in the nest
                    if self.nest == NORTH_EDGE:
                        self.agents_location[i][0] = 0
                        self.agents_location[i][1] = self.np_random.uniform(0, self.size)
                    elif self.nest == SOUTH_EDGE:
                        self.agents_location[i][0] = self.size - 1
                        self.agents_location[i][1] = self.np_random.uniform(0, self.size)
                    elif self.nest == WEST_EDGE:
                        self.agents_location[i][0] = self.np_random.uniform(0, self.size)
                        self.agents_location[i][1] = 0
                    elif self.nest == EAST_EDGE:
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
        for i in range(self.n_agents):
            self._detect(i)
            observations.append(self._get_obs(i))

        return observations, info
    
    def _is_close(self, a, b):
        return np.linalg.norm(a - b) < self.sensitivity
    
    def _is_agent_close_to_edge(self, agent):
        return agent[0] < self.sensitivity or self.size - agent[0] - 1 < self.sensitivity \
                or agent[1] < self.sensitivity or self.size - agent[1] - 1 < self.sensitivity
    
    def _is_correct_edge(self, agent, block_color):
        current_edge = -1
        if agent[0] < self.sensitivity:
            current_edge = NORTH_EDGE
        elif self.size - agent[0] - 1 < self.sensitivity:
            current_edge = SOUTH_EDGE
        elif agent[1] < self.sensitivity:
            current_edge = WEST_EDGE
        elif self.size - agent[1] - 1 < self.sensitivity:
            current_edge = EAST_EDGE
        
        current_color = block_color
        task = (current_color, current_edge)
        
        if task in self.objective:
            return True
        
        return False

    def _get_distance_to_objective_edge(self, agent, block_color):
        if block_color not in self._objective_colors:
            return -1
        target_edge = self.objective[self._objective_colors.index(block_color)][1]
        if target_edge == NORTH_EDGE:
            target_position = np.array([0, agent[1]])
        elif target_edge == SOUTH_EDGE:
            target_position = np.array([self.size - 1, agent[1]])
        elif target_edge == WEST_EDGE:
            target_position = np.array([agent[0], 0])
        elif target_edge == EAST_EDGE:
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
        
        # Reset sensors
        self._previous_neighbors = self._neighbors.copy()
        # self._neighbors = np.zeros((self.n_agents, self.n_neighbors, 3), dtype=float)
        
        for i in range(self.n_agents):
            # Get the action of the agent
            distance, direction = action[i]
            if distance > self.max_distance_covered_per_step:
                distance = self.max_distance_covered_per_step
            
            # Calculate the new position
            direction_radians = direction
            if direction > 180:
                direction_radians -= 360
            direction_radians = np.radians(direction)
            dx = round(distance * np.cos(direction_radians), 2)
            dy = round(distance * np.sin(direction_radians), 2)
            # `np.clip` to make sure we don't leave the grid
            new_x = self.agents_location[i][0] + dx
            new_y = self.agents_location[i][1] + dy
            new_position = np.array([new_x, new_y])
            # # Check if it is trying to leave the arena
            # if np.any(new_position < 0) or np.any(new_position >= self.size):
            #     self._rewards[i] += REWARD_WRONG_MOVE # Penalize the agent for crushing into wall
            # Clip values to stay within the grid
            new_position = np.clip(new_position, 0, self.size - 1)
            
            # TODO: this checks or penalize with reward system to avoid these
            # Check if the new position is not too close to another agent
            agent_locations_but_i = np.delete(self.agents_location, i, axis=0)
            differences = new_position - agent_locations_but_i
            distances = np.linalg.norm(differences, axis=1)
            occupied_by_agent = np.any(distances < self.sensitivity)
            # Check if the new position is not too close to a block while carrying one (can't pick up two blocks)
            occupied_by_block_while_carrying = np.any(np.linalg.norm(new_position - self.blocks_location, axis=1) < self.sensitivity) and self._agents_carrying[i] != -1
            
            if not (occupied_by_agent or occupied_by_block_while_carrying):
                self.agents_location[i] = new_position # Move the agent to the new position
                
                flag_pick = False
                flag_drop = False
                
                # --- PICK ---
                # Check if the agent is picking up a block
                for j in range(self.n_blocks):
                    # If the block is not picked up by any agent and the agent is not carrying a block
                    # and the agent is close to the block
                    if self._blocks_picked_up[j] == -1 \
                            and self._agents_carrying[i] == -1 \
                            and self._is_close(self.agents_location[i], self.blocks_location[j]):
                        flag_pick = True
                        # Reward the agent for picking up the block
                        if self.blocks_color[j] in self._objective_colors:
                            self._rewards[i] += (REWARD_PICK / 2) + self._get_bootstrap_reward_pick(distance, True)
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
                                self._rewards[i] += (REWARD_DROP / 2) + self._get_bootstrap_reward_drop(distance, self._agents_carrying[i])
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

                     

        observations = []
        for i in range(self.n_agents):
            observation = self._get_obs(i)
            observations.append(observation)
        
        done = False
        # Check if the shared objective is met
        if len(self._task_completed) == self.n_task:
            done = True

        reward = sum(self._rewards) # Sum the rewards of all agents of the swarm
        info = {"completed": self._task_completed}
        truncated = False
        
        return observations, reward, done, truncated, info

    
    def print_env(self):
        # Define the size of the visualization grid
        vis_grid_size = 25  # Adjust based on desired resolution

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
        carrying = np.array([agent['carrying'] for agent in obs])
        

        # One-hot encode types
        types = np.eye(self.n_types)[neighbors[:, :, 0].astype(int)]

        # Normalize distances and directions
        distances = neighbors[:, :, 1] / self.sensor_range
        directions = neighbors[:, :, 2] / self.sensor_angle

        # One-hot encode carrying status
        # Assuming carrying values range from -1 (not carrying) to max_carrying_id
        carrying[carrying == -1] = 0 # Change -1 to 0
        carrying[carrying > 0] = carrying[carrying > 0] - 2 # Change 3, 4, 5, ... to 1, 2, 3, ...
        carrying_one_hot = np.eye(self.n_types - 2)[carrying]

        # Flatten all features and concatenate them into a single vector per agent
        flat_features = np.concatenate([
            types.reshape(types.shape[0], -1),  # Flatten types
            distances.reshape(distances.shape[0], -1),  # Flatten distances
            directions.reshape(directions.shape[0], -1),  # Flatten directions
            carrying_one_hot  # Already appropriate shape
        ], axis=1)

        return flat_features
        
    def close(self):
        pass