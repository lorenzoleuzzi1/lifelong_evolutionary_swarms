import numpy as np

import gymnasium as gym
from gymnasium import spaces
from math import sqrt, atan2, degrees

TIME_PER_STEP = 1 # a step is seconds
ROBOT_SIZE = 25 # in cm (diameter)
SENSOR_RANGE = 75 # in cm #TODO change
MAX_VELOCITY = 200 # in cm/s
VELOCITY = 100 # initial 0 max 200, in cm/s
MAX_ACC = 400 # in cm/s^2
MAX_DISTANCE = VELOCITY * TIME_PER_STEP # in cm

SIMULATION_ROBOT_SIZE = ROBOT_SIZE / ROBOT_SIZE # 1
SIMULATION_SENSOR_RANGE = SENSOR_RANGE / ROBOT_SIZE # 3
SIMULATION_MAX_DISTANCE = MAX_DISTANCE / ROBOT_SIZE # 4
ARENA_SIZE = 500 # in cm
SIMULATION_ARENA_SIZE = ARENA_SIZE / ROBOT_SIZE # robot size is 1 in the simulation

MOVE = 0
PICK_UP = 1
PUT_DOWN = 2

TOP_EDGE = 0
RIGHT_EDGE = 1
LEFT_EDGE = 2
BOTTOM_EDGE = 3
RED = 3
BLUE = 4
GREEN = 5
YELLOW = 6
PURPLE = 7
ORANGE = 8
GREY = 9

REWARD_RIGHT_PICKUP = 10
REWARD_RIGHT_PUTDOWN = 20
REWARD_WRONG_PICKUP = -5
REWARD_WRONG_PUTDOWN = -10
REWARD_MOVING_RIGHT_DIRECTION = 2
REWARD_MOVE = -1

class Environment(gym.Env):

    def __init__(
            self, 
            objective = [(2, 0)], # objective is a list of tuples (color, edge)
            seed=None,
            size=100, 
            n_agents=3, 
            n_blocks=3, 
            n_neighbors = 4,
            sensor_range = 2,
            sensor_angle = 360,
            max_distance_covered_per_step = 5,
            sensitivity = 0.5, # How close the agent can get to the block to pick it up 
            initial_setting = None
            ):
        self.objective = objective
        self._completed = []
        
        self.seed = seed
        self.size = size  # The size of the square grid
        
        self.n_agents = n_agents
        self.agents_location = np.zeros((self.n_agents, 2), dtype=float)
        self._agents_picked_up = np.full(self.n_agents, -1, dtype=int)

        self.n_blocks = n_blocks
        self.blocks_location = np.zeros((self.n_blocks, 2), dtype=float)
        self.blocks_color = np.zeros(self.n_blocks, dtype=int)
        self._blocks_picked_up = np.full(self.n_blocks, -1, dtype=int)
        
        self._sensitivity = sensitivity # How close to interact
        self.n_neighbors = n_neighbors
        self._neighbors = np.zeros((self.n_agents, n_neighbors, 3), dtype=float) # init sensors
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
        
        # Define the action space for a single robot
        single_robot_action_space = spaces.Dict({
            "move": spaces.Box(low=np.array([0, 0]), high=np.array([max_distance_covered_per_step, sensor_angle]), dtype=float),
            "action": spaces.Discrete(3)  # 0: MOVE, 1: PICK_UP, 2: PUT_DOWN
        })

        # Create a tuple of action spaces, one for each robot
        self.action_space = spaces.Tuple([single_robot_action_space for _ in range(self.n_agents)])

        self.observation_space = spaces.Dict(
            {
                "neighbors": spaces.Box(
                                low=np.zeros((n_agents, n_neighbors, 3), dtype=float),
                                high=np.array([[[self.n_types, sensor_range, sensor_angle]] * n_neighbors] * n_agents),
                                dtype=np.float32
                            ),
                "carrying": spaces.Box(-1, 9, shape=(self.n_agents,), dtype=int)
            }
        )

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
    
    def _get_obs(self):
        # Reset sensors
        self._neighbors = np.zeros((self.n_agents, self.n_neighbors, 3), dtype=float)
        
        # Mimic sensors reading
        for i in range(self.n_agents):
            neighbor_counter = -1
            # TODO: ensure that the sensors only detect one agent per direction (the closest one)
            
            # Check if sensors detect the edge of the arena
            if self.agents_location[i][0] < self.sensor_range: # Top edge
                neighbor_counter += 1
                self._neighbors[i, neighbor_counter] = [1, self.agents_location[i][0], 180]
            if self.size - self.agents_location[i][0] - 1 < self.sensor_range: # Bottom edge
                neighbor_counter += 1
                self._neighbors[i, neighbor_counter] = [1, self.size - self.agents_location[i][0] - 1, 0]
            if self.agents_location[i][1] < self.sensor_range: # Left edge
                neighbor_counter += 1
                self._neighbors[i, neighbor_counter] = [1, self.agents_location[i][1], 270]
            if self.size - self.agents_location[i][1] - 1 < self.sensor_range: # Right edge
                neighbor_counter += 1
                self._neighbors[i, neighbor_counter] = [1, self.size - self.agents_location[i][1] - 1, 90]
            
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
                                if self._neighbors[i, k, 1] > max_distance:
                                    max_distance = self._neighbors[i, k, 1]
                                    max_distance_index = k
                            if distance < max_distance:
                                self._neighbors[i, max_distance_index] = [2, distance, direction]
                        else:
                            neighbor_counter += 1
                            self._neighbors[i, neighbor_counter] = [2, distance, direction] # 1 to indicate an agent
            
            # Check if the sensors detect blocks
            for j in range(self.n_blocks):
                distance, direction = self._calculate_distance_direction(self.agents_location[i], 
                                                                        self.blocks_location[j])
                if distance <= self.sensor_range: # If the block is within the sensor range
                    if (neighbor_counter >= self.n_neighbors - 1):
                        # Substitute with the furthest
                        max_distance = -1
                        max_distance_index = -1
                        for k in range(self.n_neighbors):
                            if self._neighbors[i, k, 1] > max_distance:
                                max_distance = self._neighbors[i, k, 1]
                                max_distance_index = k
                        if distance < max_distance:
                            self._neighbors[i, max_distance_index] = [self.blocks_color[j], distance, direction]
                    else:
                        neighbor_counter += 1
                        self._neighbors[i, neighbor_counter] = [self.blocks_color[j], distance, direction]
        
        # Sort the neighbors by distance
        # Define a custom sort key that ignores rows with 0 in the second column
        def sort_key(row):
            return row[1] if row[0] != 0 else np.inf
        
        self._neighbors = np.array([sorted(subarr, key=sort_key) for subarr in self._neighbors])

        return {"neighbors": self._neighbors, "carrying": self._agents_picked_up}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agents_picked_up = np.full(self.n_agents, -1, dtype=int)
        self._blocks_picked_up = np.full(self.n_blocks, -1, dtype=int)
        
        if self._initial_setting is not None:
            self.agents_location = self._initial_setting['agents'].copy()
            self.blocks_location = self._initial_setting['blocks'].copy()
            self.blocks_color = self._initial_setting['colors'].copy()
        else:
            # Choose the agent's location uniformly at random
            for i in range(self.n_agents):
                # Check if the agents are not spawning in the same location
                while True:
                    self.agents_location[i] = self.np_random.uniform(0, self.size, size=2)
                    if i == 0 or not np.any(np.linalg.norm(self.agents_location[i] - self.agents_location[:i], axis=1) < self._sensitivity):
                        break

            for i in range(self.n_blocks):
                # Check if the blocks are not spawning in the same location
                while True:
                    self.blocks_location[i] = self.np_random.uniform(0, self.size, size=2)
                    if i == 0 or not np.any(np.linalg.norm(self.blocks_location[i] - self.blocks_location[:i], axis=1) < self._sensitivity):
                        break
                self.blocks_color[i] = self.np_random.integers(3, 3 + len(self._colors_map), dtype=int)
        
        self._task_counter = 0
        for color, _ in self.objective:
            for i in range(self.n_blocks):
                if self.blocks_color[i] == color:
                    self._task_counter += 1

        observation = self._get_obs()
        info = {}

        return observation, info
    
    def step(self, action):
        
        self._rewards = np.zeros(self.n_agents)
        
        for i in range(self.n_agents):
            
            if action[i]['action'] == PICK_UP:
                if self._agents_picked_up[i] == -1: # If the agent is not carrying a block
                    for j in range(self.n_blocks):
                        # If the agent is in the same location as the block
                        if np.linalg.norm(self.agents_location[i] - self.blocks_location[j]) < self._sensitivity: 
                            self.blocks_location[j] = [-1,-1] # Set as picked up
                            self._agents_picked_up[i] = self.blocks_color[j] # The agent knows the color of the block it picked up
                            self._blocks_picked_up[j] = i # The block is picked up by the agent

                            # Reward the agent for picking up the block if in objective or not
                            if self.blocks_color[j] in [obj[0] for obj in self.objective]:
                                self._rewards[i] += REWARD_RIGHT_PICKUP
                            else:
                                self._rewards[i] += REWARD_WRONG_PICKUP
                else:
                    self._rewards[i] += REWARD_MOVE
            
            if action[i]['action'] == PUT_DOWN:
                if self._agents_picked_up[i] != -1: # If the agent is carrying a block
                    for j in range(self.n_blocks):
                        if (self._blocks_picked_up[j] == i): # If the block is picked up by the agent
                            self.blocks_location[j] = self.agents_location[i] # Set the block location to the agent location
                            self._agents_picked_up[i] = -1 # The agent is not carrying a block anymore
                            self._blocks_picked_up[j] = -1 # The block is not picked up by any agent anymore

                            # Reward the agent for putting down the block if in objective or not
                            for color, edge in self.objective:
                                if color == self.blocks_color[j]:
                                    if edge == TOP_EDGE and self.blocks_location[j][0] < self._sensitivity:
                                        self._rewards[i] += REWARD_RIGHT_PUTDOWN
                                        self._completed.append(i)
                                    elif edge == RIGHT_EDGE and self.blocks_location[j][1] > self.size - 1 - self._sensitivity:
                                        self._rewards[i] += REWARD_RIGHT_PUTDOWN
                                        self._completed.append(i)
                                    elif edge == LEFT_EDGE and self.blocks_location[j][1] < self._sensitivity:
                                        self._rewards[i] += REWARD_RIGHT_PUTDOWN
                                        self._completed.append(i)
                                    elif edge == BOTTOM_EDGE and self.blocks_location[j][0] > self.size - 1 - self._sensitivity:
                                        self._rewards[i] += REWARD_RIGHT_PUTDOWN
                                        self._completed.append(i)
                                    else:
                                        self._rewards[i] += REWARD_WRONG_PUTDOWN
                else:
                    self._rewards[i] += REWARD_MOVE

            if action[i]['action'] == MOVE:
                # Map the action to the direction we walk in
                distance, direction = action[i]['move']
                
                # Calculate the new position
                direction_radians = direction
                if direction > 180:
                    direction_radians -= 360
                direction_radians = np.radians(direction)
                dx = round(distance * np.cos(direction_radians), 5)
                dy = round(distance * np.sin(direction_radians), 5)
                # `np.clip` to make sure we don't leave the grid
                new_x = np.clip(self.agents_location[i][0] + dx, 0, self.size - 1)
                new_y = np.clip(self.agents_location[i][1] + dy, 0, self.size - 1)
                new_position = np.array([new_x, new_y])
                
                # Check if the new position is not too close to another agent
                agent_locations_but_i = np.delete(self.agents_location, i, axis=0)
                differences = new_position - agent_locations_but_i
                distances = np.linalg.norm(differences, axis=1)
                occupied_by_agent = np.any(distances < self._sensitivity)
                # Check if the new position is not too close to a block while carrying one (can't pick up two blocks)
                occupied_by_block_while_carrying = np.any(np.linalg.norm(new_position - self.blocks_location, axis=1) < self._sensitivity) and self._agents_picked_up[i] != -1
                # Same poisition as before
                same_position = np.all(new_position == self.agents_location[i])
                
                if not occupied_by_agent and not occupied_by_block_while_carrying and not same_position:
                    self.agents_location[i] = new_position

                    # Reward the agent for moving in the right direction
                    if self._agents_picked_up[i] != -1: # If the agent is carrying a block
                        target_edge = -1
                        for color, edge in self.objective:
                            if color == self._agents_picked_up[i]:
                                target_edge = edge
                                break
                        if target_edge == TOP_EDGE:
                            if direction > 135 and direction < 225: # Moving up
                                self._rewards[i] += distance * REWARD_MOVING_RIGHT_DIRECTION
                        elif target_edge == RIGHT_EDGE:
                            if direction > 45 and direction < 135: # Moving right
                                self._rewards[i] += distance * REWARD_MOVING_RIGHT_DIRECTION
                        elif target_edge == LEFT_EDGE:
                            if direction > 225 and direction < 315: # Moving left
                                self._rewards[i] += distance * REWARD_MOVING_RIGHT_DIRECTION
                        elif target_edge == BOTTOM_EDGE:
                            if direction > 315 or direction < 45: # Moving down
                                self._rewards[i] += distance * REWARD_MOVING_RIGHT_DIRECTION

                self._rewards[i] += REWARD_MOVE # Punish the agent for timestep (force efficiency)? TODO: check if it's necessary

        observation = self._get_obs()

        done = False
        # Check if the objective is met
        if len(self._completed) == self._task_counter:
            done = True

        reward = sum(self._rewards)
        info = self._completed
        
        return observation, reward, done, info
    
    def print_env(self):
        # Define the size of the visualization grid
        vis_grid_size = 25  # Adjust based on desired resolution
        
        # Create an empty visual representation of the environment
        visual_grid = [["." for _ in range(vis_grid_size)] for _ in range(vis_grid_size)]
        
        # Populate the visual grid with blocks
        for i, block in enumerate(self.blocks_location):
            # Convert continuous coordinates to discrete grid positions
            x, y = int(block[0] * vis_grid_size / self.size), int(block[1] * vis_grid_size / self.size)
            if 0 <= x < vis_grid_size and 0 <= y < vis_grid_size:
                color_id = self.blocks_color[i]
                color_code = self._colors_map.get(color_id, self._reset_color)
                visual_grid[x][y] = f"{color_code}O{self._reset_color}"
        
        # Populate the visual grid with agents
        for i, agent in enumerate(self.agents_location):
            # Convert continuous coordinates to discrete grid positions
            x, y = int(agent[0] * vis_grid_size / self.size), int(agent[1] * vis_grid_size / self.size)
            if 0 <= x < vis_grid_size and 0 <= y < vis_grid_size:
                if self._agents_picked_up[i] != 0:
                    color_id = self._agents_picked_up[i]
                    color_code = self._colors_map.get(color_id, self._reset_color)
                    visual_grid[x][y] = f"{color_code}{i}{self._reset_color}"
                else:
                    visual_grid[x][y] = str(i)
        
        # Print the visual representation
        for row in visual_grid:
            print(" ".join(row))
        
        print()

        
    def print_neighbors(self):
        for i in range(self.n_agents):
            flag = False
            for j in range(self.n_neighbors):
                if self._neighbors[i,j,0] != 0:
                    entity = "agent"
                    if self._neighbors[i,j,0] != 1:
                        entity = f"block (color: {self._neighbors[i,j,0]})"
                    distance = self._neighbors[i,j,1]
                    direction = self._neighbors[i,j,2]
                    print(f"Agent {i} sees {entity}: {distance} distance and {direction} degrees direction")
                    flag = True
            if not flag:
                print(f"Agent {i} doesn't see anything")
            print()
        
    def close(self):
        pass