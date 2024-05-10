# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:02:12 2023

@author: Adrian Menor
"""
import numpy as np

def calculate_perception(agent_positions, target_positions, obstacle_centers, obstacle_heights, obstacle_widths):
    '''
    Calculates the actor and critic perception arrays. Except for the critic's distances between
    other agents, all are distances to closest agent, target, or obstacle, respectively.
    
    Output:
        actor_perception = [Δx agent, Δy agent, Δx target, Δy target, Δx obstacle, Δy obstacle]
        critic_perception = [Δx all agents, Δy all agents, Δx target, Δy target, Δx obstacle, Δy obstacle]
    '''
    
    # Initializing output arrays and loop
    n_agents = agent_positions.shape[0]
    actor_perception = np.zeros((6, n_agents))
    critic_perception = np.zeros((2 * (n_agents - 1) + 4, n_agents))

    for i in range(n_agents):
        agent_pos = agent_positions[i]
        
        # Calculating closest obstacle
        obstacle_distances = np.linalg.norm(obstacle_centers - agent_pos, axis=1)
        closest_obstacle_idx = np.argmin(obstacle_distances)
        
        closest_obstacle_dist_x = obstacle_centers[closest_obstacle_idx][0] - agent_pos[0]
        closest_obstacle_dist_y = obstacle_centers[closest_obstacle_idx][1] - agent_pos[1]

        # Calculating closest agent
        closest_agent_dist = float('inf')  # initialize closest distance to infinity
        agent_distances = [] # for the critic
        
        for j in range(n_agents):
            if i != j:  # Skip comparing the same agent
                # Calculate relative x, y distances
                delta_x = agent_positions[j][0] - agent_pos[0]
                delta_y = agent_positions[j][1] - agent_pos[1]
                
                # Calculating agent distances for the critic
                agent_distances.extend([delta_x, delta_y])
    
                # Calculate distance
                distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
    
                # Update closest distance and closest agent position
                if distance < closest_agent_dist:
                    closest_agent_dist = distance
                    closest_agent_dist_x = delta_x
                    closest_agent_dist_y = delta_y

        # Calculating closest target
        closest_target_pos = target_positions[np.argmin(np.linalg.norm(target_positions - agent_pos, axis=1))]
        closest_target_dist_x = closest_target_pos[0] - agent_pos[0]
        closest_target_dist_y = closest_target_pos[1] - agent_pos[1]

        # Constructing actor_perception array
        actor_perception[:, i] = [
            closest_agent_dist_x,
            closest_agent_dist_y,
            closest_target_dist_x,
            closest_target_dist_y,
            closest_obstacle_dist_x,
            closest_obstacle_dist_y,
        ]

        # Fill critic_perception array with relative distances
        critic_perception[:2 * (n_agents - 1), i] = agent_distances
        
        critic_perception[2 * (n_agents - 1):2 * (n_agents - 1) + 4, i] = [
            closest_target_dist_x,
            closest_target_dist_y,
            closest_obstacle_dist_x,
            closest_obstacle_dist_y
        ]

    return actor_perception, critic_perception

class LIDAR:
    def __init__(self, agent_positions, target_positions, obstacle_centers, obstacle_heights, obstacle_widths):
        
        # Agent positions and environment information
        self.agent_positions = agent_positions
        self.target_positions = np.asarray(target_positions)
        self.obstacle_centers = obstacle_centers
        self.obstacle_heights = obstacle_heights
        self.obstacle_widths = obstacle_centers
        
        # Agent and target's sizes
        self.agent_radious = 1
        self.target_radious = 1
        
        # LIDAR parameters
        self.num_beams = 36
        self.radious = 20 # Perception radius
        
    def sense_agents(self, agent_index):
        '''
        Perceive whether there are agents within the LIDAR beam.
        '''
        angles = np.linspace(0, 2 * np.pi, self.num_beams, False)
        lidar_measurements = np.ones(self.num_beams) * (self.radious + 10) # Dummy distance value to filter array
        
        # Deleting the current agent position in the array with all agent positions
        agent_pos = self.agent_positions[agent_index]
        modified_agent_positions = np.delete(self.agent_positions, agent_index, axis=0)
        
        # Calculate angle and distance between agents and current agent
        for agent in range(np.shape(modified_agent_positions)[0]):
            
            # Check whether agent is within LIDAR reach
            distance = np.linalg.norm(agent_pos - modified_agent_positions[agent])
            
            # Calculate angle between agents and current agent
            if distance <= self.radious:
                # Array in which to store the radial data
                current_measurement = np.ones(self.num_beams) * (self.radious + 10) # Dummy distance value to filter array
                
                # Calculate angle
                delta_x = modified_agent_positions[agent,0] - agent_pos[0]
                delta_y = modified_agent_positions[agent,1] - agent_pos[1]
                
                angle = np.arctan2(delta_y, delta_x)
                
                # Check the half-angle aperture of the agent
                aperture_angle = np.arctan2(self.agent_radious, distance)
                
                # Angular coverage of the agent in the FoV
                min_angle = np.mod(angle - aperture_angle, 2*np.pi)
                max_angle = np.mod(angle + aperture_angle, 2*np.pi)
                
                # Find the indices of angles within the specified angle aperture
                if min_angle > max_angle:
                    
                    # Fixing boundary conditions issue when the detected agent lies between the I and IV quadrants
                    indices_within_aperture = np.where(~((angles <= min_angle) & (angles >= max_angle)))
                    
                else:
                    indices_within_aperture = np.where((angles >= min_angle) & (angles <= max_angle))
                
                # Update the values at the detected indices with the measured distance
                current_measurement[indices_within_aperture] = distance
                
                # Update lidar_measurements where current_measurement is smaller
                lidar_measurements[current_measurement < lidar_measurements] = current_measurement[current_measurement < lidar_measurements]
             
        # Replace values in lidar_measurements larger than distance with distance
        lidar_measurements[lidar_measurements >= self.radious] = self.radious
             
        return lidar_measurements, angles

    def sense_targets(self, agent_index):
        '''
        Perceive whether there are targets within the LIDAR beam.
        '''
        angles = np.linspace(0, 2 * np.pi, self.num_beams, False)
        lidar_measurements = np.ones(self.num_beams) * (self.radious + 10) # Dummy distance value to filter array
        
        # Extracting the current agent position
        agent_pos = self.agent_positions[agent_index]
        
        # Calculate angle and distance between the targets and current agent
        for target in range(np.shape(self.target_positions)[0]):
            
            # Check whether target is within LIDAR reach
            distance = np.linalg.norm(agent_pos - self.target_positions[target])
            
            # Calculate angle between agents and current agent
            if distance <= self.radious:
                # Array in which to store the radial data
                current_measurement = np.ones(self.num_beams) * (self.radious + 10) # Dummy distance value to filter array
                
                # Calculate angle
                delta_x = self.target_positions[target,0] - agent_pos[0]
                delta_y = self.target_positions[target,1] - agent_pos[1]
                
                angle = np.arctan2(delta_y, delta_x)
                
                # Check the half-angle aperture of the agent
                aperture_angle = np.arctan2(self.agent_radious, distance)
                
                # Angular coverage of the agent in the FoV
                min_angle = np.mod(angle - aperture_angle, 2*np.pi)
                max_angle = np.mod(angle + aperture_angle, 2*np.pi)
                
                # Find the indices of angles within the specified angle aperture
                if min_angle > max_angle:
                    
                    # Fixing boundary conditions issue when the detected target lies between the I and IV quadrants
                    indices_within_aperture = np.where(~((angles <= min_angle) & (angles >= max_angle)))
                    
                else:
                    indices_within_aperture = np.where((angles >= min_angle) & (angles <= max_angle))
                
                # Update the values at the detected indices with the measured distance
                current_measurement[indices_within_aperture] = distance
                
                # Update lidar_measurements where current_measurement is smaller
                lidar_measurements[current_measurement < lidar_measurements] = current_measurement[current_measurement < lidar_measurements]
             
        # Replace values in lidar_measurements larger than distance with distance
        lidar_measurements[lidar_measurements >= self.radious] = self.radious
             
        return lidar_measurements, angles
    
    def occlusion(self, agent_index):
        '''
        Simulates occlusion by filtering the LIDAR reading to only detect the
        closest agent, target, or obstacle. Currently not fully implemented.
        '''
        # Obtaining LIDAR readings of the different environment features
        agents, _ = self.sense_agents(agent_index)
        targets, _ = self.sense_targets(agent_index)
        obstacles, _ = self.sense_obstacles(agent_index)
    
    def sense_obstacles(self, agent_index):
        '''
        Perceive whether there are obstacles within the LIDAR beam.
        '''
        angles = np.linspace(0, 2 * np.pi, self.num_beams, False)
        lidar_measurements = np.ones(self.num_beams) * (self.radious + 10) # Dummy distance value to filter array
        
        # Extracting the current agent position
        agent_pos = self.agent_positions[agent_index]
        
        # Calculate angle and distance between the targets and current agent
        for obstacle in range(np.shape(self.obstacle_centers)[0]):
            
            # Check whether obstacle is within LIDAR reach
            distance = np.linalg.norm(agent_pos - self.obstacle_centers[obstacle])
            
            # Calculate angle between agents and current agent
            if distance <= self.radious:
                # Array in which to store the radial data
                current_measurement = np.ones(self.num_beams) * (self.radious + 10) # Dummy distance value to filter array
                
                # Calculate angle
                delta_x = self.obstacle_centers[obstacle,0] - agent_pos[0]
                delta_y = self.obstacle_centers[obstacle,1] - agent_pos[1]
                
                angle = np.arctan2(delta_y, delta_x)
                
                # Check the half-angle aperture of the agent
                aperture_angle = np.arctan2(self.agent_radious, distance)
                
                # Angular coverage of the agent in the FoV
                min_angle = np.mod(angle - aperture_angle, 2*np.pi)
                max_angle = np.mod(angle + aperture_angle, 2*np.pi)
                
                # Find the indices of angles within the specified angle aperture
                if min_angle > max_angle:
                    
                    # Fixing boundary conditions issue when the detected target lies between the I and IV quadrants
                    indices_within_aperture = np.where(~((angles <= min_angle) & (angles >= max_angle)))
                    
                else:
                    indices_within_aperture = np.where((angles >= min_angle) & (angles <= max_angle))
                
                # Update the values at the detected indices with the measured distance
                current_measurement[indices_within_aperture] = distance
                
                # Update lidar_measurements where current_measurement is smaller
                lidar_measurements[current_measurement < lidar_measurements] = current_measurement[current_measurement < lidar_measurements]
             
        # Replace values in lidar_measurements larger than distance with distance
        lidar_measurements[lidar_measurements >= self.radious] = self.radious
             
        return lidar_measurements, angles
    
    def FF_LIDAR(self):
        '''
        Return all the LIDAR readings as a flat vector for Feed Forwards ANNs.
        '''
        perception = np.zeros((int(self.num_beams*3), self.agent_positions.shape[0]))
        
        for agent in range(self.agent_positions.shape[0]):
            agents, _ = self.sense_agents(agent)
            targets, _ = self.sense_targets(agent)
            obstacles, _ = self.sense_obstacles(agent)
            
            # Combining all readings into 1
            combination = np.hstack((agents, targets, obstacles))
            
            # Appending to perception array
            perception[:,agent] = combination
            
        # Normalising matrix
        perception = perception / self.radious
            
        return perception
    
    def combined_perception(self):
        '''
        Combines the perception of all agents as a tensor of shape [3, num_beams, num_agents]
        '''
        perception = np.zeros((3, self.num_beams, self.agent_positions.shape[0]))
        
        for agent in range(self.agent_positions.shape[0]):
            agents, _ = self.sense_agents(agent)
            targets, _ = self.sense_targets(agent)
            obstacles, _ = self.sense_obstacles(agent)
            
            # Combining all readings into 1
            combination = np.vstack((agents, targets, obstacles))
            
            # Appending to perception array
            perception[:,:,agent] = combination
            
        # Normalising matrix
        perception = perception / self.radious
            
        return perception
    
class CNN_critic_perception:
    
    def __init__(self, agent_positions, target_positions, obstacle_centers, obstacle_widths, obstacle_lengths, grid_size, map_size, collision_radius):
        
        # Agent positions and environment information
        self.agent_positions = agent_positions
        self.target_positions = np.asarray(target_positions[:-1]) # Removing dummy target
        self.obstacle_centers = obstacle_centers
        self.obstacle_widths = obstacle_widths
        self.obstacle_lengths = obstacle_lengths
        self.map_size = map_size
        
        # Perception Tensor specifications
        self.grid_size = grid_size
        self.collision_radius = collision_radius        
        
    def create_environment_tensor(self, agent_idx):
        '''
        Creates a squared image tensor of shape [4, grid_size, grid_size]. The 4 channels refer to agents, targets, obstacles
        and self agent position.
        This is used for the critic's CNN input.
        ONLY TESTED WITH SQUARE ENVIRONMENTS!!!
        '''
        # Initialize the tensor with zeros
        environment_tensor = np.zeros((4, self.grid_size, self.grid_size)).astype(int)
    
        # Define the grid boundaries
        x_min, y_min = 0, 0
        x_max, y_max = self.map_size[0], self.map_size[1]
    
        # Map positions to grid coordinates
        agent_positions_grid = (self.agent_positions / (x_max - x_min) * self.grid_size).astype(int)
        
        if self.target_positions.shape[0] > 0: # Preventing error when no targets are present
            target_positions_grid = (self.target_positions / (x_max - x_min) * self.grid_size).astype(int)
            
        if self.obstacle_centers.shape[0] > 0: # Preventing error when no obstacles are present
            obstacle_centers_grid = (self.obstacle_centers / (x_max - x_min) * self.grid_size).astype(int)
        
        # Calculate step ratio between environment size, and grid size
        img_ratio = self.grid_size / self.map_size[0]    
    
        # Set values in the tensor for agents, targets, and obstacles
        for pos in agent_positions_grid:
            half_width = self.collision_radius
            half_length = self.collision_radius
    
            x_range = slice(int(max(0, pos[0] - half_width*img_ratio)), int(min(self.grid_size, pos[0] + half_width*img_ratio)))
            y_range = slice(int(max(0, pos[1] - half_length*img_ratio)), int(min(self.grid_size, pos[1] + half_length*img_ratio)))
            
            environment_tensor[0, x_range, y_range] = 1  # Channel 0: Agent
            
        if self.target_positions.shape[0] > 0: # Preventing error when no targets are present
            for pos in target_positions_grid:
                half_width = self.collision_radius
                half_length = self.collision_radius
        
                x_range = slice(int(max(0, pos[0] - half_width*img_ratio)), int(min(self.grid_size, pos[0] + half_width*img_ratio)))
                y_range = slice(int(max(0, pos[1] - half_length*img_ratio)), int(min(self.grid_size, pos[1] + half_length*img_ratio)))
                
                environment_tensor[1, x_range, y_range] = 1  # Channel 1: Target
         
        if self.obstacle_centers.shape[0] > 0: # Preventing error when no obstacles are present
            for i in range(self.obstacle_centers.shape[0]):
                    center = obstacle_centers_grid[i]
                    half_width = self.obstacle_widths[i] // 2
                    half_length = self.obstacle_lengths[i] // 2
        
                    x_range = slice(int(max(0, center[0] - half_width*img_ratio)), int(min(self.grid_size, center[0] + half_width*img_ratio)))
                    y_range = slice(int(max(0, center[1] - half_length*img_ratio)), int(min(self.grid_size, center[1] + half_length*img_ratio)))
        
                    environment_tensor[2, x_range, y_range] = 1  # Channel 2: Obstacle
                    
        if agent_idx > self.agent_positions.shape[0]:
            raise Exception("Agent index is larger than the number of agents in the environment.")
            
        #  Adding self agent position
        pos = agent_positions_grid[agent_idx]
        half_width = self.collision_radius
        half_length = self.collision_radius
        
        x_range = slice(int(max(0, pos[0] - half_width*img_ratio)), int(min(self.grid_size, pos[0] + half_width*img_ratio)))
        y_range = slice(int(max(0, pos[1] - half_length*img_ratio)), int(min(self.grid_size, pos[1] + half_length*img_ratio)))
        environment_tensor[3, x_range, y_range] = 1  # Channel 3: Self agent
    
        return environment_tensor
    
    def full_CNN_perception(self):
        '''
        Creates full observation tensor containing all the critic observations of all agents.
        Output Tensor shape is [4, grid_size, grid_size, num_agents]
        '''
        # Initialize the tensor with zeros
        full_environment_tensor = np.zeros((4, self.grid_size, self.grid_size,self.agent_positions.shape[0])).astype(int)
        
        # Appending to tensor
        for agent in range(self.agent_positions.shape[0]):
            full_environment_tensor[:,:,:,agent] = self.create_environment_tensor(agent).astype(int)
            
        return full_environment_tensor

if __name__ == "main":
    # Example usage
    n_agents = 3
    n_targets = 52
    n_obstacles = 2
    
    agent_positions = -np.zeros((n_agents, 2)) + np.random.uniform(low=-10, high=10, size=(n_agents, 2))
    target_positions = np.random.uniform(low=-10, high=10, size=(n_targets, 2))
    obstacle_centers = np.random.uniform(low=-10, high=10, size=(n_obstacles, 2))
    obstacle_heights = np.random.uniform(low=1, high=5, size=n_obstacles)
    obstacle_widths = np.random.uniform(low=1, high=5, size=n_obstacles)
    
    # Calculate perceptions
    actor_perception, critic_perception = calculate_perception(agent_positions, target_positions, obstacle_centers,
                                                              obstacle_heights, obstacle_widths)
    
    # Displaying the calculated perceptions
    print("Actor Perception:")
    print(actor_perception)
    print("Shape:", actor_perception.shape)
    print("\nCritic Perception:")
    print(critic_perception)
    print("Shape:", critic_perception.shape)