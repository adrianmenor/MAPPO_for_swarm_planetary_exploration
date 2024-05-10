# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:57:19 2023

@author: Adrian Menor
"""
import numpy as np

def check_targets(agents_pos, collision_radius, target_positions):
    '''
    Checking whether an agent has explored a target.
    
    Output: number of found targets in this step, the new target
    positions (the found targets are removed), and number of targets found by
    each individual agent.
    '''
    # Calculating swarm target findings
    distances = np.linalg.norm(agents_pos[:, None] - target_positions, axis=2)
    
    targets_found = np.any(distances < collision_radius, axis=0)
    
    found_targets = np.sum(targets_found)
    new_target_positions = np.array(target_positions)[~targets_found]
    
    # Calculating individual target findings
    agents_targets_distances  = distances < collision_radius
    individual_found_targets = np.sum(agents_targets_distances, axis=1)

    return found_targets, new_target_positions, individual_found_targets

if __name__ == "main":
    # Example usage:
    agents_pos = np.array([[2, 3], [40, 5], [6, 70]])  # Example agent positions
    collision_radius = 2.0  # Example collision radius
    target_positions = [(1, 2), (4, 5), (7, 8), (8,9)]  # Example target positions
    
    # Visualizing initial positions
    print(f"Initial agent positions: {agents_pos}")
    print(f"Initial target positions: {target_positions}")
    
    # Check and update targets
    found, updated_targets, individual_found_targets = check_targets(agents_pos, collision_radius, target_positions)
    
    # Visualizing updated positions
    print(f"Found targets: {found}")
    print(f"Updated individual found targets: {individual_found_targets}")
    print(f"Updated target positions: {updated_targets}")