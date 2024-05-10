# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:19:58 2023

@author: Adrian Menor
"""

import numpy as np

# Environment configuration
env_parameters = {
    "num_agents": 7,
    "num_obstacles": 45,
    "num_targets": 100,
    "max_obstacle_width": 5.0,
    "max_obstacle_length": 5.0,
    "map_size": (100, 100), # use square map for the CNNs
    "spawn_radius": 10.0, # from map's center
    "collision_radius": 2.1,
    "perception_radius": 1.0, # max distance to target for target collection
    "random_map": False # for random obstacle and target locations
    }