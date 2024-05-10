# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:52:40 2024

@author: Adrian Menor
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Load rewards and timesteps
folder_path = 'logging'  # Change this to the correct path if needed
rewards_path = os.path.join(folder_path, 'rewards_example.npy')
timesteps_path = os.path.join(folder_path, 'timesteps_example.npy')

rewards = np.load(rewards_path, allow_pickle=True)
timesteps = np.load(timesteps_path, allow_pickle=True)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(timesteps, rewards, label='Average Rewards')

# Customize the plot
ax.set_title('Rewards Over Timesteps', fontsize=18)
ax.set_xlabel('Timesteps [a.u.]', fontsize=16)
ax.set_ylabel('Average Collective Reward [a.u.]', fontsize=16)
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.legend(fontsize=16)
ax.xaxis.offsetText.set_fontsize(14)

# Add a grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add a light grey background
ax.set_facecolor('#f0f0f0')

# Save the plot as an image
# plt.savefig('rewards_over_timesteps.pdf', format='pdf', bbox_inches='tight')

# Show the plot (optional)
plt.show()
