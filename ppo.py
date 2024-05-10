# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:55:46 2023

@author: Adrian Menor
"""
import os
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from networks import ActorNetwork, CriticNetwork, CNN_CriticNetwork
from environment.perception import calculate_perception as perception
from environment.perception import LIDAR as LIDAR
from environment.perception import CNN_critic_perception

class PPO:
    def __init__(self, env, MAPPO: bool, load_pretrained_models: bool):
        # Assessing whether to do IPPO or MAPPO
        self.MAPPO = MAPPO

        # Extracting environment information
        self.env = env
        self.actor_obs_dim = 108 # CHANGE into flexible variables

        # Deciding whether to use a global critic
        if self.MAPPO:
          # self.critic_obs_dim = 2 * (self.env.n_agents - 1) + 4 # CHANGE
          self.critic_obs_dim = 100 # CHANGE, corresponding to the tensor grid size

        else:
          self.critic_obs_dim = self.actor_obs_dim

        self.act_dim = 2 # CHANGE into flexible variables

        # Initializing hyperparameters
        self._init_hyperparameters()

        # Checking if a GPU is available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initializing actor and critic networks
        self.actor = ActorNetwork(self.actor_obs_dim, self.act_dim)
        
        if self.MAPPO:
            self.critic = CNN_CriticNetwork(input_channels=4, grid_size=self.critic_obs_dim)
        
        else:
            self.critic = CriticNetwork(self.critic_obs_dim)
        
        # Loading pretrained models
        self.training_event = 0 # Useful when doing several training runs
        if load_pretrained_models:
            self.load_pretrained_models()

        # Initialize actor optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        # Initialize critic optimizer
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Creating the covariance matrix for get_action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = torch.diag(self.cov_var)

        # Logger to print learning progress
        self.logger = {
            'batch_rews:': [] # episodic returns in batch
        }
        
        # Reward counter
        self.rewards = []
        self.timesteps = []
        self.best_reward = -1e6 # Dummy best reward for saving best models

    def load_pretrained_models(self):
        '''
        Loading pretrained models.
        '''
        # Get the current working directory
        current_directory = os.getcwd()

        # Specify the folder name
        folder_name = 'models'

        # Create the full path to the folder
        folder_path = os.path.join(current_directory, folder_name)

        # Create the full path to the file
        actor_file_path = os.path.join(folder_path, 'actor_' + str(self.training_event) + '.pth')
        critic_file_path = os.path.join(folder_path, 'critic_' + str(self.training_event) + '.pth')

        # Check if the files exist
        if not os.path.exists(actor_file_path):
            raise FileNotFoundError(f"The model file {actor_file_path} does not exist.")

        # Load the state dictionary from the pre-trained model
        pretrained_actor_state_dict = torch.load(actor_file_path, map_location=self.actor.device)
        pretrained_critic_state_dict = torch.load(critic_file_path, map_location=self.critic.device)
        
        # Filter out unnecessary keys and update the state dictionary of the actor and the critic
        actor_state_dict = self.actor.state_dict()
        pretrained_actor_state_dict = {k: v for k, v in pretrained_actor_state_dict.items() if k in actor_state_dict}
        actor_state_dict.update(pretrained_actor_state_dict)
        self.actor.load_state_dict(actor_state_dict)
        
        critic_state_dict = self.critic.state_dict()
        pretrained_critic_state_dict = {k: v for k, v in pretrained_critic_state_dict.items() if k in critic_state_dict}
        critic_state_dict.update(pretrained_critic_state_dict)
        self.critic.load_state_dict(critic_state_dict)
        
        print('Pretrained models have been loaded.', flush=True)

    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        time_index = 0 # For model checkpoint purposes
        checkpoint_freq = np.arange(0,total_timesteps,self.save_freq)

        while t_so_far < total_timesteps:

            # Obtain batch of data
            batch_actor_obs, batch_critic_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected in this batch
            t_so_far += np.sum(batch_lens)

            # Calculate V_{phi, k}
            V, _, _ = self.evaluate(batch_actor_obs, batch_critic_obs, batch_acts)

            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs, entropy = self.evaluate(batch_actor_obs, batch_critic_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor loss
                entropy_loss = entropy.mean()
                actor_loss = (-torch.min(surr1, surr2)).mean() - self.ent_coef * entropy_loss

                # Calculate critic loss
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # For gradient clipping
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

            # Print a summary of our training so far
            self._log_summary() 
            
            # Saving training progress data
            self.rewards.append(np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']]))
            self.timesteps.append(t_so_far)
            
            # Saving models if best performance is achieved
            # Tracking best-reward so far
            if self.rewards[-1] > self.best_reward:
                self.save_models(t_so_far, best_model=True)
                self.best_reward = self.rewards[-1]
            
            # Save models and training progress
            if int(t_so_far)>checkpoint_freq[time_index]:
                self.save_models(t_so_far, best_model=False)
                self.save_progress()
                time_index += 1

    def evaluate(self, batch_actor_obs, batch_critic_obs, batch_acts):
        '''
        Calculates value of the current state and the log probabilities of
        batch actions using most recent actor network.
        '''
        # Value of current state
        V = self.critic(batch_critic_obs).squeeze()

        # Calculate log probabilities of batch actions
        mean = self.actor(batch_actor_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs, dist.entropy()

    def rollout(self):
        '''
        Collects batch of data to update the actor and critic.
        '''
        # Batch data
        batch_actor_obs = []       # batch actor observations
        batch_critic_obs = []      # batch critic observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        # Number of timesteps run so far in this batch
        t=0

        while t < self.timesteps_per_batch:

            # Rewards this episode
            ep_rews= []

            # Resetting the environment
            self.env.reset()
            agent_positions = self.env.agent_positions 
            target_positions = self.env.target_positions
            obstacle_centers = self.env.obstacle_centers
            obstacle_heights = self.env.obstacle_heights
            obstacle_widths = self.env.obstacle_widths

            # Getting observations from the environment
            # actor_obs, critic_obs = perception(agent_positions, target_positions, obstacle_centers, obstacle_heights, obstacle_widths)
            
            # Actor observation
            lidar = LIDAR(agent_positions, target_positions, obstacle_centers, obstacle_heights, obstacle_widths)
            actor_obs = lidar.FF_LIDAR()
            
            # Critic observation
            if self.MAPPO:
                CNN_perception = CNN_critic_perception(agent_positions, target_positions, obstacle_centers, 
                                                                obstacle_widths, obstacle_heights, 100,
                                                                env.params["map_size"], env.params["collision_radius"])
                critic_obs = CNN_perception.full_CNN_perception()
                
            else:
                critic_obs = actor_obs

            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran in this batch so far
                t += 1

                # Collecting observations and actions from all agents
                actions = np.zeros((self.env.n_agents, 2))

                for agent in range(self.env.n_agents):
                  # Collect observation
                  batch_actor_obs.append(actor_obs[:, agent])

                  if self.MAPPO:
                    batch_critic_obs.append(critic_obs[:,:,:,agent])

                  else:
                    batch_critic_obs.append(actor_obs[:, agent])

                  action, log_prob = self.get_action(actor_obs[:, agent])

                  # Collect action and log prob
                  batch_acts.append(action)
                  batch_log_probs.append(log_prob)

                  # Stacking actions to pass on 
                  actions[agent, :] = action

                # Observing next state and reward
                new_agent_positions, rew, done, info = env.step(actions)
                # actor_obs, critic_obs = perception(new_agent_positions, info['target_positions'], info['obstacle_centers'], info['obstacle_heights'], info['obstacle_widths'])
                
                # New Actor observation
                lidar = LIDAR(new_agent_positions, info['target_positions'], info['obstacle_centers'], info['obstacle_heights'], info['obstacle_widths'])
                actor_obs = lidar.FF_LIDAR()
                
                # New Critic observation
                if self.MAPPO:
                    CNN_perception = CNN_critic_perception(new_agent_positions, info['target_positions'], info['obstacle_centers'], 
                                                                    info['obstacle_widths'], info['obstacle_heights'], 100,
                                                                    env.params["map_size"], env.params["collision_radius"])
                    critic_obs = CNN_perception.full_CNN_perception()
                    
                else:
                    critic_obs = actor_obs

                # Collect reward
                for i in range(self.env.n_agents):
                  ep_rews.append(rew[i]) # Slicing of rew because each agent can have different rewards, depends on local ratio
                
                # Break if the env says the episode is done
                if done:
                    break

            # Collect episodic length and rewards
            for _ in range(self.env.n_agents):
              batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

        # Reshape data as NumPy arrays
        batch_actor_obs = np.array(batch_actor_obs, dtype=np.float32)
        batch_critic_obs = np.array(batch_critic_obs, dtype=np.float32)
        batch_acts = np.array(batch_acts, dtype=np.float32)
        batch_log_probs = np.array(batch_log_probs, dtype=np.float32)

        # Reshape data as tensors
        batch_actor_obs = torch.tensor(batch_actor_obs, device=self.device).clone().detach()
        batch_critic_obs = torch.tensor(batch_critic_obs, device=self.device).clone().detach()
        batch_acts = torch.tensor(batch_acts, device=self.device).clone().detach()
        batch_log_probs = torch.tensor(batch_log_probs, device=self.device).clone().detach()

        # Computing rewards-to-go
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Logging the episodic rewards in this batch
        self.logger['batch_rews'] = batch_rews

        # Return batch data
        return batch_actor_obs, batch_critic_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, actor_obs):
        '''
        Gets sample action from the actor.
        '''
        # Query the actor for a mean action
        mean = self.actor.forward(actor_obs)

        # Create Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat) # there is a GPT alternative

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Returning action and log_prob, without the graphs
        return np.asarray(action.cpu()), np.asarray(log_prob.detach().cpu())

    # def compute_rtgs(self, batch_rews):
    #     '''
    #     Computes the rewards-to-go per episode, per batch.
    #     The output shape is (num timesteps per episode)
    #     '''
    #     batch_rtgs = []

    #     # Iterate through each episode backwards to maintain same order in batch_rtgs
    #     for ep_rews in reversed(batch_rews):
    #         discounted_reward = 0 # The discounted reward so far

    #         for rew in reversed(ep_rews):
    #             discounted_reward = rew + discounted_reward * self.gamma
    #             batch_rtgs.insert(0,discounted_reward)

    #     # Convert rewards-to-go into a tensor
    #     batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)

    #     return batch_rtgs
    
    def compute_rtgs(self, batch_rews):
        '''
        Computes the rewards-to-go per episode, per batch.
        The output shape is (timesteps_per_batch * number of agents)
        '''
        batch_rtgs = []

        # Looping over all collected episodes
        for episode in batch_rews:
          # Checking that all agents have collected the same amount of experiences
          # on a given episode
          if len(episode) % self.env.n_agents !=0: # self.env.n_agents 
              raise TypeError('Episode list must be a multiple of the number of agents.')
          # Slicing episode to filter a specific agent reward.
          # Each list element in agent_rewards contains the filtered rewards obtained by that agent
          # in that episode.
          agent_rewards = []

          for agent in range(self.env.n_agents):
            sliced_agent_reward = [episode[k] for k in np.arange(agent, len(episode), self.env.n_agents)]
            agent_rewards.append(sliced_agent_reward)

          # Calculating the rtgs of each agent, at each point in time
          rtgs = [] # array to which to append the rtgs at each timestep
          for agent_history in agent_rewards: # looping over agents
              discounted_rewards = []
              running_sum = 0
              
              for r in reversed(agent_history):
                  running_sum = r + self.gamma * running_sum
                  discounted_rewards.insert(0, running_sum) # Insert at the beginning to maintain order
            
              # Appending the discounted rewards to rtgs
              rtgs.append(discounted_rewards)
          
          # Rearranging rtgs list
          rtgs = [list(item) for item in zip(*rtgs)]
          
          # Flattening rtgs list
          rtgs = [val for sublist in rtgs for val in sublist]
          
          # Appending rtgs in correct order to batch_rtgs
          for element in rtgs:
              batch_rtgs.append(element)
        
        # Convert rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)

        return batch_rtgs
    
    def save_models(self, t_so_far, best_model=False):
        '''
        Saves models into folder.
        '''
        # Get the current directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define the folder path and model file name
        folder_path = os.path.join(script_dir, 'models')
        
        # Ensure that the 'models' folder exists, create it if it doesn't
        os.makedirs(folder_path, exist_ok=True)
        
        # Full path to the model file
        if best_model:
            actor_path = os.path.join(folder_path, 'actor_best_' + str(self.training_event) + '.pth')
            critic_path = os.path.join(folder_path, 'critic_best_' + str(self.training_event) + '.pth')
        else:
            actor_path = os.path.join(folder_path, 'actor_' + str(self.training_event) + '.pth')
            critic_path = os.path.join(folder_path, 'critic_' + str(self.training_event) + '.pth')
        
        # Save models
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        if best_model:
            print(f'Best models have been saved at timestep {t_so_far}.', flush=True)
        else:
            print(f'Models have been saved at timestep {t_so_far}.', flush=True)
        
    def save_progress(self):
        '''
        Saves the rewards and timesteps of the training.
        '''
        
        # Create progress arrays
        rewards = np.asarray(self.rewards)
        timesteps = np.asarray(self.timesteps)
        
        # Get the current directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define the folder path and model file name
        folder_path = os.path.join(script_dir, 'logging')
        
        # Ensure that the 'logging' folder exists, create it if it doesn't
        os.makedirs(folder_path, exist_ok=True)
        
        # Full path to the model file
        rewards_path = os.path.join(folder_path, 'rewards_' + str(self.training_event) + '.npy')
        timesteps_path = os.path.join(folder_path, 'timesteps_' + str(self.training_event) + '.npy')
        
        # Save rewards and timesteps
        np.save(rewards_path, rewards)
        np.save(timesteps_path, timesteps)

    def _init_hyperparameters(self):
        # Default hyperparameters
        self.timesteps_per_batch = 4800 // self.env.n_agents # To update ANN
        self.max_timesteps_per_episode = 100
        self.gamma = 0.99 # discount factor
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005
        self.ent_coef = 0.001 # Entropy coefficient
        self.max_grad_norm = 10.0 # Max allowed gradient (for clipping)
        self.save_freq = 30_000 # Saving models every save_freq timesteps

    def _log_summary(self):
      '''
      Printing learning progress.
      '''

      # Calculating average episodic reward
      avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])

      # Printing reward information
      print(f"Average Episodic Return: {avg_ep_rews}", flush=True)

      # Reset batch specific logging data
      # self.logger['batch_rews'] = []

if __name__=="__main__":
    import time
    from environment.env_config import env_parameters
    from environment.env_main import Environment

    # Number of training events, for statistical analysis
    events = 1
    
    start_time = time.time()
    for event in range(0, events):
        env = Environment(env_parameters)
        model = PPO(env, MAPPO=True, load_pretrained_models=True)
        model.training_event = event
        model.learn(300_000_000)
        
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime:.2f} seconds")