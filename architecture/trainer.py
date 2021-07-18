from architecture.transition import Transition
from architecture.network import Network
from architecture.environment import Environment
from architecture.memory import Memory

import math
import random
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
TARGET_EVALUATE = 100
MEMORY_SIZE = 50000


class Trainer:

  def __init__(self, learning_rate):
    print('--- Learning Rate: ', learning_rate, '---')
    self.steps_done = 0
    self.environment = Environment()

    _, _, self.screen_height, self.screen_width = self.environment.get_screen().shape
    self.action_space = self.environment.env.action_space.n
    self.device = self.environment.device

    self.policy_net = Network(self.screen_height, self.screen_width, self.action_space).to(self.device)
    self.target_net = Network(self.screen_height, self.screen_width, self.action_space).to(self.device)

    self.memory = Memory(MEMORY_SIZE)

    self.optimizer = optim.RMSprop(self.policy_net.parameters(), learning_rate)
    self.learning_rate = learning_rate

  def select_training_action(self, state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
    self.steps_done += 1

    if sample > eps_threshold:
      with torch.no_grad():
        return self.policy_net(state).max(1)[1].view(1, 1)
    else:
      return torch.tensor([[random.randrange(self.action_space)]], device = self.device, dtype = torch.long)

  def optimize(self):
    if len(self.memory) < BATCH_SIZE:
      return

    transitions = self.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype = torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device = self.device)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

    # compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize the model
    self.optimizer.zero_grad()
    loss.backward()

    for param in self.policy_net.parameters():
      param.grad.data.clamp_(-1, 1)

    self.optimizer.step()

    return loss

  def train(self, num_episodes: int):
    episode_durations = []

    for i_episode in range(num_episodes):
      # initialize the environment and state
      self.environment.env.reset()
      last_screen = self.environment.get_screen()
      current_screen = self.environment.get_screen()
      state = current_screen - last_screen

      for t in count():
        # select and perform an action
        action = self.select_training_action(state)
        _, reward, done, _ = self.environment.env.step(action.item())

        reward = torch.tensor([reward], device = self.device)

        # Observe new state
        last_screen = current_screen
        current_screen = self.environment.get_screen()
        if not done:
          next_state = current_screen - last_screen
        else:
          next_state = None

        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        self.optimize()
  
        if done:
          episode_durations.append(t + 1)
          break

      # Update the target network, copying all weights and biases in Network
      if i_episode % TARGET_UPDATE == 0:
          self.target_net.load_state_dict(self.policy_net.state_dict())

      if i_episode % TARGET_EVALUATE == 0:
        self.evaluate()

    self.environment.env.render()
    self.environment.env.close()
    self.plot_durations(episode_durations)
  
  def evaluate(self, num_episodes: int = 100):
    episode_durations = []

    with torch.no_grad():
      for i_episode in range(num_episodes):
        self.environment.env.reset()
        last_screen = self.environment.get_screen()
        current_screen = self.environment.get_screen()
        state = current_screen - last_screen

        for t in count():
          action = self.target_net(state).max(1)[1].view(1, 1)
          _, _, done, _ = self.environment.env.step(action.item())

          # Observe new state
          last_screen = current_screen
          current_screen = self.environment.get_screen()
          if not done:
            next_state = current_screen - last_screen
          else:
            next_state = None
        
          # Move to the next state
          state = next_state

          if done:
            episode_durations.append(t + 1)
            break

    print('--- Stats ---')
    print('Variance: ', np.var(episode_durations))
    print('Mean: ', np.std(episode_durations))
    print('Standard Deviation: ', np.mean(episode_durations))

  def plot_durations(self, episode_durations, title: str = 'Training'):
    plt.figure()
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype = torch.float)
    plt.title(f'{title}...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
      means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
      means = torch.cat((torch.zeros(99), means))
      plt.plot(means.numpy())

    plt.savefig(f'{title}-{self.learning_rate}.jpg')
