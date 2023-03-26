import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4,
                 history_length=100000):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q
        self.Q_target = Q_target
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        self.replay_buffer.add_transition(
            state, action, next_state, reward, terminal)

        # 2. sample next batch and perform batch update:
        #       2.1 compute td targets and loss
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(
            self.batch_size)
        batch_states = torch.tensor(batch_states, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long)
        batch_next_states = torch.tensor(
            batch_next_states, dtype=torch.float32)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        # batch_dones = torch.tensor(batch_dones, dtype=torch.bool)
        batch_indices = np.arange(self.batch_size, dtype=np.int64)

        predictions = self.Q(batch_states)
        next_predictions = self.Q_target(batch_next_states)

        predicted_value_of_now = predictions[batch_indices, batch_actions]
        predicted_value_of_future = torch.max(
            next_predictions, dim=1)[0].detach()

        td_target = batch_rewards + self.gamma * \
            predicted_value_of_future * torch.tensor(1 - batch_dones)

        #       2.2 update the Q network
        loss = self.loss_function(predicted_value_of_now, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #       2.3 call soft update for target network
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            pass
            # TODO: take greedy action (argmax)
            state = torch.tensor(state).float().detach()
            state = state.unsqueeze(0)
            action = self.Q(state)
            action_id = torch.argmax(action).item()
        else:
            pass
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = np.random.randint(0, 2)
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
