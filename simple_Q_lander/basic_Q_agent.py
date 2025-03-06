import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random , argparse , time , uuid , pickle

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class QAgent:
    def __init__(self , state_size , action_size , gamma=0.99 , lr=0.001 , batch_size=64 , buffer_size=10000 , epsilon_start=1.0 , epsilon_end=0.01 , epsilon_decay=0.995 ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.qnetwork = QNetwork( state_size , action_size )
        self.optimizer = optim.Adam( self.qnetwork.parameters() , lr=self.lr )

    def act( self , state , deterministic=False ) :
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork( state )
        self.qnetwork.train()

        if deterministic or random.random() > self.epsilon:
            return np.argmax( action_values.cpu().data.numpy() )
        else:
            return random.choice( np.arange(self.action_size) )

    def learn( self , memory ):

        experiences = random.sample( memory , self.batch_size )
        states, actions, rewards, next_states, dones = zip( *experiences )

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

class QAgent_DblQ( QAgent ):
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, batch_size=64, buffer_size=10000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995 ):

        super(QAgent_DblQ, self).__init__( state_size=state_size, action_size=action_size, gamma=gamma, lr=lr, batch_size=batch_size, buffer_size=buffer_size, epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay )

        self.qnetwork_target = QNetwork( state_size , action_size )

    def act(self, state , deterministic=False ):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        if deterministic or random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn( self , memory ):

        experiences = random.sample( memory , self.batch_size )
        states, actions, rewards, next_states, dones = zip( *experiences )

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork, self.qnetwork_target)

        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def soft_update(self, local_model, target_model, tau=0.001):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Crowd() :

    def __init__( self , Qagent_list , batch_size=64 , epsilon_start=1.0 , epsilon_end=0.01 , epsilon_decay=0.995 ) :
        self.Qagent_list = Qagent_list
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size


    def act( self , state , deterministic=False ) :
        action_list = []
        if deterministic or random.random() > self.epsilon:
            for agent in self.Qagent_list :
                action_list.append( agent.act(state) )
            counts = np.bincount(action_list)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            selected_action = np.random.choice(candidates)
        else:
            selected_action = random.choice( np.arange( self.Qagent_list[0].action_size ) )

        return selected_action
    
    def learn( self , memory ) :
        for agent_i , agent in enumerate( self.Qagent_list ) :
            # print("Updating agent ", agent_i)
            agent.learn( memory )

        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)