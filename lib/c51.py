#!/usr/bin/env python
from __future__ import print_function

import math
import random
from collections import deque
from lib.plot_histogram import PlotHistogramRT
import sys

import numpy as np


class C51Agent:
    def __init__(self, state_size, action_size, num_atoms):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        # self.observe = 2000
        self.observe = 2000
        self.explore = 50000
        self.frame_per_action = 4
        self.update_target_freq = 3000
        self.timestep_per_train = 500  # Number of timesteps between training interval

        # Initialize Atoms
        self.num_atoms = num_atoms  # 51 for C51
        self.v_max = 300  # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        self.v_min = -5  # -0.1*26 - 1 = -3.6
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # Create replay memory using deque
        self.memory = deque()
        self.max_memory = 50000  # number of previous transitions to remember

        # Models for value distribution
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.ep_cum_reward = 0
        self.stats_window_size = 5  # window size for computing rolling statistics
        self.mavg_life = []  # Moving Average of Survival Time
        self.var_life = []  # Variance of Survival Time
        self.mavg_score = []  # Moving Average of Survival Time
        self.var_score = []  # Variance of Survival Time
        self.real_time_plotter = PlotHistogramRT(action_size, num_atoms)


    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            # print("----------Random Action----------")
            action_idx = random.randrange(self.action_size)
        else:
            action_idx = self.get_optimal_action(state)

        return action_idx

    def get_optimal_action(self, state):
        """Get optimal action for a state
        """
        z = self.model.predict(state)  # Return a list [1x51, 1x51, 1x51]
        z_concat = np.vstack(z)

        # plotea la distribucion
        # self.real_time_plotter.update(np.array(self.z), z_concat)

        # esta multiplicacion esta en el articulo
        # es la probabilidad por el valor
        # en este caso regresa algo asi [1, 2, 3]
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)

        # Pick action with the biggest Q value
        action_idx = np.argmax(q)

        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):

        # Check any kill count
        # if (misc[0] > prev_misc[0]):
        #     r_t = r_t + 1
        #
        # if (misc[1] < prev_misc[1]):  # Use ammo
        #     r_t = r_t - 0.1
        #
        if misc['ale.lives'] < prev_misc['ale.lives']:  # Loss HEALTH
            r_t = r_t - 1
        self.ep_cum_reward += r_t
        return r_t

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if t % self.update_target_freq == 0:
            self.update_target_model()

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        # print ("=num_samples")
        # print (num_samples)
        # print("=memory shape")
        # print (self.memory[0][0].shape)

        state_inputs = np.zeros(((num_samples,) + self.state_size))
        next_states = np.zeros(((num_samples,) + self.state_size))
        m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(self.action_size)]
        action, reward, done = [], [], []

        # print("=state inputs")
        # print (state_inputs.shape)
        # print("=next states")
        # print (next_states.shape)
        # print("=m prob")
        # print (m_prob[0].shape)

        for i in range(num_samples):
            state_inputs[i, :, :, :] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            next_states[i, :, :, :] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        z = self.model.predict(next_states)  # Return a list [32x51, 32x51, 32x51]
        z_ = self.target_model.predict(next_states)  # Return a list [32x51, 32x51, 32x51]

        # print("=z")
        # print(np.asarray(z).shape)

        # Get Optimal Actions for the next states (from distribution z)
        optimal_action_idxs = []
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)  # length (num_atoms x num_actions)

        # print("=q1")
        # print(q.shape)

        q = q.reshape((num_samples, self.action_size), order='F')
        optimal_action_idxs = np.argmax(q, axis=1)

        # print("=z_concat")
        # print(z_concat.shape)
        # print("=q2")
        # print(q.shape)
        # print("=optimal_action_idxs")
        # print (optimal_action_idxs)

        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            if done[i]:  # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][i][int(m_l)] += (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                    m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)

        loss = self.model.fit(state_inputs, m_prob, batch_size=self.batch_size, nb_epoch=1, verbose=0)

        return loss.history['loss']

    # load the saved model
    def load_model(self, name):
        print ("loading model... ")
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)