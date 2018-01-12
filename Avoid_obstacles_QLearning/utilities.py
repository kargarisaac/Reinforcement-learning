import numpy as np
import pandas as pd
np.random.seed(1)
import tkinter as tk
import time


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=1.0):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        # in self.epsilon % of times do the action correctly and other times act randomly
        # action selection
        if np.random.uniform() < self.epsilon:
            
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            
         
        
class Maze(tk.Tk, object):
    def __init__(self, MAZE_H, MAZE_W, UNIT):
        super(Maze, self).__init__()
        self.MAZE_H = MAZE_H
        self.MAZE_W = MAZE_W
        self.UNIT = UNIT
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('world')
        self.geometry('{0}x{1}'.format(self.MAZE_H * self.UNIT, self.MAZE_H * self.UNIT))
        self._build_maze()
        
        
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=self.MAZE_H * self.UNIT,
                           width=self.MAZE_W * self.UNIT)

        # create grids
        for c in range(0, self.MAZE_W * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = c, 0, c, self.MAZE_H * self.UNIT
            self.canvas.create_line(x0, y0, x1, y1)
            
        for r in range(0, self.MAZE_H * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = 0, r, self.MAZE_W * self.UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # hell
        hell1 = [2*self.UNIT, 2*self.UNIT, 3*self.UNIT, 3*self.UNIT] 
        self.hell1 = self.canvas.create_rectangle(hell1[0], hell1[1], hell1[2], hell1[3], fill='black')
        
        # hell
        hell2 = [5*self.UNIT, 2*self.UNIT, 6*self.UNIT, 3*self.UNIT] 
        self.hell2 = self.canvas.create_rectangle(hell2[0], hell2[1], hell2[2], hell2[3], fill='black')
        
        # hell
        hell3 = [8*self.UNIT, 2*self.UNIT, 9*self.UNIT, 3*self.UNIT] 
        self.hell3 = self.canvas.create_rectangle(hell3[0], hell3[1], hell3[2], hell3[3], fill='black')
        
        # hell
        hell4 = [2*self.UNIT, 5*self.UNIT, 3*self.UNIT, 6*self.UNIT] 
        self.hell4 = self.canvas.create_rectangle(hell4[0], hell4[1], hell4[2], hell4[3], fill='black')
        
        # hell
        hell5 = [5*self.UNIT, 5*self.UNIT, 6*self.UNIT, 6*self.UNIT] 
        self.hell5 = self.canvas.create_rectangle(hell5[0], hell5[1], hell5[2], hell5[3], fill='black')
        
        # hell
        hell6 = [8*self.UNIT, 5*self.UNIT, 9*self.UNIT, 6*self.UNIT] 
        self.hell6 = self.canvas.create_rectangle(hell6[0], hell6[1], hell6[2], hell6[3], fill='black')
        
        # create oval
        oval = [10*self.UNIT, 5*self.UNIT]
        self.oval = self.canvas.create_oval(
            oval[0], oval[1],
            oval[0] + self.UNIT, oval[1] + self.UNIT,
            fill='yellow')

        # create red rect
        origin = np.array([1*self.UNIT, 8*self.UNIT])
        self.rect = self.canvas.create_rectangle(
            origin[0], origin[1],
            origin[0] + self.UNIT, origin[1] + self.UNIT,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([1*self.UNIT, 8*self.UNIT])
        self.rect = self.canvas.create_rectangle(
            origin[0], origin[1],
            origin[0] + self.UNIT, origin[1] + self.UNIT,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > self.UNIT:
                base_action[1] -= self.UNIT
        elif action == 1:   # down
            if s[1] < (self.MAZE_H - 1) * self.UNIT:
                base_action[1] += self.UNIT
        elif action == 2:   # right
            if s[0] < (self.MAZE_W - 1) * self.UNIT:
                base_action[0] += self.UNIT
        elif action == 3:   # left
            if s[0] > self.UNIT:
                base_action[0] -= self.UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state
        
        # reward function
        if s_ == self.canvas.coords(self.oval): # if reach the goal -> r=1
            reward = 0
            done = False
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3), self.canvas.coords(self.hell4), self.canvas.coords(self.hell5), self.canvas.coords(self.hell6)]:  # else if get to obstacles -> r=-1
            reward = -1
            done = True
        else: 
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

