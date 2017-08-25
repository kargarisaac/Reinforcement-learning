import numpy as np
from flat_game import carmunk
import random
import timeit
from collections import deque
import keras
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import Callback
from keras.optimizers import Adam, RMSprop


class DDQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see learning process, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 1000
        self.buffer = 5000
        # create replay memory using deque
        self.memory = deque(maxlen=self.buffer)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("saved_models_ddqn/model600000.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(self.state_size,), activation='relu',init='lecun_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu', init='lecun_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear',init='lecun_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state, batch_size=1)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon -= (1 / train_frames)

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][np.argmax(target_next[i])])

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size, nb_epoch=1, verbose=0)


if __name__== "__main__":
    # create a new game instance
    env = carmunk.GameState()

    state_size = 3
    action_size = 3
    train_frames = 1000000
    # observe = 500  # Number of frames to observe before training.
    agent = DDQNAgent(state_size, action_size)

    # scores, episodes = [], []
    if not agent.load_model:
        done = False
        score = 0
        # get initial state by doing nothing and getting the state
        _, state = env.frame_step(2)
        # state = np.reshape(state, [1, state_size])
        start_time = timeit.default_timer()

        t = 0
        while t < train_frames:
            t += 1

            action = agent.get_action(state)
            reward, new_state = env.frame_step(action)
            score += 1
            # new_state = np.reshape(new_state, [1, state_size])
            if reward == -500:
                done = True
            else:
                done = False
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, new_state, done)

            agent.train_model()

            state = new_state

            if done:  #if done
                agent.update_target_model()
                tot_time = timeit.default_timer() - start_time
                fps = score / tot_time
                print("n_frames: %d, fps: %d, epsilon: %f" % (t, fps, agent.epsilon))
                #reset
                score = 0
                start_time = timeit.default_timer()

            if t % 20000 == 0:
                agent.model.save_weights('saved_models_ddqn/model'+str(t)+'.h5', overwrite=True)

    else:
        score = 0
        _, state = env.frame_step(2)

        while True:

            # choose action
            action = np.argmax(agent.model.predict(state, batch_size=1))

            # take action
            reward, state = env.frame_step(action)

            #update score
            score += 1
