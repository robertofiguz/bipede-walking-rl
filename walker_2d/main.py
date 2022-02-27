import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import walker
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
# Configuration paramaters for the whole setup
seed = 42
gamma = 0.95  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 20  # Size of batch taken from replay buffer
max_steps_per_episode = 200

env = walker.Walker()


states = env.observation_space.shape
actions = 8*3


def build_model(states):
        # Define model layers.
    input_layer = Input(1,8)
    first_dense = Dense(units='128', activation='relu')(input_layer)
    # Y1 output will be fed from the first dense
    y1_output = Dense(units='3', name='motor_1')(first_dense)

    second_dense = Dense(units='128',activation='relu')(first_dense)
    # Y2 output will be fed from the second dense
    y2_output = Dense(units='3',name='motor_2')(second_dense)

    third_dense = Dense(units='128',activation='relu')(second_dense)
    # Y2 output will be fed from the second dense
    y3_output = Dense(units='3',name='motor_3')(third_dense)

    four_dense = Dense(units='128',activation='relu')(third_dense)
    # Y2 output will be fed from the second dense
    y4_output = Dense(units='3',name='motor_4')(four_dense)

    five_dense = Dense(units='128',activation='relu')(four_dense)
    # Y2 output will be fed from the second dense
    y5_output = Dense(units='3',name='motor_5')(five_dense)

    six_dense = Dense(units='128',activation='relu')(five_dense)
    # Y2 output will be fed from the second dense
    y6_output = Dense(units='3',name='motor_6')(six_dense)

    seven_dense = Dense(units='128',activation='relu')(six_dense)
    # Y2 output will be fed from the second dense
    y7_output = Dense(units='3',name='motor_7')(seven_dense)

    eight_dense = Dense(units='128',activation='relu')(seven_dense)
    # Y2 output will be fed from the second dense
    y8_output = Dense(units='3',name='motor_8')(eight_dense)

    # Define the model with the input layer 
    # and a list of output layers
    model = Model(inputs=input_layer,outputs=[y1_output, y2_output, y3_output,y4_output,y5_output,y6_output,y7_output,y8_output])

    return model

# The first model makes the predictions for Q-values which are used to
# make a action.
model = build_model(states)
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = build_model(states)
model.summary()


import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import gym
from gym import logger
import wandb
from wandb.keras import WandbCallback

run = wandb.init(
    config={
        "gamma": 0.99, 
        "epsilon": 1,
        "epsilon_min": 0.1,
        "target_reward": 400.0,
        "batch_size": 64,
        "win_trials": 100,
        "units": 256,
        "learning_rate": 0.001
        },
    project="cartpole-v2")


class DQNAgent:
    def __init__(self,
                 state_space, 
                 action_space, 
                 episodes=500):
        """DQN Agent on CartPole-v0 environment

        Arguments:
            state_space (tensor): state space
            action_space (tensor): action space
            episodes (int): number of episodes to train
        """
        self.action_space = action_space

        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = run.config.gamma

        # initially 90% exploration, 10% exploitation
        self.epsilon = run.config.epsilon
        # iteratively applying decay til 
        # 10% exploration/90% exploitation
        self.epsilon_min = run.config.epsilon_min
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** \
                             (1. / float(episodes))

        # Q Network weights filename
        self.weights_file = 'dqn_cartpole.h5'
        # Q Network for training
        n_inputs = state_space.shape[0]
        n_outputs = env.observation_space
  
                
        self.q_model = build_model(n_inputs)
        self.q_model.compile(loss='mae', optimizer=Adam(learning_rate=run.config.learning_rate))
        # target Q Network
        self.target_q_model = build_model(n_inputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0

    
    


    def save_weights(self):
        """save Q Network params to a file"""
        self.q_model.save_weights(self.weights_file)


    def update_weights(self):
        """copy trained Q Network params to target Q Network"""
        self.target_q_model.set_weights(self.q_model.get_weights())


    def act(self, state):
        """eps-greedy policy
        Return:
            action (tensor): action to execute
        """
        run.log({"epsilon":self.epsilon})
        if np.random.rand() < self.epsilon or episode_count<20:
            # explore - do random action
            return self.action_space.sample()
        # exploit
        print("\n\npredicting\n\n")
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        action = np.argmax(q_values[0])
        return action


    def remember(self, state, action, reward, next_state, done):
        """store experiences in the replay buffer
        Arguments:
            state (tensor): env state
            action (tensor): agent action
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        """
        item = (state, action, reward, next_state, done)
        self.memory.append(item)


    def get_target_q_value(self, next_state, reward):
        """compute Q_max
           Use of target Q Network solves the 
            non-stationarity problem
        Arguments:
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        Return:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DQN chooses the max Q value among next actions
        # selection and evaluation of action is 
        # on the target Q Network
        # Q_max = max_a' Q_target(s', a')
        q_value = np.amax(\
                     self.target_q_model.predict(next_state)[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value


    def replay(self, batch_size):
        """experience replay addresses the correlation issue 
            between samples
        Arguments:
            batch_size (int): replay buffer batch 
                sample size
        """
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)
            
            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         verbose=0,
                         epochs=1,
                         callbacks=WandbCallback())
                

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after 
        # every 10 training updates
        if self.replay_counter % 10 == 0:
            self.update_weights()

        self.replay_counter += 1

    
    def update_epsilon(self):
        """decrease the exploration, increase exploitation"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
if __name__ == '__main__':
    

    # the number of trials without falling over
    win_trials = run.config.win_trials
    # the CartPole-v0 is considered solved if 
    # for 100 consecutive trials, he cart pole has not 
    # fallen over and it has achieved an average 
    # reward of 195.0 
    # a reward of +1 is provided for every timestep 
    # the pole remains upright
    win_reward = { 'CartPole-v1' : run.config.target_reward }

    # stores the reward per episode
    scores = deque(maxlen=win_trials)
    rewards_history_full = []
    logger.setLevel(logger.ERROR)
    env = walker.Walker()

    #env.seed(0)

    # instantiate the DQN/DDQN agent

    agent = DQNAgent(env.observation_space, env.action_space)

    # should be solved in this number of episodes
    episode_count = 3000
    state_size = env.observation_space.shape[0]
    batch_size = run.config.batch_size

    # by default, CartPole-v1 has max episode steps = 500
    # you can use this to experiment beyond 500
    # env._max_episode_steps = 4000

    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        while not done:
            # in CartPole-v0, action=0 is left and action=1 is right
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #env.render()
            # in CartPole-v0:
            # state = [pos, vel, theta, angular speed]
            next_state = np.reshape(next_state, [1, state_size])
            # store every experience unit in replay buffer
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        rewards_history_full.append(total_reward)
        # call experience relay
        if len(agent.memory) >= batch_size:
            #agent.replay(batch_size)
            pass
        scores.append(total_reward)
        mean_score = np.mean(scores)
        run.log({"episode_score": mean_score})
        if mean_score >= win_reward["CartPole-v1"] \
                and episode >= win_trials:
            print("Solved in episode %d: \
                   Mean survival = %0.2lf in %d episodes"
                  % (episode, mean_score, win_trials))
            print("Epsilon: ", agent.epsilon)
            agent.save_weights()
            break
        if (episode + 1) % win_trials == 0:
            print("Episode %d: Mean survival = \
                   %0.2lf in %d episodes" %
                  ((episode + 1), mean_score, win_trials))

    # close the env and write monitor result info to disk
  #  artifact = wandb.Artifact("weights_v1", "weights")
  #  artifact.add_file("dqn_cartpole.h5")
  #  run.log_artifact(artifact)
    run.finish()
    env.close() 