
# from google.colab import drive
# drive.mount('/content/drive/')
base_path = "./"

import os
import sys
import gym
import math
import wandb
import pymunk
import base64
import pygame
import imageio
import IPython
import numpy as np
import pyvirtualdisplay
import tensorflow as tf
import pymunk.pygame_util
from gym import Env
from PIL import Image
from gym import logger
from numpy import append
from tensorflow import keras
from collections import deque
# from google.colab import auth
from wandb.keras import WandbCallback
from google.cloud import secretmanager
from gym.spaces import Box, MultiDiscrete
from tensorflow.keras.models import Model
from random import randint, random, sample
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Input

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print(
        '\n\nThis error most likely means that this notebook is not '
        'configured to use a GPU.  Change this in Notebook Settings via the '
        'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
    raise SystemError('GPU device not found')

auth.authenticate_user()
client = secretmanager.SecretManagerServiceClient()
secret_name = "wandb-key"  # => To be replaced with your secret name
project_id = '786036037251'  # => To be replaced with your GCP Project
resource_name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
response = client.access_secret_version(request={"name": resource_name})
wandb.login(key=response.payload.data.decode('UTF-8'))

display = pyvirtualdisplay.Display(visible=0, size=(1904, 960)).start()

"""# Hyperparameters"""

# Configuration paramaters for the whole setup
seed = 42  # @param {type:"integer"}
gamma = 0.99  # @param {type:"number"} # Discount factor for past rewards
epsilon = 0.1  # @param {type:"number"} # Epsilon greedy parameter
epsilon_min = 0.01  # @param {type:"number"} # Minimum epsilon greedy parameter
epsilon_max = 0.1  # @param {type:"number"} # Maximum epsilon greedy parameter
epsilon_interval = (
        epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 64  # @param {type:"integer"} # Size of batch taken from replay buffer
units = 128  # @param{type:"integer"}
learning_rate = 0.01  # @param{type:"number"}
target_reward = 0  # @param{type:"number"}
win_trials = 100  # @param{type:"integer"}
nr_episodes = 100000  # @param{type:"integer"}
loss = "mae"  # @param{type:"string"}
session_name = "softmax_output_activation_update_target"  # @param{type:"string"}

run = wandb.init(
    config={
        "gamma": gamma,
        "epsilon": epsilon,
        "epsilon_min": epsilon_min,
        "target_reward": target_reward,
        "batch_size": batch_size,
        "win_trials": win_trials,
        "units": units,
        "learning_rate": learning_rate,
        "loss": loss,
        "nr_episodes": nr_episodes
    },
    project="walker-v10", id=session_name)
base_path += f"{run.project}-{run.name}/"
os.makedirs(base_path, exist_ok=True)

"""# Walker"""

screen_width = 1904
screen_height = 960


class Robot():
    def __init__(self, space):

        self.tick = 0

        moment = 10
        friction = 0.6

        self.shape = pymunk.Poly.create_box(None, (50, 100))
        body_moment = pymunk.moment_for_poly(moment, self.shape.get_vertices())
        self.body = pymunk.Body(moment, body_moment)

        self.body.position = (200, 350)
        self.shape.body = self.body
        self.shape.color = (150, 150, 150, 0)

        head_moment = pymunk.moment_for_circle(moment, 0, 30)
        self.head_body = pymunk.Body(moment, head_moment)
        self.head_body.position = (self.body.position.x, self.body.position.y + 80)
        self.head_shape = pymunk.Circle(self.head_body, 30)
        self.head_shape.friction = friction
        self.head_joint = pymunk.PivotJoint(self.head_body, self.body, (-5, -30), (-5, 50))
        self.head_joint2 = pymunk.PivotJoint(self.head_body, self.body, (5, -30), (5, 50))

        arm_size = (100, 20)
        self.left_arm_upper_shape = pymunk.Poly.create_box(None, arm_size)
        left_arm_upper_moment = pymunk.moment_for_poly(moment, self.left_arm_upper_shape.get_vertices())
        self.left_arm_upper_body = pymunk.Body(moment, left_arm_upper_moment)
        self.left_arm_upper_body.position = (self.body.position.x - 70, self.body.position.y + 30)
        self.left_arm_upper_shape.body = self.left_arm_upper_body
        self.left_arm_upper_joint = pymunk.PivotJoint(self.left_arm_upper_body, self.body, (arm_size[0] / 2, 0),
                                                      (-25, 30))
        self.la_motor = pymunk.SimpleMotor(self.body, self.left_arm_upper_body, 0)

        self.right_arm_upper_shape = pymunk.Poly.create_box(None, arm_size)
        right_arm_upper_moment = pymunk.moment_for_poly(moment, self.right_arm_upper_shape.get_vertices())
        self.right_arm_upper_body = pymunk.Body(moment, right_arm_upper_moment)
        self.right_arm_upper_body.position = (self.body.position.x + 70, self.body.position.y + 30)
        self.right_arm_upper_shape.body = self.right_arm_upper_body
        self.right_arm_upper_joint = pymunk.PivotJoint(self.right_arm_upper_body, self.body, (-arm_size[0] / 2, 0),
                                                       (25, 30))
        self.ra_motor = pymunk.SimpleMotor(self.body, self.right_arm_upper_body, 0)

        thigh_size = (30, 60)
        self.lu_shape = pymunk.Poly.create_box(None, thigh_size)
        lu_moment = pymunk.moment_for_poly(moment, self.lu_shape.get_vertices())
        self.lu_body = pymunk.Body(moment, lu_moment)
        self.lu_body.position = (self.body.position.x - 20, self.body.position.y - 75)
        self.lu_shape.body = self.lu_body
        self.lu_shape.friction = friction
        self.lu_joint = pymunk.PivotJoint(self.lu_body, self.body, (0, thigh_size[1] / 2), (-20, -50))
        self.lu_motor = pymunk.SimpleMotor(self.body, self.lu_body, 0)

        self.ru_shape = pymunk.Poly.create_box(None, thigh_size)
        ru_moment = pymunk.moment_for_poly(moment, self.ru_shape.get_vertices())
        self.ru_body = pymunk.Body(moment, ru_moment)
        self.ru_body.position = (self.body.position.x + 20, self.body.position.y - 75)
        self.ru_shape.body = self.ru_body
        self.ru_shape.friction = friction
        self.ru_joint = pymunk.PivotJoint(self.ru_body, self.body, (0, thigh_size[1] / 2), (20, -50))
        self.ru_motor = pymunk.SimpleMotor(self.body, self.ru_body, 0)

        leg_size = (20, 70)
        self.ld_shape = pymunk.Poly.create_box(None, leg_size)
        ld_moment = pymunk.moment_for_poly(moment, self.ld_shape.get_vertices())
        self.ld_body = pymunk.Body(moment, ld_moment)
        self.ld_body.position = (self.lu_body.position.x, self.lu_body.position.y - 65)
        self.ld_shape.body = self.ld_body
        self.ld_shape.friction = friction
        self.ld_joint = pymunk.PivotJoint(self.ld_body, self.lu_body, (0, leg_size[1] / 2), (0, -thigh_size[1] / 2))
        self.ld_motor = pymunk.SimpleMotor(self.lu_body, self.ld_body, 0)

        self.rd_shape = pymunk.Poly.create_box(None, leg_size)
        rd_moment = pymunk.moment_for_poly(moment, self.rd_shape.get_vertices())
        self.rd_body = pymunk.Body(moment, rd_moment)
        self.rd_body.position = (self.ru_body.position.x, self.ru_body.position.y - 65)
        self.rd_shape.body = self.rd_body
        self.rd_shape.friction = friction
        self.rd_joint = pymunk.PivotJoint(self.rd_body, self.ru_body, (0, leg_size[1] / 2), (0, -thigh_size[1] / 2))
        self.rd_motor = pymunk.SimpleMotor(self.ru_body, self.rd_body, 0)

        foot_size = (45, 20)
        self.lf_shape = pymunk.Poly.create_box(None, foot_size)
        rd_moment = pymunk.moment_for_poly(moment, self.lf_shape.get_vertices())
        self.lf_body = pymunk.Body(moment, rd_moment)
        self.lf_body.position = (
        self.ld_body.position.x + foot_size[0] / 6, self.ld_body.position.y - (foot_size[1] * 2))
        self.lf_shape.body = self.lf_body
        self.lf_shape.friction = friction
        self.lf_shape.elasticity = 0.1
        self.lf_joint = pymunk.PivotJoint(self.ld_body, self.lf_body, (-5, -leg_size[1] / 2),
                                          (-foot_size[0] / 2 + 10, foot_size[1] / 2))
        self.lf_motor = pymunk.SimpleMotor(self.ld_body, self.lf_body, 0)

        self.rf_shape = pymunk.Poly.create_box(None, foot_size)
        rd_moment = pymunk.moment_for_poly(moment, self.rf_shape.get_vertices())
        self.rf_body = pymunk.Body(moment, rd_moment)
        self.rf_body.position = (
        self.rd_body.position.x + foot_size[0] / 6, self.rd_body.position.y - (foot_size[1] * 2))
        self.rf_shape.body = self.rf_body
        self.rf_shape.friction = friction
        self.rf_shape.elasticity = 0.1
        self.rf_joint = pymunk.PivotJoint(self.rd_body, self.rf_body, (-5, -leg_size[1] / 2),
                                          (-foot_size[0] / 2 + 10, foot_size[1] / 2))
        self.rf_motor = pymunk.SimpleMotor(self.rd_body, self.rf_body, 0)

        space.add(self.body, self.shape, self.head_body, self.head_shape, self.head_joint, self.head_joint2)
        space.add(self.left_arm_upper_body, self.left_arm_upper_shape, self.left_arm_upper_joint, self.la_motor)
        space.add(self.right_arm_upper_body, self.right_arm_upper_shape, self.right_arm_upper_joint, self.ra_motor)
        space.add(self.lu_body, self.lu_shape, self.lu_joint, self.lu_motor)
        space.add(self.ru_body, self.ru_shape, self.ru_joint, self.ru_motor)
        space.add(self.ld_body, self.ld_shape, self.ld_joint, self.ld_motor)
        space.add(self.rd_body, self.rd_shape, self.rd_joint, self.rd_motor)
        space.add(self.lf_body, self.lf_shape, self.lf_joint, self.lf_motor)
        space.add(self.rf_body, self.rf_shape, self.rf_joint, self.rf_motor)

        shape_filter = pymunk.ShapeFilter(group=1)
        self.shape.filter = shape_filter
        self.head_shape.filter = shape_filter
        self.left_arm_upper_shape.filter = shape_filter
        self.right_arm_upper_shape.filter = shape_filter
        self.lu_shape.filter = shape_filter
        self.ru_shape.filter = shape_filter
        self.ld_shape.filter = shape_filter
        self.rd_shape.filter = shape_filter
        self.lf_shape.filter = shape_filter
        self.rf_shape.filter = shape_filter

        self.lu_flag = False
        self.ld_flag = False
        self.ru_flag = False
        self.rd_flag = False
        self.la_flag = False
        self.ra_flag = False
        self.lf_flag = False
        self.rf_flag = False

    def get_data(self):
        lu = ((360 - math.degrees(self.lu_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        ld = ((360 - math.degrees(self.ld_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        lf = ((360 - math.degrees(self.lf_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        ru = ((360 - math.degrees(self.ru_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        rd = ((360 - math.degrees(self.rd_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        rf = ((360 - math.degrees(self.rf_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        la = ((360 - math.degrees(self.left_arm_upper_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        ra = ((360 - math.degrees(self.right_arm_upper_body.angle)) - (360 - math.degrees(self.body.angle))) / 360.0
        return ru, rd, lu, ld, la, ra, lf, rf
        # removed self.body,angle

    def update(self):
        # lu
        self.lu_flag = False
        if (360 - math.degrees(self.lu_body.angle)) - (
                360 - math.degrees(self.body.angle)) >= 90 and self.lu_motor.rate > 0:
            self.lu_motor.rate = 0
            self.lu_flag = True
        elif (360 - math.degrees(self.lu_body.angle)) - (
                360 - math.degrees(self.body.angle)) <= -90 and self.lu_motor.rate < 0:
            self.lu_motor.rate = 0
            self.lu_flag = True

        # ld
        self.ld_flag = False
        if (360 - math.degrees(self.ld_body.angle)) - (
                360 - math.degrees(self.lu_body.angle)) >= 90 and self.ld_motor.rate > 0:
            self.ld_motor.rate = 0
            self.ld_flag = True
        elif (360 - math.degrees(self.ld_body.angle)) - (
                360 - math.degrees(self.lu_body.angle)) <= -90 and self.ld_motor.rate < 0:
            self.ld_motor.rate = 0
            self.ld_flag = True

        # ru
        self.ru_flag = False
        if (360 - math.degrees(self.ru_body.angle)) - (
                360 - math.degrees(self.body.angle)) >= 90 and self.ru_motor.rate > 0:
            self.ru_motor.rate = 0
            self.ru_flag = True
        elif (360 - math.degrees(self.ru_body.angle)) - (
                360 - math.degrees(self.body.angle)) <= -90 and self.ru_motor.rate < 0:
            self.ru_motor.rate = 0
            self.ru_flag = True

        # rd
        self.rd_flag = False
        if (360 - math.degrees(self.rd_body.angle)) - (
                360 - math.degrees(self.ru_body.angle)) >= 90 and self.rd_motor.rate > 0:
            self.rd_motor.rate = 0
            self.rd_flag = True
        elif (360 - math.degrees(self.rd_body.angle)) - (
                360 - math.degrees(self.ru_body.angle)) <= -90 and self.rd_motor.rate < 0:
            self.rd_motor.rate = 0
            self.rd_flag = True

        # lf
        self.lf_flag = False
        if (360 - math.degrees(self.lf_body.angle)) - (
                360 - math.degrees(self.ld_body.angle)) >= 90 and self.lf_motor.rate > 0:
            self.lf_motor.rate = 0
            self.lf_flag = True
        elif (360 - math.degrees(self.lf_body.angle)) - (
                360 - math.degrees(self.ld_body.angle)) <= -45 and self.lf_motor.rate < 0:
            self.lf_motor.rate = 0
            self.lf_flag = True

        # rf
        self.rf_flag = False
        if (360 - math.degrees(self.rf_body.angle)) - (
                360 - math.degrees(self.rd_body.angle)) >= 90 and self.rf_motor.rate > 0:
            self.rf_motor.rate = 0
            self.rf_flag = True
        elif (360 - math.degrees(self.rf_body.angle)) - (
                360 - math.degrees(self.rd_body.angle)) <= -45 and self.rf_motor.rate < 0:
            self.rf_motor.rate = 0
            self.rf_flag = True

    def add_land(self, space):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (0, 100)
        land = pymunk.Segment(body, (0, 50), (99999, 50), 10)
        land.friction = 1.0
        land.elasticity = 0.1
        space.add(body, land)

        body_2 = pymunk.Body(body_type=pymunk.Body.STATIC)
        body_2.position = (300, -50)
        t_block = pymunk.Segment(body_2, (0, 100), (20, 100), 10)
        space.add(body_2, t_block)


class Walker(Env):
    def __init__(self):
        self.action_space = MultiDiscrete([3] * 8)
        self.observation_space = Box(-20, 20, [8])
        self.viewer = None
        self.last_horizontal_pos = 0
        self.last_vertical_pos = 0

    def check_fall(self):
        if self.robot.body.position[1] < self.initial_height - 50:
            return True
        if self.robot.body.position[0] < 0 or self.robot.body.position[0] > screen_width:
            return True
        return False

    def calculate_reward(self):
        shape = env.space.shapes[-2]
        contact_lf = len(env.robot.lf_shape.shapes_collide(b=shape).points)
        contact_rf = len(env.robot.rf_shape.shapes_collide(b=shape).points)

        if (self.robot.body.position[0] - self.last_horizontal_pos) > 1:
            reward = 0
        elif 1 > (self.robot.body.position[0] - self.last_horizontal_pos) > -1:
            reward = -100
        elif (self.robot.body.position[0] - self.last_horizontal_pos) < -1:
            reward = -200
        if not contact_lf and not contact_rf:
            reward -= 50
        return -1  # set to only get reward if reaches target

    def check_complete(self):
        if self.robot.body.position[0] > 300:  # 500 is the position of the target
            return True

    def step(self, actions):
        actions = [(a - 1) * 2 for a in actions]
        self.robot.ru_motor.rate = actions[0]
        self.robot.rd_motor.rate = actions[1]
        self.robot.lu_motor.rate = actions[2]
        self.robot.ld_motor.rate = actions[3]
        self.robot.la_motor.rate = actions[4]
        self.robot.ra_motor.rate = actions[5]
        self.robot.lf_motor.rate = actions[6]
        self.robot.rf_motor.rate = actions[7]

        self.robot.update()
        self.space.step(1 / 50)

        self.robot.ru_motor.rate = 0
        self.robot.rd_motor.rate = 0
        self.robot.lu_motor.rate = 0
        self.robot.ld_motor.rate = 0
        self.robot.la_motor.rate = 0
        self.robot.ra_motor.rate = 0
        self.robot.lf_motor.rate = 0
        self.robot.rf_motor.rate = 0

        done = False
        reward = self.calculate_reward()

        if self.check_fall():
            done = True
            reward = (1 / (1 - run.config.gamma)) * -1  # -200 reresents the highest penalty

        if self.check_complete():
            done = True
            reward = 100

        info = {}
        observation = self.robot.get_data()

        self.last_horizontal_pos = self.robot.body.position[0]
        self.last_vertical_pos = self.robot.body.position[1]

        return (
            observation,
            reward,
            done,
            info)

    def render(self):
        if self.viewer is None:
            self.viewer = pygame.init()
            pymunk.pygame_util.positive_y_is_up = True
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(25)
        return pygame.surfarray.array3d(self.screen)

    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -990)
        self.robot = Robot(self.space)
        self.robot.add_land(self.space)
        self.initial_height = self.robot.body.position[1]
        self.initial_horizontal = self.robot.body.position[0]
        observation = self.robot.get_data()
        return (observation)


"""# Model"""


def build_model():
    # Define model layers.
    units = run.config.units
    input_layer = Input(8, 64)
    first_dense = Dense(units=units, activation='relu')(input_layer)
    second_dense = Dense(units=units, activation='relu')(first_dense)
    third_dense = Dense(units=units, activation='relu')(second_dense)
    fourth_dense = Dense(units=units, activation='relu')(third_dense)
    fifth_dense = Dense(units=units, activation='relu')(fourth_dense)
    six_dense = Dense(units=units, activation='relu')(fifth_dense)
    seven_dense = Dense(units=units, activation='relu')(six_dense)
    # Y1 output will be fed from the first dense
    y1_output = Dense(units='3', name='motor_1', activation="softmax")(seven_dense)

    y2_output = Dense(units='3', name='motor_2', activation="softmax")(seven_dense)

    y3_output = Dense(units='3', name='motor_3', activation="softmax")(seven_dense)

    y4_output = Dense(units='3', name='motor_4', activation="softmax")(seven_dense)

    y5_output = Dense(units='3', name='motor_5', activation="softmax")(seven_dense)

    y6_output = Dense(units='3', name='motor_6', activation="softmax")(seven_dense)

    y7_output = Dense(units='3', name='motor_7', activation="softmax")(seven_dense)

    y8_output = Dense(units='3', name='motor_8', activation="softmax")(seven_dense)

    # Define the model with the input layer 
    # and a list of output layers
    model = Model(inputs=input_layer,
                  outputs=[y1_output, y2_output, y3_output, y4_output, y5_output, y6_output, y7_output, y8_output])

    return model


"""# DQN Agent"""


class DQNAgent:
    def __init__(self,
                 state_space,
                 action_space,
                 episodes=500,
                 weights=None):
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
        self.q_model = build_model()
        self.q_model.compile(loss=run.config.loss, optimizer=Adam(learning_rate=run.config.learning_rate))
        self.q_model.summary()
        if weights != None:
            self.load_weights(weights)
        # target Q Network
        self.target_q_model = build_model()
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0

    def save_weights(self, episode):
        """save Q Network params to a file"""
        self.q_model.save_weights(f'{base_path}{episode}-steps.h5')

    def update_weights(self):
        """copy trained Q Network params to target Q Network"""
        self.target_q_model.set_weights(self.q_model.get_weights())

    def load_weights(self, path):
        self.q_model.load_weights(path)

    def act(self, state):
        """eps-greedy policy
        Return:
            action (tensor): action to execute
        """

        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()
        # exploit
        state = np.expand_dims(state, 0)

        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        action = np.argmax(q_values, axis=2).flatten()
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
        next_state = np.expand_dims(next_state, 0)
        q_values = self.target_q_model.predict(next_state)
        q_value = np.amax(self.target_q_model.predict(next_state), axis=2)

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
        sars_batch = sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            state = np.expand_dims(state, 0)
            q_values = self.q_model.predict(state)

            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)
            # correction on the Q value for the action used
            for i in q_values:
                i[0][action] = [reward] * 8 if done else q_value.flatten()
            # q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         verbose=0,
                         epochs=1,
                         callbacks=[WandbCallback()])

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


"""# Video"""


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def create_policy_eval_video(filename, num_episodes=1, fps=25):
    filename = base_path + filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            state = env.reset()
            video.append_data(env.render())
            done = False
            steps = 0
            while not done and steps < 900:
                steps += 1
                state = np.expand_dims(state, 0)
                q_values = agent.q_model.predict(state)
                action = np.argmax(q_values, axis=2).flatten()
                state, reward, done, _ = env.step(action)
                video.append_data(env.render())
    run.log(
        {"video": wandb.Video(filename, fps=30, format="mp4")})


"""# Get Weights"""

weights_file = wandb.restore('599-steps.h5', "sudofork/walker-v2/16y25q39")
weights = weights_file.name

"""# Train"""

# Commented out IPython magic to ensure Python compatibility.
if __name__ == '__main__':

    # the number of trials without falling over
    win_trials = run.config.win_trials
    # the CartPole-v0 is considered solved if 
    # for 100 consecutive trials, he cart pole has not 
    # fallen over and it has achieved an average 
    # reward of 195.0 
    # a reward of +1 is provided for every timestep 
    # the pole remains upright

    # stores the reward per episode
    scores = deque(maxlen=win_trials)
    rewards_history_full = []
    logger.setLevel(logger.ERROR)
    env = Walker()

    # env.seed(0)

    # instantiate the DQN/DDQN agent
    # weights = "/content/drive/MyDrive/Final_Project/walker-v1-likely-donkey-30/100-steps.h5"

    agent = DQNAgent(env.observation_space, env.action_space, run.config.nr_episodes, weights=None)

    # should be solved in this number of episodes
    state_size = env.observation_space.shape[0]
    batch_size = run.config.batch_size

    # by default, CartPole-v1 has max episode steps = 500
    # you can use this to experiment beyond 500
    # env._max_episode_steps = 4000

    # Q-Learning sampling and fitting
    for episode in range(run.config.nr_episodes):
        state = env.reset()
        state = state
        done = False
        total_reward = 0
        step_count = 0
        reward_history = []
        while not done:
            # in CartPole-v0, action=0 is left and action=1 is right
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if step_count > 900:
                done = True
            # env.render()
            # in CartPole-v0:
            # state = [pos, vel, theta, angular speed]
            next_state = next_state
            # store every experience unit in replay buffer
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            reward_history.append(reward)
        rewards_history_full.append(total_reward)
        # call experience relay
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)
        scores.append(total_reward)
        mean_score = np.mean(scores)
        run.log({
            "reward": total_reward,
            "epsilon": agent.epsilon,
            "average reward": np.average(reward_history),
            "mean score": mean_score,
            "minimum reward": np.min(reward_history),
            "max reward": np.max(reward_history),

        })

        if mean_score >= run.config.target_reward \
                and episode >= win_trials:
            print("Solved in episode %d: Mean survival = %0.2lf in %d episodes" % (episode, mean_score, win_trials))
            print("Epsilon: ", agent.epsilon)
            break
        if (episode + 1) % win_trials == 0:
            create_policy_eval_video(f"{episode}-episodes")
            agent.save_weights(episode)
            print("Episode %d: Mean survival = %0.2lf in %d episodes" %((episode + 1), mean_score, win_trials))
            # agent.save_weights(episode)
    create_policy_eval_video(f"{episode}-episodes")
    agent.save_weights(episode)
    env.close()
    run.save(f'{base_path}{episode}-steps.h5')
run.finish()