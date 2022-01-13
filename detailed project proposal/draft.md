# Detailed Project Proposal - Draft


## Problem 
Robotic locomotion is a difficult problem to tackle, this is because of many variables, including the complexity of the gait and adapting to unpredictable terrain, for a long time robots have moved using wheels which, although it has many benefits, including it's easy implementation and hability to move with no complexity on simple, predictable environments. It lacks, altough, the hability to overcome obstacles, move up stairs and the flexibility humans have (include more mobility reasons to choose bipede/gait over wheels). In the robocup competition, in the league where the team boldhearts competes (teen size humanoid soccer), robots are required to move using limbs, and rule changes keep increasing the complexity of the competition, implementing for example tall grass, which is a big obstacle problem that the robot has to overcome, since the ground is irregullar and behaves in an unpredictable way, (research regular walking algorithms vs RL) 

## Solution
To solve the problem of increasing difficulty and irregular terrain I'll develop an algorithm trained using reinforcement learning, this technique will be usefull to train the robot independently of the robot structure, the type of soil (as long as it can be modeled, or it's irregularities be taken in acount in training). To implement this RL algorithm, I'll be using Mujoco along with OpenAI Gym to run the simulation, environment and training. The algorithm will be using Keras (DeepQNetwork and Policy Gradient), both algorithms will be tested to test its performance, with expectations of better performance using DeepQNetwork


## Steps
The main development will be split into three main stages split into levels of difficulty, although, they should relate and the results should be transmissible between the stages.

- The first stage will cover the development of a simple agent simulating the well- known cart pole environment, this stage will use OpenAI but omit Mujoco for a simple 2d simulation. This will improve the knowledge and understanding of the process of developing a reinforcement learning algorithm. 
- The second stage of development will apply a high-level approach to a simplified humanoid (full-bodied or partial) this will introduce Mujoco. The process of tunning and developing the reward system can actively take place at this stage; this will be an important step to understand the complexity of the problem and relevant issues such as the performance of training on such a large action space. 
- The third stage would apply the same strategy developed on stage 2 to a realistic model of the humanoid used by the team Boldhearts and further iterations on the reward system. The desired final stage would be to successfully apply the learned algorithm to the real robot.


## References
https://arxiv.org/pdf/2103.04616.pdf



## Bits


[HOW IS MUJOCO BETTER THAN GAZEBO]

Another problem this development tries to cover is the transition between simulation training and running the algorithm on the robot, to try and close the gap, the training stage will use the recently open-sourced simulator Mujoco instead of Gazebo, the current simulator used by the team, this decision is based on the better physics engine present on Mujoco



In order to target this problem, I will attempt to develop a reinforcement learning algorithm, this will tackle a few problems, it will run/learn autonomously should be able to train even if the robot structure or the environment conditions change, this is true because the reward function doesn’t take into account any of the variables that might change with new regulations, this is especially important to streamline the adaptation process when new rules are implemented, allowing the team to focus on new improvements. Even though the simulator used by the team is gazebo, the simulator selected to be used in the training stage is Mujoco, this decision was based on the software changing [find better word] to open source and its superior physics engine, likely allowing for a smoother transition from simulation to the real robot. To train the algorithm, I will be using OpenAI gym to implement the training process structure, this helps standardize the implementation and benchmark and compare the results. The algorithm chosen to train were DeepQNetwork and Policy gradient.




## Notes from the meeting

make arguments compeling. 
-- Proposed 
Mention robocup rule changes and how the architecture changes, such as structure,height of the robot etc.

go from general to specific explicit programming, supervised learning and reinforcement learning using rewards. - before mentioning I'm going to use RL.

going to sue RL because of the project compelxity.


walking is difficult because of ... - primary goal
    - gait is a specific part of walking
    - split walking into the diferent difficulties
  not only about the requirements but also to assimilate to humans and be more autonomous


context on the tools - mujoco, open ai gym.

Discussed about mujoco vs gazebo
clear > popular. on python
Bold Hearts

stakeholders:

humanoid walking researches in general and robocupers in specific  and boldhearts

specify the timming risc- all the variables

order risks by order

explain why mujoco willslow down the integration
specify avoiding abinguity

change reward to 

