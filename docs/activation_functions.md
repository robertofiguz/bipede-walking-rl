# Activaton functions
 Help a model account for interaction effects - interaction effect is when one variable affects one prediction depending on the value of a different variable
    
### To explore:

- why ReLU doesn't account for negative values
- how to measure performance of the nn and compare activation functions/algorithms
  
## Rectified linear unit (ReLU)

- defined as the positive part of its argument:
  
  ![relu](https://www.researchgate.net/profile/Leo-Pauly/publication/319235847/figure/fig3/AS:537056121634820@1505055565670/ReLU-activation-function.png)

## Sigmoid
  - It is the most widely used activation function as it is a non-linear function. Sigmoid function transforms the values in the range 0 to 1.
  
    ![Sigmoid](https://qph.fs.quoracdn.net/main-qimg-07066668c05a556f1ff25040414a32b7)
## Tanh
  - Tanh function is continuous and differentiable, the values
lies in the range -1 to 1. As compared to the sigmoid function the gradient of tanh function is more steep. Tanh is preferred over sigmoid function as it has gradients which are not restricted to vary in a certain direction and also, it is zero centered.
  
    ![Tanh](https://www.tutorialexample.com/wp-content/uploads/2020/08/the-graph-of-tanhx-function.png)

## Leaky ReLU
  - Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation function based on a ReLU, but it has a small slope for negative values instead of a flat slope. The slope coefficient is determined before training, i.e. it is not learnt during training. This type of activation function is popular in tasks where we we may suffer from sparse gradients, for example training generative adversarial networks.
  
    ![Leaky ReLU](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-25_at_3.09.45_PM.png)



# Value function
The value function of an optimization problem gives the value attained by the objective function at a solution, while only depending on the parameters of the problem. In a controlled dynamical system, the value function represents the optimal payoff of the system over the interval [t, t1] when started at the time-t state variable x(t)=x. If the objective function represents some cost that is to be minimized, the value function can be interpreted as the cost to finish the optimal program, and is thus referred to as "cost-to-go function.

# Gradients
Gradients represent the slope, the change over the y and x (or in 3 dimensions) axis over a period of time.

The gradient of a function f, written grad f or ∇f, is ∇f = ifx + jfy + kfz where fx, fy, and fz are the first partial derivatives of f and the vectors i, j, and k are the unit vectors of the vector space.
## Saturation Gradient Problems
Sigmoid and tanh activation functions after a series of epochs of training, the linear part of each neuron will have very large values or very small values. This means that the linear part will have a big output value regardless of the sign. Therefore, the input of the sigmoid-like functions in each neuron which adds non-linearity will be far from the center of these functions. 
![sigmoid](https://i.stack.imgur.com/vvy9I.png)
In these locations, the gradient is very small, leading to the weights getting updated very slowly.

# Back Propagation
Back-propagation is the essence of neural net training. It is the practice of fine-tuning the weights of a neural net based on the error rate (i.e. loss) obtained in the previous epoch (i.e. iteration). Proper tuning of the weights ensures lower error rates, making the model reliable by increasing its generalization.

# Loss
Loss is the penalty for a bad prediction. A machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called empirical risk minimization.
![Loss](https://developers.google.com/machine-learning/crash-course/images/LossSideBySide.png)


# RL Algorithms
  ![Algorithms taxonomy](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)
  - **Model Based**: A model of the environment, is a function which predicts state transitions and rewards. The main upside to having a model is that it allows the agent to plan by thinking ahead, seeing what would happen for a range of possible choices, and explicitly deciding between its options. Model-based RL is a very good option for simple environments as this are easy to model and it can result in a substantial improvement in sample efficiency over methods that don’t have a model. The downside is that most times the entire environment is not available to the agent.
  - **Model Free**: While model-free methods forego the potential gains in sample efficiency from using a model, they tend to be easier to implement and tune. Model-free methods are more popular and have been more extensively developed and tested than model-based methods.
    - **Policy Optimization**: Methods in this family represent a policy explicitly as ![](https://spinningup.openai.com/en/latest/_images/math/400068784a9d13ffe96c61f29b4ab26ad5557376.svg). They optimize the parameters ![](https://spinningup.openai.com/en/latest/_images/math/ce5edddd490112350f4bd555d9390e0e845f754a.svg) either directly by gradient ascent on the performance objective ![](https://spinningup.openai.com/en/latest/_images/math/96b876944de9cf0f980fe261562e8e07029245bf.svg), or indirectly, by maximizing local approximations of ![](https://spinningup.openai.com/en/latest/_images/math/96b876944de9cf0f980fe261562e8e07029245bf.svg). This optimization is almost always performed on-policy, which means that each update only uses data collected while acting according to the most recent version of the policy. Policy optimization also usually involves learning an approximator ![](https://spinningup.openai.com/en/latest/_images/math/693bb706835fbd5903ad9758837acecd07ef13b1.svg) for the on-policy value function ![](https://spinningup.openai.com/en/latest/_images/math/a81303323c25fc13cd0652ca46d7596276e5cb7e.svg), which gets used in figuring out how to update the policy.
    - **Q-Learning**: Methods in this family learn an approximator ![](https://spinningup.openai.com/en/latest/_images/math/de947d14fdcfaa155ef3301fc39efcf9e6c9449c.svg) for the optimal action-value function, ![](https://spinningup.openai.com/en/latest/_images/math/cbed396f671d6fb54f6df5c044b82ab3f052d63e.svg). Typically they use an objective function based on the **Bellman equation**. This optimization is almost always performed off-policy, which means that each update can use data collected at any point during training, regardless of how the agent was choosing to explore the environment when the data was obtained. The corresponding policy is obtained via the connection between ![](https://spinningup.openai.com/en/latest/_images/math/c2e969d09ae88d847429eac9a8494cc89cabe4bd.svg) and ![](https://spinningup.openai.com/en/latest/_images/math/1fbf259ac070c92161e32b93c0f64705a8f18f0a.svg): the actions taken by the Q-learning agent are given by:
        - ![](https://spinningup.openai.com/en/latest/_images/math/d353412962e458573b92aac78df3fbe0a10d998d.svg)
    - **Trade-offs Between Policy Optimization and Q-Learning**: The primary strength of policy optimization methods is that they are principled, in the sense that you directly optimize for the thing you want. This tends to make them stable and reliable. By contrast, Q-learning methods only indirectly optimize for agent performance, by training ![](https://spinningup.openai.com/en/latest/_images/math/713b5ea31ad66705079ea5786dd84e06944402b7.svg) to satisfy a self-consistency equation. There are many failure modes for this kind of learning, so it tends to be less stable. But, Q-learning methods gain the advantage of being substantially more sample efficient when they do work, because they can reuse data more effectively than policy optimization techniques.

# Markov decision process
  Markov Decision Process is Markov Reward Process with a decisions. Everything is same like MRP but now we have actual agent that makes decisions or take actions.
  It is a tuple of (S, A, P, R, 𝛾) where:

  S is a set of states, <br>
  A is the set of actions agent can choose to take,<br>
  P is the transition Probability Matrix,<br>
  R is the Reward accumulated by the actions of the agent,<br>
  𝛾 is the discount factor.

  ## Markov Reward Process 
  As the name suggests, MDPs are the Markov chains with values judgement.Basically, we get a value from every state our agent is in.<br>
  Rs = E[R+1 | S]

  ## Markov Property
  ![](https://miro.medium.com/max/990/1*Tr7GE76SiHh8_jVSYl2mug.png)

  This means that the transistion to state t+1  from state t is independent of the past, meaning that our current state already captures the information of the past states.

  ## **Markov chains**
  A Markov chain is a mathematical system that experiences transitions from one state to another according to certain probabilistic rules. The defining characteristic of a Markov chain is that no matter how the process arrived at its present state, the possible future states are fixed. In other words, the probability of transitioning to any particular state is dependent solely on the current state and time elapsed

  
# Experience Replay
  - The basic idea behind experience replay is to storing past experiences and then using a random subset of these experiences to update the Q-network, rather than using just the single most recent experience. This are used in order to avoid harmfull correlations.

# todo:
- compare performance of gcloud vs local jupyter
- contact is the main problem of the transition from the cartpole model, daniels sugestion:
  - start by using flat feet that are horizontaly constrained bot being able to tilt
- what metric to use mae has better results than acc on cartpole
- is DDPG better (more common) on continuos action spaces and DQN better(used only and not DDPG) for discrete action spaces
- box2d no reset state if need to go back
- Back propagation
- on and off policy
- Decision to abandon Keras-rl? favoring more controll, avoiding unecessary, possibly unmaintained libraries
- https://github.com/pybox2d/pybox2d/wiki/manual - docs on box2d
- Target Network
      [^This Docs](https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c) 
- To fix the problem of multiple actions i need to understand the continuous actions space, actor critic method
- Conv vs fully connected (Dense)
- [multiple action RL](https://arxiv.org/pdf/1711.08946.pdf)
- optimizer such as Adam ?!
- how to detect a fall? / fail for the done state
- calculate how straight the body is to increment reward and obtain a better targeted reward
- problems with continuous increase of reward:
  - robot will go back and forth to win more reward. suggestion: give 1 point if moved forward or even increasing reward. give negative reward if moves back to avoid this behaviour.
# Decisions
  changed from mujoco as it would require a new control system
# Problems found along the way:
- How to output multiple actions. On a robot need to controll multiple motors
- Reward system and combination of other hyperparameters (possibly gamma)led to learning to stand [wandb logs](https://wandb.ai/sudofork/walker-v2/runs/1zhv4wxe?workspace=user-sudofork)
  wasn't using an activation function on the output layers until V10(on wandb)

# References
[ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning)

[Activation functions](https://www.ijeast.com/papers/310-316,Tesma412,IJEAST.pdf)

[Leaky ReLU](https://paperswithcode.com/method/leaky-relu)

[Back propagation](https://towardsdatascience.com/how-does-back-propagation-in-artificial-neural-networks-work-c7cad873ea7)

[ML crash course Google - Loss](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss)

[Reinforcement learning algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

### Not Explored
[Natural gradients](https://towardsdatascience.com/its-only-natural-an-excessively-deep-dive-into-natural-gradient-optimization-75d464b89dbb)

[ML learning theory](https://www.cs.cmu.edu/afs/cs/user/avrim/www/Talks/mlt.pdf)

[Gradient descent](https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21)

[RL key concepts - OpenAI GYM](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations)

[Dense Layers](https://analyticsindiamag.com/a-complete-understanding-of-dense-layers-in-neural-networks/)

[2d biped walker](https://github.com/Kyziridis/BipedalWalker-v2/blob/master/Research%20Paper.pdf)

[Might cover multiple motor problem](https://arxiv.org/pdf/1701.08878.pdf)

[robot RL control](https://github.com/normandipalo/intelligent-control-techniques-for-robots/blob/master/report.pdf)



# write about

write about how i setup all the log and all the code logging

write about how i made it so everything can be reproduced

# ADE INTERNET ACCESS 
  ## used to install dependenies of openai_ros2

pkill docker
iptables -t nat -F
ifconfig docker0 down
brctl delbr docker0
docker -d
sudo systemctl restart docker