- [ ] A reflection on how well you feel the project is progressing with reference to the original aims and your project plan detailed in the third assignment on Project Planning
- [ ] A breakdown of the tasks and a short description of how they have enabled you to improve your knowledge and understanding of Computer Science/Information Technology principles and practice
- [ ] Identification of changes you have made to your project aims and targets and the reasons why they have been made


The innitial goal of the project consited of achieving walking for a bipedal robot, targeting the Bold Hearts team's robots. The project showed to be very complex for the time avilable, while getting a fully walking robot might not be achieved, the project still covers extensively the development process and all of its outcomes, including what painpoints where found and how the reward function affected the outcome, it also develops a 2D Gym environment and the integration of ROS and openai gym in order to use the gazebo simulator along with gym.

Tasks in the development:

To simplify the development and ease the increasing complexity it was split into three parts. 

    The first level consited of the classic reinforcement learning problem, cartpole, where the objective is to balance a pole in a cart that can move in the horizontal axis.
    This allowed to test the implementation of the learning algorithm in a simple problem it also allowed to test different implementations such as a high level implementation of keras agents and a lower level implementation of the learning algorithm, allowing for more controll and understanding of how each element worked.
    Defining a structure of logging and code transferable to the next stages. This stage should also access the best training environment, including google colab/local.

    The second level consisted of the development of a 2d simulation of a simplified humanoid robot, this involves the selection of an adequate physics simulator and a rendering engine. The second task is to setup the gym environment, such as step, reset, and the reward function. The third task involves adapting the learning algorithm and neural network to be able to control all of the joints from the robot simultaneously. The final task is hyperparameter search, reward function itteration and logging the data.

    The third level consisted of the implementation of the currently used simmulation by the team Bold Hearts, implemented in gazebo to train walking using openai gym. The second task involves the re-adaptation of the learning algorithm to use the states from the 3d environment and the higher number of joints.

    (expand how improved knowledge)

     While the innitial target for the project was to get achieve walking on a simulation of the team's robot and possibly the transition to the real robot, givven the reduced time and the complexity of the project, allong with the many difficulties faced the main target of the project became the implementation of both the learning algorithm and the implementation of openai_ros, the documentation of the results and the painpoints of development allong with documentation on how to setup this environment along with the setup of the teams environment on a new architecture (ARM64) which requires the creation of a new image and some workarrounds allows for other members to setup the environment if moved to the architecture.

