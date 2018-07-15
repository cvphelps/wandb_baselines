# Introduction

I'm going to give you a brief introduction to reinforcement learning and an overview of the key concepts. I'll detail my learning process and my favorite resources. I used `wandb` to track how different algorithms performed. I really recommend `wandb` because it's so low friction. With just a couple of lines in my script I was automatically tracking the progression of my model as I trained.

I tested various RL algorithms on the simple environment `MountainCar` and the complex environment `Humanoid`. I'll show you examples of trained agents from both these environments.

### Mountain Car

Gym-id: `MountainCarContinuous-v0`

The goal is to make that wagon reach the top of the mountain.
For each timestep:
* state: the car's position at a timestep
* reward: x displacement
* action: force applied in a direction

![Mountain Car](https://thumbs.gfycat.com/WideeyedUntriedJellyfish-size_restricted.gif)

The car doesn't have enough acceleration to drive up the hill, so the agent has to learn to back up and use gravity to generate some momentum.

### Humanoid

Gym-id: `RoboschoolHumanoid-v1`

The goal here is to make the humanoid bot walk as far as possible.

![Roboschool Humanoid](https://thumbs.gfycat.com/WaryUnhappyEuropeanfiresalamander-size_restricted.gif)


Let's look at the RL definition and then get into the details.

# Reinfrocement Learning

In this type of machine learning we've got an agent with a goal, and the agent is placed into an environment. At each timestep the environment has a state, and the agent performs an action based on that state. The "reinforcement" comes in here— we give the agent a reward when it chooses an action that leads to success.

The agent's goal is to figure out what action to take given the current state of the environment to maximize the expected cumulative reward over time.

In the mountain car example above, the goal of the agent is to make the wagon reach the top of the mountain. At every timestep you are given the position of the car, the agent has to figure out how much force and in what direction it has to apply to the car such that the cumulative reward is maximized over time. At every timestep when an action is performend, the environment will give you the next state of the car which is it's position and the reward it got and the next position of the car. This environment is tricky because the force available to us is not enought to make the car go forward and climb uphill. So the agent has to learn to go back first and use gravity to generate some momentum to reach the top of the hill.

![Imgur](https://i.imgur.com/nOx1lE1.png)

Formally we can describe reinforcement learning as a Markov Decision Process problem.

A Markov Decision Process is a tuple ⟨S, A, P, R, γ⟩
* S is a finite set of states
* A is a finite set of actions
* P is a state transition probability matrix,
* Pa′ =P[St+1=s′|St=s,At=a] ss
* R is a reward function, Ras = E[Rt+1 | St = s,At = a] 
* γ is a discount factor γ ∈ [0, 1].


# RL Challenges
* There is no supervisor, only a single reward signal. A model in the learning process uses this reward signal carefully to explore the environment.
* Feedback is delayed, not instantaneous. Because as you can see from `MountainCar` example it has to get negative rewards initially by going back to get the maximal cumulative reward.
* Time really matters. This data is sequential, not [independent and identically distributed](https://www.wikiwand.com/en/Independent_and_identically_distributed_random_variables). While exploring the state space, the data you will be collecting is highly correlated. We make some assumptions to simplify this, but past and future observations are not independent.
* An agent’s actions affect the subsequent data it receives.

# Keywords
## Agent
An RL agent includes one or more of these components:
* Policy: The agent's behaviour function which determines which action to take given the current state.
* Value Function: How much reward is given for each state.
* Model: Agents representation of the environment.
## Observations, State
The current state of the environment. For example, in the `Humanoid` environment an observation might be the position of all the limbs.
## Actions
The pong game has a simple example of this: actions are move right, move left, or stay still.
## Reward
An agent takes an action receives a reward which represents how good that action was in the given state.
## Discount Factor
A proxy for memory and planning. This edits the amount of reward based on past actions or future considerations.
## Value Function
The expected cumulative reward for achieving a given state. Input state and the value function outputs how much potential reward is possible. Usually it is a neural network.
## Q Values
In a given state, what actions give the best expected cumulative reward. A Q value is calculated based on both state and action. The updated Q value *does* depend on the current Q value. Q = f(s, a)
## Policy Function
Policy function basically gives you what action to take given the state. Its the behaviour of RL agent. This is what we are trying to figure out.
## Bellman Equation
Used to compute values of a state or Q values of a state action pair. Its an iterative algorithm. We use this to solve MDPs.

# Weights & Biases
Before I discovered `wandb` I used to create a terminal window in tmux to track each hyperparameter variant in a headless machine, with another htop window open to track system metrics.

![Learning progress before wandb](https://i.imgur.com/aCV4rDs.png)

It took me more than 20 commands to set this up using `grep` and `watch` commands, and then I had to recreate this dashboard by handfor each variant. I was ssh'ing into machines every time I wanted to check on training- often a process that took days.

Now all it takes is a single line of code to automatically track training. Another feature that has come in handy is the system metrics `wandb` logs. Sometimes a run will crash silently because it ran out of memory— the `wandb` system metrics let me diagnose this fail state and avoid wasting hours of compute resources.

Training on complex problems often takes over a month of testing various algorithms and hyperparameters, so hyperparameter 'bookkeeping' was a huge pain.

Now all I have to do is start several variants in multiple machines and I just log in to the wandb dashboard, and see how various algorithms, hyperparameters are performing. I would even get the print logs to the stdout streamed to the dashboard.

But overall I get so much value by just adding these 3 lines below.

```
import wandb; wandb.init()
wandb.config.update() # log hyperparameters
wandb.log() # log training progress
```

# How I learnt RL, Resources.

## [Berkeley's, Artificial Intelligence Course]( https://courses.edx.org/courses/BerkeleyX/CS188x_1/1T2013/20021a0a32d14a31b087db8d4bb582fd/)

This course introduces you to the basics of RL, Markov Decision Processes, Bellman Equation, Policy Iteration algorithms, Value Iteration Algorithms. After this, you can easily solve environments whose observation and action spaces are discrete. It also has a cool assignment where you will play with pac man agent, where you build an agent to solve pacman game.

## [David Silver's Youtube lecture](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)

This course actually goes a little bit deeper into  MDP's, Dynamic programming, Model-Free Prediction, Model-Free Control, Value function approximation, actually introduces Neural Networks as Policy and Value functions, Policy gradient algorithm, Actor-Critic algorithm.

I think this is one of the best RL courses out there, not only because the instructor is behind all the cutting edge research happening in this field, but also because of the questions asked by the students. It might not be as polished as Coursera or any other platforms, the participation by the students makes it so much worth it.

## [An Introduction to Reinforcement Learning, Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)
Both the above courses are based on this book. Free book you can download from the above link. Very accessible book. The only place where math gets hard is during the discussion of **Policy Gradient** but it is also the basis for all the cool RL algorithms.

Even with vanilla policy gradient algorithm, it's hard to converge hard continuous environemnts like Humanoid and Atari etc. So instead of implementing them myself I'm using one of the popular RL libraries out there Below I'm describing the experiment setup.

# Experiment Setup.
## Open AI GYM
Basically this library makes lots of environments available that you can play with.
## Baselines, RLLAB
RL libraries from OpenAI, that implements most of the popular RL algorithms, that are easy to hack and play around with. They made sure it's easy to reproduce results from all the RL papers.
## Roboschool
Alternative to MUJOCO environments, I didn't have a license to play with MUJOCO environemtns, so instead I installed Roboschool to play with complex environments like Humanoid and etc.

I'm running these experiments on AWS Ubuntu instances, so also needed to have  xvfb installed, to record how the episodes of the environment.

All I have to do is run my python script with `xvfb-run` prefixed like below.
```
xvfb-run -s "-screen 0 1400x900x24" python baselines/ppo1/run_mujoco.py --env RoboschoolHumanoid-v1 --num-timesteps 30000000
```
This attaches a fake monitor that the python script can access and record and capture the environment actually playing.

Because I'm using roboschool, it's slightly different how I capture the frames, from the normal gym environments. Usually I just use the `Monitor` wrapper to record the video and progress. For some reason this doesn't work with roboschool environments. Instead I use `VideoRecorder` wrapper to do it manually. Below is a sample code of how I record video of a single episode of an environment.

```
env = gym.make('RoboschoolHumanoid-v1')
total_reward = 0
ob = env.reset()
video = True
if video:
    video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env=env, base_path=os.path.join('/home/ubuntu/wandb_baselines', 'humanoid_run2_%i'%seed), enabled=True)

while True:
    action = pi.act(stochastic=False, ob=ob)[0]
    ob, r, done, _ = env.step(action)
    if video:
        video_recorder.capture_frame()
    tot_r += r
    if done:
        ob = env.reset()  
        if video:
            video_recorder.close()
        print(total_reward)
        break
```
# Description of the environment

# Hyperparameters

# Policy Gradient

# Actor Critic

# DDPG

# TRPO

# PPO1

# PPO2

# Results

