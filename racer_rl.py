import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import argparse
from two_racers import *
import os
# import gym
# import roboschool
import sys
import pickle

import random

ep_num = 0

# from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): highest action to take
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer
            
        Return:
            action output of network with tanh activation
    """
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 32)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.l2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.l3 = nn.Linear(16, action_dim-1)
        self.l4 = nn.Linear(16, 1)

        self.max_action = max_action


    def forward(self, x):
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        vec = torch.tanh(self.l3(x))
        vel = torch.sigmoid(self.l4(x))  
        action = torch.cat([vec,vel],1)
        return action

class Critic(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): highest action to take
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer
            
        Return:
            value output of network 
    """
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        # self.l1 = nn.Linear(state_dim + action_dim, 32)
        # self.bn1 = nn.BatchNorm1d(num_features=32)
        # self.l2 = nn.Linear(32, 16)
        # self.bn2 = nn.BatchNorm1d(num_features=16)
        # self.l3 = nn.Linear(16, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 32)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.l2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.l3 = nn.Linear(16, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 32)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.l5 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(num_features=16)
        self.l6 = nn.Linear(16, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.bn1(self.l1(xu)))
        x1 = F.relu(self.bn2(self.l2(x1)))
        x1 = self.l3(x1)

        x2 = F.relu(self.bn3(self.l4(xu)))
        x2 = F.relu(self.bn4(self.l5(x2)))
        x2 = self.l6(x2)
        return x1, x2


    # def Q1(self, x, u):
    #     xu = torch.cat([x, u], 1)
    #     x1 = F.relu(self.l1(xu))
    #     x1 = F.relu(self.l2(x1))
    #     x1 = self.l3(x1)
    #     return x


# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""
    
    def __init__(self, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """        
        self.storage = []
        try:
            print("Attempt to load replay buff")
            self.storage = self.load_buff()
        except:
            print("Failed to load replay buff")
            pass
        self.max_size = max_size
        self.ptr = (len(self.storage) + 1) % self.max_size

    def load_buff(self):
        my_file = open("./analysis/replay_buffer","rb")
        return pickle.load(my_file)

    def dump_buff(self):
        my_file = open("./analysis/replay_buffer","wb+")
        pickle.dump(self.storage,my_file)

    def add(self, data):
        """Add experience tuples to buffer
        
        Args:
            data (tuple): experience replay tuple
        """
        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size
        
        Args:
            batch_size (int): size of sample
        """
        
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind: 
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)

class TD3(object):
    """Agent class that handles the training of the networks and provides outputs as actions
    
        Args:
            state_dim (int): state size
            action_dim (int): action size
            max_action (float): highest action to take
            device (device): cuda or cpu to process tensors
            env (env): gym environment to use
    
    """
    
    def __init__(self, state_dim, action_dim, max_action, env):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.ep_num = 0

        self.lr = 1e-3

        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay = 0.1)

        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay = 0.1)

        try:
            self.load()
        except Exception as e:
            print("failed to load checkpoint",e)

        for param in self.actor.parameters():
            param.requires_grad = True

        for param in self.critic.parameters():
            param.requires_grad = True

        self.max_action = max_action
        self.env = env


    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = max(self.lr * (0.3333333 ** (epoch // 10)),(10**-5))
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.lr

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.lr

        
    def select_action(self, state, noise=0.1):
        """Select an appropriate action from the agent policy
        
            Args:
                state (array): current state of environment
                noise (float): how much noise to add to acitons
                
            Returns:
                action (float): action clipped within action range
        
        """

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.env.action_space[0]))
            
        return action.clip(self.env.action_low, self.env.action_high)

    def eval_mode(self):
        self.actor.eval()

    def train_mode(self):
        self.actor.train()

    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """Train and update actor and critic networks
        
            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                iterations (int): how many times to run training
                batch_size(int): batch size to sample from replay buffer
                discount (float): discount factor
                tau (float): soft update for main networks to target networks
                
            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network
        
        """
        
        for it in range(iterations):

            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            simple_q1 = current_Q1.detach().mean().numpy()
            log_file.write("Q1 " + str(simple_q1) + "\n")

            simple_q2 = current_Q2.detach().mean().numpy()
            log_file.write("Q2 " + str(simple_q2) + "\n")
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 
            log_c_loss = critic_loss.detach().mean().numpy()
            log_file.write("CL " + str(log_c_loss) + "\n")

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                comb_action = self.actor(state)
                # comb_action = torch.cat([vec, vel],1)
                q1, _ = self.critic(state,comb_action)
                actor_loss = q1.mean()

                log_a_loss = actor_loss.detach().numpy()


                log_file.write("AL "+str(log_a_loss)+"\n")

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, episode_number, filename, directory):
        torch.save({
            "ep": episode_number,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim" : self.critic_optimizer.state_dict(),
            },"./saves/model.pth")

        # torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        # torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename="best_avg", directory="./saves"):
        print("Loading model ....")

        checkpoint = torch.load("./saves/model.pth")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optim"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optim"])
        self.ep_num = checkpoint["ep"]

        print("Model load complete")

        # self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        # self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


class Runner():
    """Carries out the environment steps and adds experiences to memory"""
    
    def __init__(self, env, agent, replay_buffer):
        
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset()
        self.done = False
        
    def next_step(self, episode_timesteps, noise=0.1):
        
        action = self.agent.select_action(np.array(self.obs), noise=0.1)
        
        # Perform action
        new_obs, reward, done, _ = self.env.step(action) 
        done_bool = float(done)
        
        # Store data in replay buffer
        replay_buffer.add((self.obs, new_obs, action, reward, done_bool))
        
        self.obs = new_obs
        
        if done:
            self.obs = self.env.reset()
            done = False
            
            return reward, True
        
        return reward, done

def evaluate_policy(policy, env, eval_episodes=100,render=False):
    """run several episodes using the best agent policy
        
        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
            render (bool): show training
        
        Returns:
            avg_reward (float): average reward over the number of evaluations
    
    """
    
    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            # if render:
            #     env.render()
            action = policy.select_action(np.array(obs), noise=0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

def observe(env,replay_buffer, observation_steps):
    """run episodes while taking random actions and filling replay_buffer
    
        Args:
            env (env): gym environment
            replay_buffer(ReplayBuffer): buffer to store experience replay
            observation_steps (int): how many steps to observe for
    
    """
    
    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:

        print(time_steps)

        action = np.random.normal(0.5,0.3,4) #### CHANGE
        action = np.clip(action,0,1)
        action[0:3] = action[0:3]*2 -1

        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add((obs, new_obs, action, reward, done))

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

def train(agent, test_env):
    """Train the agent for exploration steps
    
        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            writer (SummaryWriter): tensorboard writer
            exploration (int): how many training steps to run
    
    """
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = agent.ep_num
    episode_reward = 0
    episode_timesteps = 0
    done = False 
    obs = env.reset()
    evaluations = []
    rewards = []
    best_avg = -2000

    agent.eval_mode()
    
    writer = SummaryWriter(log_dir="./runs")

    while total_timesteps < EXPLORATION:
    
        if done: 

            if total_timesteps != 0:

                # env.sim_Pause() 
                rewards.append(episode_reward)
                log_file.write("E " + str(episode_reward) + " \n")

                avg_reward = np.mean(rewards[-100:])
                
                writer.add_scalar("avg_reward", avg_reward, total_timesteps)
                writer.add_scalar("reward_step", reward, total_timesteps)
                writer.add_scalar("episode_reward", episode_reward, total_timesteps)

                agent.ep_num = episode_num
                
                if best_avg < avg_reward:
                    best_avg = avg_reward
                    print("saving best model....\n")
                    agent.save(episode_timesteps,"best_avg","saves")

                print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(
                    total_timesteps, episode_num, episode_reward, avg_reward), end="")
                sys.stdout.flush()


                if avg_reward >= REWARD_THRESH:
                    break

                agent.train_mode()
                agent.train(replay_buffer, episode_timesteps, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP, POLICY_FREQUENCY)
                agent.eval_mode()
                replay_buffer.dump_buff()
                
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

                agent.adjust_learning_rate(episode_num)

                # env.sim_Play() 
                env.reset()



        reward, done = runner.next_step(episode_timesteps)
        log_file.write("r " + str(reward) + "\n")
        episode_reward += reward
        if episode_timesteps > 2200:
            done = True
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

        if(total_timesteps%100==0):
            print(total_timesteps,"|",episode_num)


def getRacer(args):
    drone_names = ["drone_1", "drone_2"]
    drone_params = [
        {"r_safe": 0.5,
         "r_coll": 0.5,
         "v_max": 80.0,
         "a_max": 40.0},
        {"r_safe": 0.4,
         "r_coll": 0.3,
         "v_max": 20.0,
         "a_max": 10.0}]

    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    return BaselineRacerGTP(
        drone_names=drone_names,
        drone_i=0,  # index of the first drone
        drone_params=drone_params,
        use_vel_constraints=args.vel_constraints,
        race_tier=args.race_tier)

def setupRacer(racer,args):
    racer.level_name = args.level_name
    racer.race_tier = args.race_tier
    racer.load_level(args.level_name)


SEED = 0
OBSERVATION = 1000
EXPLORATION = 5000000
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
EVAL_FREQUENCY = 5000
REWARD_THRESH = 8000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Commands:
    No Graphics - ./AirSimExe.sh -nullrhi
    With Graphics - ./AirSimExe.sh -windowed -opengl4
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--vel_constraints', dest='vel_constraints', action='store_true', default=False)
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard", 
        "Qualifier_Tier_1", "Qualifier_Tier_2", "Qualifier_Tier_3"], default="ZhangJiaJie_Medium")
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=False)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    
    log_file = open("analysis/log-mlp.txt", "a+")

    env = getRacer(args)
    setupRacer(env,args)
    # env.run()
    env.cold_start()


    state_dim = env.observation_space[0]
    action_dim = env.action_space[0] 
    max_action = float(env.action_high)

    policy = TD3(state_dim, action_dim, max_action, env)
    replay_buffer = ReplayBuffer()
    runner = Runner(env, policy, replay_buffer)

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    if len(replay_buffer.storage)==0:
        observe(env, replay_buffer, OBSERVATION)
    train(policy, env)

    policy.load()
    for i in range(100):
        evaluate_policy(policy, env, render=True)
    log_file.close()










# #env = gym.make(ENV)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Set seeds
# #env.seed(SEED)
# #torch.manual_seed(SEED)
# #np.random.seed(SEED)

# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0] 
# max_action = float(env.action_space.high[0])

# policy = TD3(state_dim, action_dim, max_action, env)

# replay_buffer = ReplayBuffer()

# runner = Runner(env, policy, replay_buffer)

# total_timesteps = 0
# timesteps_since_eval = 0
# episode_num = 0
# done = True

# observe(env, replay_buffer, OBSERVATION)

# train(policy, env)

# policy.load()

# for i in range(100):
#     evaluate_policy(policy, env, render=True)

# env.close()
