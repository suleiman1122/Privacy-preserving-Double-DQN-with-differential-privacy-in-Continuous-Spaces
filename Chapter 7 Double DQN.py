import math, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt

import gym
import numpy as np
import dpsgd
import argparse
from collections import deque
from tqdm import trange

parser = argparse.ArgumentParser(description='DPSGD')
parser.add_argument('--batchsize', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
# noise clip lr 0.0000000001
parser.add_argument('--norm-clip', type=float, default=10000.0, metavar='M',
                    help='L2 norm clip (default: 1.0)')
parser.add_argument('--noise-multiplier', type=float, default=0.0000000001, metavar='M',
                    help='Noise multiplier (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--microbatches',type=int, default=1, metavar='N',
                    help='Majority Thresh')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()

train_batch = args.microbatches
test_batch = args.test_batch_size

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

env_id = 'LunarLander-v2'
env = gym.make(env_id)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 3000
buffer_size = 3000
neurons = 128

eps_by_episode = lambda episode: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode / epsilon_decay)

# plt.plot([eps_by_episode(i) for i in range(50000)])
# plt.show()

class DDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DDQN, self).__init__()        
        
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = autograd.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)
        return action



def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())



def compute_td_loss(batch_size,optimizer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = autograd.Variable(torch.FloatTensor(np.float32(state)))
    next_state = autograd.Variable(torch.FloatTensor(np.float32(next_state)))
    action     = autograd.Variable(torch.LongTensor(action))
    reward     = autograd.Variable(torch.FloatTensor(reward))
    done       = autograd.Variable(torch.FloatTensor(done))

    q_values      = current_model(state)
    next_q_values = target_model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
    


def plot_fig(run,score_0 ,score_1 , score_2,score_3):
 
    plt.xlabel('"Number of samples trained')
    plt.ylabel('Scores')
    plt.plot(run,score_0,color='r',label='No noise')
    plt.plot(run,score_1,color='k',label='noise 1e-10')
    plt.plot(run,score_2,color='b',label='noise 1e-5')
    plt.plot(run,score_3,color='g',label='noise 1e-2')
    plt.legend() 
    plt.show() 

def play_game():
    done = False
    state = env.reset()
    while(not done):
        action = current_model.act(state, epsilon_final)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        state = next_state

episodes = 50000
batch_size = 32
gamma      = 0.99
score_0=[]
score_1=[]
score_2=[]
score_3=[]
run=[]
run_t=[]
noise=[0, 0.0000000001 , 0.00001 , 0.01]
for i in range(4):
    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    tot_reward = 0

    current_model = DDQN(env.observation_space.shape[0], env.action_space.n)
    target_model  = DDQN(env.observation_space.shape[0], env.action_space.n)
    replay_buffer = ReplayBuffer(buffer_size)
    update_target(current_model, target_model)
    if i==0:
        optimizer = optim.Adam(current_model.parameters())
    else:
        optimizer = dpsgd.DPSGD(current_model.parameters(),lr=args.lr,batch_size=args.batchsize//args.microbatches,C=args.norm_clip,noise_multiplier=noise[i])

    tr = trange(episodes+1, desc='Agent training', leave=True)
    print(tr)
    for episode in tr:
        tr.set_description("Agent training (episode{}) Avg Reward {}".format(episode+1,tot_reward/(episode+1)))
        tr.refresh() 
        run_t.append(episode)
        globals()['score_'+ str( i ) ].append(tot_reward/(episode+1))
        epsilon = eps_by_episode(episode)
        action = current_model.act(state, epsilon)
        
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        tot_reward += reward
        
        state = next_state
        episode_reward += reward
        
        if done:
            if episode > buffer_size:
                play_game()
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0        
            
        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size,optimizer)
            losses.append(loss.item())
            
        # if episode % 10000 == 0:
        #     plot(episode, all_rewards, losses)
            
        if episode % 500 == 0:
            update_target(current_model, target_model)
    run=run_t   
    run_t=[]         

plot_fig(run , score_0 , score_1 , score_2 ,score_3 )          