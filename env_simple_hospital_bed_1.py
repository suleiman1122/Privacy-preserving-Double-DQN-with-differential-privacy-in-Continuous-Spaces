from simpy_envs.env_simple_hospital_bed_1 import HospGym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim
import dpsgd
import argparse
# Use a double ended queue (deque) for memory
# When memory is full, this will replace the oldest value with the new one
from collections import deque

# Supress all warnings (e.g. deprecation warnings) for regular use
import warnings
warnings.filterwarnings("ignore")

DISPLAY_ON_SCREEN = False
# Discount rate of future rewards
GAMMA = 0.95
# Learing rate for neural network
LEARNING_RATE = 0.0003
# Maximum number of game steps (state, action, reward, next state) to keep
MEMORY_SIZE = 1000000
# Sample batch size for policy network update
BATCH_SIZE = 3
# Number of game steps to play before starting training (all random actions)
REPLAY_START_SIZE = 365 * 5
# Time step between actions
TIME_STEP = 1
# Number of steps between policy -> target network update
SYNC_TARGET_STEPS = 365
# Exploration rate (episolon) is probability of choosign a random action
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
# Reduction in epsilon with each game step
EXPLORATION_DECAY = 0.999
# Simulation duration
SIM_DURATION = 365
# Training episodes
TRAINING_EPISODES = 5000


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


class DQN(nn.Module):

    """Deep Q Network. Udes for both policy (action) and target (Q) networks."""

    def __init__(self, observation_space, action_space, neurons_per_layer=48):
        """Constructor method. Set up neural nets."""

        # Set starting exploration rate
        self.exploration_rate = EXPLORATION_MAX
        
        # Set up action space (choice of possible actions)
        self.action_space = action_space
              
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, action_space)
            )
        
    def act(self, state):
        """Act either randomly or by redicting action that gives max Q"""
        
        # Act randomly if random number < exploration rate
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
            
        else:
            # Otherwise get predicted Q values of actions
            q_values = self.net(torch.FloatTensor(state))
            # Get index of action with best Q
            action = np.argmax(q_values.detach().numpy()[0])
        
        return  action
        
        
    def forward(self, x):
        """Forward pass through network"""
        return self.net(x)


def optimize(policy_net, target_net, memory,optimizer1):
    """
    Update  model by sampling from memory.
    Uses policy network to predict best action (best Q).
    Uses target network to provide target of Q for the selected next action.
    """
      
    # Do not try to train model if memory is less than reqired batch size
    if len(memory) < BATCH_SIZE:
        return    
 
    # Reduce exploration rate (exploration rate is stored in policy net)
    policy_net.exploration_rate *= EXPLORATION_DECAY
    policy_net.exploration_rate = max(EXPLORATION_MIN, 
                                      policy_net.exploration_rate)
    # Sample a random batch from memory
    batch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, state_next, terminal in batch:
        
        state_action_values = policy_net(torch.FloatTensor(state))
        
        # Get target Q for policy net update
       
        if not terminal:
            # For non-terminal actions get Q from policy net
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach next state values from gradients to prevent updates
            expected_state_action_values = expected_state_action_values.detach()
            # Get next state action with best Q from the policy net (double DQN)
            policy_next_state_values = policy_net(torch.FloatTensor(state_next))
            policy_next_state_values = policy_next_state_values.detach()
            best_action = np.argmax(policy_next_state_values[0].numpy())
            # Get target net next state
            next_state_action_values = target_net(torch.FloatTensor(state_next))
            # Use detach again to prevent target net gradients being updated
            next_state_action_values = next_state_action_values.detach()
            best_next_q = next_state_action_values[0][best_action].numpy()
            updated_q = reward + (GAMMA * best_next_q)      
            expected_state_action_values[0][action] = updated_q
        else:
            # For termal actions Q = reward (-1)
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach values from gradients to prevent gradient update
            expected_state_action_values = expected_state_action_values.detach()
            # Set Q for all actions to reward (-1)
            expected_state_action_values[0] = reward
 
        # Set net to training mode
        policy_net.train()
        # Reset net gradients
        optimizer1.zero_grad()  
        # calculate loss
        loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
        # Backpropogate loss
        loss_v.backward()
        # Update network gradients
        optimizer1.step()  

    return

class Memory():
    """
    Replay memory used to train model.
    Limited length memory (using deque, double ended queue from collections).
      - When memory full deque replaces oldest data with newest.
    Holds, state, action, reward, next state, and episode done.
    """
    
    def __init__(self):
        """Constructor method to initialise replay memory"""
        self.memory = deque(maxlen=MEMORY_SIZE)

    def remember(self, state, action, reward, next_state, done):
        """state/action/reward/next_state/done"""
        self.memory.append((state, action, reward, next_state, done))

def plot_fig(run,score_0 ,score_1 , score_2,score_3):
 
    plt.xlabel('"Number of samples trained')
    plt.ylabel('Scores')
    plt.plot(run,score_0,color='r',label='No noise')
    plt.plot(run,score_1,color='k',label='noise 1e-10')
    plt.plot(run,score_2,color='b',label='noise 1e-5')
    plt.plot(run,score_3,color='g',label='noise 1e-2')
    plt.legend() 
    plt.show() 

def plot_results(avg_rewards1,run, exploration, score, run_details):
    """Plot and report results at end of run"""
    plt.xlabel("Number of samples trained")
    plt.ylabel("Score")
    plt.plot(run,avg_rewards1)
    plt.show()
    # Get beds and patirents from run_detals DataFrame
    beds = run_details['beds']
    patients = run_details['patients']    
    
    # # Set up chart (ax1 and ax2 share x-axis to combine two plots on one graph)
    # fig = plt.figure(figsize=(9,5))
    # ax1 = fig.add_subplot(121)
    # ax2 = ax1.twinx()
    
    # # Plot results
    # average_rewards = np.array(score)/SIM_DURATION
    # ax1.plot(run, exploration, label='exploration', color='g')
    # ax2.plot(run, average_rewards, label='average reward', color='r')
    
    # # Set axes
    # ax1.set_xlabel('run')
    # ax1.set_ylabel('exploration', color='g')
    # ax2.set_ylabel('average reward', color='r')
    
    # # Show last run tracker of beds and patients

    # ax3 = fig.add_subplot(122)
    # day = np.arange(len(beds))*TIME_STEP
    # ax3.plot(day, beds, label='beds', color='g')
    # ax3.plot(day, patients, label='patients', color='r')
    
    # # Set axes
    # ax3.set_xlabel('day')
    # ax3.set_ylabel('beds/patients')
    # ax3.set_ylim(0)
    # ax3.legend()
    # ax3.grid()
    # # Show
    
    # plt.tight_layout(pad=2)
    # plt.show()
    
    # Calculate summary results
    results = pd.Series()
    beds = np.array(beds)
    patients = np.array(patients)
    results['days under capacity'] = np.sum(patients > beds)
    results['days over capacity'] = np.sum(beds > patients)
    results['average patients'] = np.round(np.mean(patients), 0)
    results['average beds'] = np.round(np.mean(beds), 0)
    results['% occupancy'] = np.round((patients.sum() / beds.sum() * 100), 1)
    print (results);

def hosp_bed_management(i,noise):
    """Main program loop"""
    
    ############################################################################
    #                          8 Set up environment                            #
    ############################################################################
        
    # Set up game environemnt
    sim = HospGym(sim_duration=SIM_DURATION, time_step=TIME_STEP)

    # Get number of observations returned for state
    observation_space = sim.observation_size
    
    # Get number of actions possible
    action_space = sim.action_size
    
    ############################################################################
    #                    9 Set up policy and target nets                       #
    ############################################################################
    
    # Set up policy and target neural nets
    policy_net = DQN(observation_space, action_space)
    target_net = DQN(observation_space, action_space)
    policy_net.exploration_rate=1.0
    # Set loss function and optimizer
    if i==0:
        optimizer1 = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)
    else:
        optimizer1 = dpsgd.DPSGD(policy_net.parameters(),lr=args.lr,batch_size=args.batchsize//args.microbatches,C=args.norm_clip,noise_multiplier=noise)

    # Copy weights from policy_net to target
    target_net.load_state_dict(policy_net.state_dict())
    
    # Set target net to eval rather than training mode
    # We do not train target net - ot is copied from policy net at intervals
    target_net.eval()
    
    ############################################################################
    #                            10 Set up memory                              #
    ############################################################################
        
    # Set up memomry
    memory = Memory()
    
    ############################################################################
    #                     11 Set up + start training loop                      #
    ############################################################################
    
    # Set up run counter and learning loop    
    run = 0
    all_steps = 0
    continue_learning = True
    
    # Set up list for results
    results_run = []
    results_exploration = []
    results_score = []
    avg_rewards1 = []
    avg_rewards_t = []
    
    # Continue repeating games (episodes) until target complete
    while continue_learning:
        
        ########################################################################
        #                           12 Play episode                            #
        ########################################################################
        
        # Increment run (episode) counter
        run += 1
        
        ########################################################################
        #                             13 Reset game                            #
        ########################################################################
        
        # Reset game environment and get first state observations
        state = sim.reset()
        
        # Trackers for state
        weekday = []
        beds = []
        patients = []
        spare_beds = []
        pending_change = []
        rewards = []
        counter=0
        # Reset total reward
        total_reward = 0
        
        # Reshape state into 2D array with state obsverations as first 'row'
        state = np.reshape(state, [1, observation_space])
              
        # Continue loop until episode complete
        while True:
            
        ########################################################################
        #                       14 Game episode loop                           #
        ########################################################################
            
            ####################################################################
            #                       15 Get action                              #
            ####################################################################
            
            # Get action to take (se eval mode to avoid dropout layers)
            policy_net.eval()
            action = policy_net.act(state)
            
            ####################################################################
            #                 16 Play action (get S', R, T)                    #
            ####################################################################
            
            # Act 
            state_next, reward, terminal, info = sim.step(action)
            total_reward += reward

            # Update trackers
            weekday.append(state_next[0])
            beds.append(state_next[1])
            patients.append(state_next[2])
            spare_beds.append(state_next[3])
            pending_change.append(state_next[4])
            rewards.append(reward)
            counter=counter+1                                  
            # Reshape state into 2D array with state obsverations as first 'row'
            state_next = np.reshape(state_next, [1, observation_space])
            
            # Update display if needed
            if DISPLAY_ON_SCREEN:
                sim.render()
            
            ####################################################################
            #                  17 Add S/A/R/S/T to memory                      #
            ####################################################################
            
            # Record state, action, reward, new state & terminal
            memory.remember(state, action, reward, state_next, terminal)
            
            # Update state
            state = state_next
            
            ####################################################################
            #                  18 Check for end of episode                     #
            ####################################################################
            
            # Actions to take if end of game episode
            if terminal:
                # Get exploration rate
                exploration = policy_net.exploration_rate
                # Clear print row content
                clear_row = '\r' + ' '*79 + '\r'
                print (clear_row, end ='')
                print (f'Run: {run}, ', end='')
                print (f'Exploration: {exploration: .3f}, ', end='')
                average_reward = total_reward/SIM_DURATION
                print (f'Average reward: {average_reward:4.1f}', end='')
                avg_rewards_t.append(average_reward)
                avg_rewards1.append(np.mean(avg_rewards_t))
                # Add to results lists
                results_run.append(run)
                results_exploration.append(exploration)
                results_score.append(total_reward)
                
                ################################################################
                #             18b Check for end of learning                    #
                ################################################################
                
                if run == TRAINING_EPISODES:
                    continue_learning = False
                
                # End episode loop
                break
            
            
            ####################################################################
            #                        19 Update policy net                      #
            ####################################################################
            
            # Avoid training model if memory is not of sufficient length
            if len(memory.memory) > REPLAY_START_SIZE:
        
                # Update policy net
                optimize(policy_net, target_net, memory.memory,optimizer1)

                ################################################################
                #             20 Update target net periodically                #
                ################################################################
                
                # Use load_state_dict method to copy weights from policy net
                if all_steps % SYNC_TARGET_STEPS == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
    ############################################################################
    #                      21 Learning complete - plot results                 #
    ############################################################################
    
    # Add last run to DataFrame. summarise, and return
    run_details = pd.DataFrame()
    run_details['weekday'] = weekday 
    run_details['beds'] = beds
    run_details['patients'] = patients
    run_details['spare_beds'] = spare_beds
    run_details['pending_change'] = pending_change
    run_details['reward'] = rewards    
        
    # Target reached. Plot results
    # print(results_run," next ",avg_rewards1)
    # plot_results(avg_rewards1,results_run, results_exploration, results_score, run_details)
    
    return results_run,avg_rewards1

# Run model and return last run results by day
score_0=[]
score_1=[]
score_2=[]
score_3=[]
run=[]
noise=[0, 0.0000000001 , 0.00001 , 1.0]


for i in range(4):
     run , globals()['score_'+ str( i ) ]  = hosp_bed_management(i,noise[i])                     

plot_fig(run , score_0 , score_1 , score_2 ,score_3 )
                               