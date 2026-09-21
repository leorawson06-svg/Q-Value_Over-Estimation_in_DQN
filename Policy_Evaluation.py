import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import random

from gymnasium.wrappers import TimeLimit
from pathlib import Path



class Agent_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.action_space_n = 2
        self.obs_space = 4
        self.Policy = nn.Sequential(
                                    nn.Linear(self.obs_space,128),nn.ReLU(),
                                    nn.Linear(128,128),nn.ReLU(),
                                    nn.Linear(128,self.action_space_n)
        )

        self.loss_function = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=2e-4,
        )


    def forward(self,obs):
        return self.Policy(obs)

    def backward(self, prediction, target):
        
        loss = self.loss_function(prediction,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def proc_episode(Rewards,Qs):


    discount = []
    running_Q = 0 

    for i in range(len(Rewards)-1,-1,-1):
        
        if i == len(Rewards)-1:
            running_Q = Rewards[i]
        else:
            running_Q = Rewards[i] + 0.99 * running_Q

        discount.append(running_Q)
    discount.reverse()

    diff = Qs[0]-discount[0] ## just the intial value

    return diff

def initialise_run(seed):

    # Amke network start from same initial weights
    torch.manual_seed(seed)
    exploration_rng = np.random.default_rng(seed + 1)
    replay_rng = random.Random(seed + 2)

    env = gym.make("CartPole-v1").unwrapped
    env.reset(seed=seed + 3)
    env = TimeLimit(env, max_episode_steps=3000)
    
    env.action_space.seed(seed + 4)

    return env, exploration_rng, replay_rng

def eval(agent):


    env,exploration_rng,replay_rng = initialise_run(seed = 11) ## 



    ### EVAL ####
    eval_episodes = 25

    #Hyper Params#
    done = False

    ep_len = 0
    initial_diff = 0 
    ## trianing loop  

    for _ in range(eval_episodes):

        episode_rewards = []
        episode_Qs = []
      
        cur_obs,info = env.reset()
        done = False
       



        while not done:

            ep_len += 1

            Qs = agent.forward(torch.FloatTensor(cur_obs)).detach().numpy()
                       
            action = np.argmax(Qs)
            
        
            new_obs,reward,term,trun,info = env.step(action) 
            done = term or trun
                
            cur_obs = new_obs
            episode_rewards.append(reward)
            episode_Qs.append(Qs[action])


        ## Use whatever metric you need colleecting here
        initial_diff += proc_episode(episode_rewards,episode_Qs)
      
    
        



    metrics = {
        'Avg_Reward': ep_len/eval_episodes,
        'Avg_Abs_Diff': initial_diff/eval_episodes,
    }

    return metrics


Algorithims = ['DDQN']
Seeds = [x for x in range(10,20)]
Steps = [x*1000 for x in range(1,101)]
#Frequency = [535,591,647,758]
Frequency = [50,200,500,1000]

Q_UPPER = 100

agent = Agent_Net()

all_results = []

BASE_DIR = Path(
    r""
)

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "policy_metrics_2.csv"



for algo in Algorithims:
    for freq in Frequency:
        for seed in Seeds:
            for step in Steps:
            
                agent.load_state_dict(torch.load(f'/new_agents/{algo}/Seed_{seed}/Update_Freq_{freq}/Agent_{step}'))        
                metrics = eval(agent)
                print('Finished',algo,freq,seed,step)

                row = {

                    "algorithm":
                        algo,

                    "target_update_interval":
                        freq,

                    "seed":
                        seed,

                    "training_step":
                        step,

                    "q_upper":
                        Q_UPPER,

                    **metrics,

                }

                all_results.append(
                    row
                )


results_df = pd.DataFrame(
    all_results
)

# Sort so the CSV is easy to inspect
results_df = results_df.sort_values(
    [
        "algorithm",
        "target_update_interval",
        "seed",
        "training_step",
    ]
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)








