import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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



def run_metrics(agent,states,q_upper):

    with torch.no_grad():
        out = agent.forward(torch.FloatTensor(states)).detach().numpy()


    ## now we have out which is q values over all 100K states
    Q_max = np.max(out,axis = 1) ## biggest Q in each state
    Q_min = np.min(out,axis = 1)

    excess = np.maximum(Q_max-q_upper,0) ## all the violating max Q values 
    violating_excess = excess[excess>0]
    deficit = np.maximum(-Q_min,0) ## number of Q values below 0## thi sis impossible

    metrics = {
        'Mean_Q':np.mean(out),
        'Mean_Excess':np.mean(excess),
        'Mean_Violtaing_Excess':np.mean(violating_excess),
        'Mean_Over_Violation_Rate':np.mean(Q_max > q_upper),
        'Mean_Under_Violtaion_Rate':np.mean(deficit > 0),
    }

    return metrics

Algorithims = ['DQN','DDQN']
Seeds = [x for x in range(10,20)]
Steps = [x*1000 for x in range(1,101)]
Frequency = [758,535,591,647,50,200,500,1000]



Q_UPPER = 100

agent = Agent_Net()
eval_states = np.load(r'/Path/to/100K_Q_states.npy',allow_pickle=True)

all_results = []

BASE_DIR = Path(
    r"enter_your_path"
)

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "q_metrics_working.csv"


for algo in Algorithims:
    for freq in Frequency:
        for seed in Seeds:
            for step in Steps:
            
                agent.load_state_dict(torch.load(f'##path##/new_agents/{algo}/Seed_{seed}/Update_Freq_{freq}/Agent_{step}'))        
                metrics = run_metrics(agent,eval_states,Q_UPPER)
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










