import numpy as np 
import gymnasium as gym
import torch 
import torch.nn as nn
import random
from tqdm import tqdm as tqdm
import os 
from gymnasium.wrappers import TimeLimit


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

        #torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)

        self.optimizer.step()

        return loss.item()

def gen_targets(buffer_slice,target,agent,DDQN = False):

    batch_size = 128
    targets = np.zeros((batch_size,2))
    states = np.zeros((batch_size,4))
    gamma = 0.99

    
    observations = np.array(list(map(lambda x: x[0], buffer_slice)))    
    next_obs = np.array(list(map(lambda x: x[1], buffer_slice)))
    actions = np.array(list(map(lambda x: x[2], buffer_slice)))## action mask
    rewards = np.array(list(map(lambda x: x[3], buffer_slice))) 
    dones = np.array(list(map(lambda x: x[4], buffer_slice))) 

    Qs = agent.forward(torch.FloatTensor(observations)).detach().numpy()
    with torch.no_grad():
        if DDQN:
            agent_next_Qs = agent(torch.FloatTensor(next_obs))
            best_actions = agent_next_Qs.argmax(dim=1) ## get biggest Qs from agent opinon

            target_next_Qs = target(torch.FloatTensor(next_obs))
            next_Qs = target_next_Qs[torch.arange(batch_size),best_actions].numpy()

        else:

            target_next_Qs = target.forward(torch.FloatTensor(next_obs)).max(axis=1).values.detach().numpy()
            next_Qs = target_next_Qs


    tmp = (next_Qs * gamma * (1-dones) + rewards)
    targets += tmp[:,None] * actions
    targets += Qs * np.abs((1-actions))

    return targets,observations

def initialise_run(seed):

    # Amke network start from same initial weights
    torch.manual_seed(seed)
    exploration_rng = np.random.default_rng(seed + 1)
    replay_rng = random.Random(seed + 2)
 
    env = gym.make("CartPole-v1").unwrapped
    env.reset(seed=seed + 3)
    env = TimeLimit(env, max_episode_steps=1_000)

    env.action_space.seed(seed + 4)

    return env, exploration_rng, replay_rng

def train(seed, DDQN, target_interval):


    env,exploration_rng,replay_rng = initialise_run(seed = seed)

    agent = Agent_Net()
    target = Agent_Net()
    target.load_state_dict(agent.state_dict())

    save_interval = 1000

    ### TRAINING ####

    #Hyper Params#
    
        
    done = False
    buffer = []
    buffer_max_len = 25_000
    batch_size = 128

    training_steps = 100_000
    train_frequency = 10

    learning_starts = 1000

    eps = 1
    final_eps = 0.15

    eps_step = (eps-final_eps)/(training_steps//2)


    ## trianing loop 

    training_step = 0 

    while training_step < training_steps :
        cur_obs,info = env.reset()
        done = False
        ep_len = 0
        Target = False
        while not done and training_step<training_steps:
            training_step += 1
            ep_len += 1
            if exploration_rng.random() < eps:
                action = exploration_rng.integers(0, 2)
                
            else:
                Qs = agent.forward(torch.FloatTensor(cur_obs)).detach().numpy()
                action = np.argmax(Qs)
                

            eps = max(final_eps,eps-eps_step)
            
            new_obs,reward,term,trun,info = env.step(action) 
            done = term or trun

            if len(buffer) <buffer_max_len:
                if action == 0:
                    buffer.append([cur_obs,new_obs,[1,0],reward,term])
                else:
                    buffer.append([cur_obs,new_obs,[0,1],reward,term])

            else:
                del buffer[0]

                if action == 0:
                    buffer.append([cur_obs,new_obs,[1,0],reward,term])
                else:
                    buffer.append([cur_obs,new_obs,[0,1],reward,term])
                
            cur_obs = new_obs

            if len(buffer) > learning_starts and training_steps % train_frequency == 0:
                targets,states = gen_targets(replay_rng.sample(buffer, k=batch_size),target,agent,DDQN = DDQN)

                predictions = agent.forward(torch.FloatTensor(states)) 

                loss = agent.backward(predictions,torch.FloatTensor(targets))



            if training_step%target_interval == 0  and training_step!= 0:

                target.load_state_dict(agent.state_dict()) ## update target net for btoh DQN and DDQN
            if training_step % save_interval == 0 and training_step != 0:
                if DDQN:
                    torch.save(agent.state_dict(),f"/##path##/new_agents/DDQN/Seed_{seed}/Update_Freq_{target_interval}/Agent_{training_step}")
                else:
                    torch.save(agent.state_dict(),f'/##path##/new_agents/DQN/Seed_{seed}/Update_Freq_{target_interval}/Agent_{training_step}')

        if Target:
            break

    return agent,env


Frequencies = [50,200,500,1000,535,591,647,758]

for seed in range(10,20): ## for different seeds    
    os.makedirs(f"/##path##/new_agents/DDQN/Seed_{seed}", exist_ok=True)
    os.makedirs(f"/##path##/new_agents/DQN/Seed_{seed}", exist_ok=True)

    for K in Frequencies: ## varying target frequncy
           
        os.makedirs(f"/##path##/new_agents/DDQN/Seed_{seed}/Update_Freq_{K}", exist_ok=True)
        os.makedirs(f"/##path##/new_agents/DQN/Seed_{seed}/Update_Freq_{K}", exist_ok=True)
        train(seed,False,K)
        train(seed,True,K)
     

# trained_agent,env = train(7,False,100)


# env.close()

# env = gym.make('CartPole-v1',render_mode = 'human')
# cur_obs,_ = env.reset()


# for i in tqdm(range(300)):
#     cur_obs,info = env.reset()
#     done = False
#     while not done:
        
#         Qs = trained_agent.forward(torch.FloatTensor(cur_obs)).detach().numpy()
#         action = np.argmax(Qs)
#         print(Qs)
        
#         new_obs,reward,term,trun,info = env.step(action) 
#         done = trun or term   

#         cur_obs = new_obs

    