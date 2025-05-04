import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from collections import deque
import os
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

mp.set_start_method('spawn', force=True)

class ActorCritic(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor_x = nn.Sequential(
            nn.Linear(hidden_dim, 3),  # -1, 0, 1 for x-axis
            nn.Softmax(dim=-1)
        )
        self.actor_y = nn.Sequential(
            nn.Linear(hidden_dim, 3),  # -1, 0, 1 for y-axis
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        # Policy (actor)
        prob_x = self.actor_x(shared_features)
        prob_y = self.actor_y(shared_features)
        
        # Value (critic)
        value = self.critic(shared_features)
        
        return prob_x, prob_y, value

class Worker(mp.Process):
    def __init__(self, global_model, optimizer, rank, save_path, device):
        super(Worker, self).__init__()
        self.rank = rank
        self.save_path = save_path
        self.device = device
        self.game = r"C:\AllProgramming\MyProjects\UnityProjects\RL_TheHardestGame\BUILD_EasyStage1_Visual_Position"+str(rank)
        self.writer = SummaryWriter(os.path.join(save_path, f'worker_{rank}'))
        # Local model
        self.local_model = ActorCritic().to(self.device )
        self.local_model.load_state_dict(global_model.state_dict())
        
        self.optimizer = optimizer
        self.gamma = 0.99
        self.max_episodes = 500

    def get_action(self, state, model):
        state = torch.FloatTensor(state).to(self.device )
        prob_x, prob_y, _ = model(state)
        
        # Sample actions
        dist_x = torch.distributions.Categorical(prob_x)
        dist_y = torch.distributions.Categorical(prob_y)
        
        action_x = dist_x.sample()
        action_y = dist_y.sample()
        
        return (action_x.item()-1, action_y.item()-1), \
               (dist_x.log_prob(action_x), dist_y.log_prob(action_y))

    def run(self):
        env_config_channel = EngineConfigurationChannel()
        env_config_channel.set_configuration_parameters(width=64, height=64)
        env = UnityEnvironment(file_name=self.game, side_channels=[env_config_channel],base_port=5005+self.rank, no_graphics=True)
        env.reset()
        
        behavior_name = list(env.behavior_specs)[0]
        reward_window = deque(maxlen=100)
        for episode in range(self.max_episodes):
            env.reset()
            dec, term = env.get_steps(behavior_name)
            state = dec.obs[1][0]  # shape: (16,)
            
            done = False
            values = []
            log_probs = []
            rewards = []
            episode_steps = 0
            while not done:
                episode_steps += 1
                action, log_prob = self.get_action(state, self.local_model)
                
                # Convert action to ActionTuple format
                discrete_actions = np.array([[action[0], action[1]]], dtype=np.int8)
                action_tuple = ActionTuple(discrete=discrete_actions)
                
                env.set_actions(behavior_name, action_tuple)
                env.step()
                
                dec, term = env.get_steps(behavior_name)
                
                if len(term.interrupted) > 0:
                    done = True
                    next_state = term.obs[1][0]
                    reward = term.reward[0]
                elif len(dec.obs[1]) > 0:
                    next_state = dec.obs[1][0]
                    reward = dec.reward[0]
                    done = False
                
                _, _, value = self.local_model(torch.FloatTensor(state).to(self.device))
                values.append(value)
                log_probs.append(sum(log_prob))  # Sum log probs for both actions
                rewards.append(reward)
                
                state = next_state
                
            # Calculate returns and advantages
            episode_reward = sum(rewards)
            reward_window.append(episode_reward)
            avg_reward = np.mean(reward_window)
            self.writer.add_scalar(f'Worker_{self.rank}/Episode_Reward', episode_reward, episode)
            self.writer.add_scalar(f'Worker_{self.rank}/Average_Reward', avg_reward, episode)
            self.writer.add_scalar(f'Worker_{self.rank}/Episode_Length', episode_steps, episode)
            
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns).to(self.device )
            
            values = torch.cat(values)
            log_probs = torch.stack(log_probs)
            advantages = returns - values.detach()
            
            # Calculate losses
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            total_loss = actor_loss + 0.5 * critic_loss
            
            self.writer.add_scalar(f'Worker_{self.rank}/Actor_Loss', actor_loss.item(), episode)
            self.writer.add_scalar(f'Worker_{self.rank}/Critic_Loss', critic_loss.item(), episode)
            self.writer.add_scalar(f'Worker_{self.rank}/Total_Loss', total_loss.item(), episode)
            
            # Update global model
            self.optimizer.zero_grad()
            total_loss.backward()
            
            for global_param, local_param in zip(self.global_model.parameters(), 
                                               self.local_model.parameters()):
                global_param._grad = local_param.grad
            
            self.optimizer.step()
            
            # Sync local model with global model
            self.local_model.load_state_dict(self.global_model.state_dict())
            
            if episode % 10== 0:
                torch.save(self.global_model.state_dict(), 
                         os.path.join(self.save_path, f'model_{episode}.pth'))
            print(f'Worker {self.rank}, Episode {episode}, Average Reward: {np.mean(rewards)}')
            #self.write_summary(self.rank, np.mean(rewards), total_loss.item(),actor_loss.item(), critic_loss.item(), episode)    
        self.writer.close()
    def write_summary(self, rank, rewards, total_loss,actor_loss ,critic_loss, episode):
        self.writer.add_scalar(f"reward", rewards, episode)
        self.writer.add_scalar("model/total_loss", total_loss, episode)
        self.writer.add_scalar("model/actor_loss", actor_loss, episode)
        self.writer.add_scalar("model/critic_loss", critic_loss, episode)
def train_a3c():
    global_model = ActorCritic().to(device)
    global_model.share_memory()
    
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)
    main_writer = SummaryWriter(os.path.join(save_path, 'main'))

    num_workers = 5#mp.cpu_count()
    print(f'Number of workers: {num_workers}')
    workers = []
    
    for rank in range(num_workers):
        worker = Worker(global_model, optimizer, rank, save_path,device)
        workers.append(worker)
        worker.start()
    
    for worker in workers:
        worker.join()
    main_writer.close()
if __name__ == '__main__':
    date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/A3C_only_pos_BUILD_EasyStage1_Only_Pos/{date_time}"
    
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능 여부에 따라 device 설정
    game = r"C:\AllProgramming\MyProjects\UnityProjects\RL_TheHardestGame\BUILD_EasyStage1_Visual_Position0"
                
    os.makedirs(save_path, exist_ok=True)
    train_a3c()