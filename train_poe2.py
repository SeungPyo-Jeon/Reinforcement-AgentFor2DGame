import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from collections import deque
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from tqdm import tqdm

# 멀티프로세싱을 위한 설정
mp.set_start_method('spawn', force=True)

class ActorCritic(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_x = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        self.actor_y = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        #print( x, x[[-4,-2]], x[[-3,-1]])
        shared_features = self.shared_layers(x[[-4,-2]])
        prob_x = self.actor_x(shared_features)
        shared_features = self.shared_layers(x[[-3,-1]])
        prob_y = self.actor_x(shared_features)
        value = self.critic(shared_features)
        return prob_x, prob_y, value

def worker_process(rank, shared_model, save_path, device ):
    # 각 워커별 환경 설정
    env_config_channel = EngineConfigurationChannel()
    env_config_channel.set_configuration_parameters(width=64, height=64)
    env_config_channel.set_configuration_parameters(target_frame_rate=60)
    env_config_channel.set_configuration_parameters(time_scale=10)
    game = r"C:\AllProgramming\MyProjects\UnityProjects\RL_TheHardestGame\BUILD_EasyStage1_Visual_Position"+ str(rank)
    env = UnityEnvironment(file_name=game, side_channels=[env_config_channel],base_port=5005+rank, no_graphics=True)
    env.reset()
    
    # 로컬 모델 생성
    local_model = ActorCritic().to(device)
    local_model.load_state_dict(shared_model.state_dict())
    
    # Optimizer 생성
    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
    
    # Tensorboard writer 초기화
    writer = SummaryWriter(os.path.join(save_path, f'worker_{rank}'))
    
    behavior_name = list(env.behavior_specs)[0]
    reward_window = deque(maxlen=100)
    
    def get_action(state, model):
        state = torch.FloatTensor(state).to(device)
        prob_x, prob_y, _ = model(state)
        
        dist_x = torch.distributions.Categorical(prob_x)
        dist_y = torch.distributions.Categorical(prob_y)
        
        action_x = dist_x.sample()
        action_y = dist_y.sample()
        
        return (action_x.item()-1, action_y.item()-1), \
               (dist_x.log_prob(action_x), dist_y.log_prob(action_y))
    
    for episode in range(500):  # max_episodes
        env.reset()
        dec, term = env.get_steps(behavior_name)
        state = dec.obs[1][0]
        
        done = False
        values = []
        log_probs = []
        rewards = []
        episode_steps = 0
        
        while not done:
            episode_steps += 1
            action, log_prob = get_action(state, local_model)
            
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
            
            _, _, value = local_model(torch.FloatTensor(state).to(device))
            values.append(value)
            log_probs.append(sum(log_prob))
            rewards.append(reward)
            
            state = next_state
        
        # 에피소드 종료 후 처리
        episode_reward = sum(rewards)
        reward_window.append(episode_reward)
        avg_reward = np.mean(reward_window)
        
        # Tensorboard 기록
        writer.add_scalar(f'Worker_{rank}/Episode_Reward', episode_reward, episode)
        writer.add_scalar(f'Worker_{rank}/Average_Reward', avg_reward, episode)
        writer.add_scalar(f'Worker_{rank}/Episode_Length', episode_steps, episode)
        
        # 학습
        returns = []
        R = 0
        gamma = 0.99
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)
        advantages = returns - values.detach()
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        total_loss = actor_loss + 0.5 * critic_loss
        
        writer.add_scalar(f'Worker_{rank}/Actor_Loss', actor_loss.item(), episode)
        writer.add_scalar(f'Worker_{rank}/Critic_Loss', critic_loss.item(), episode)
        writer.add_scalar(f'Worker_{rank}/Total_Loss', total_loss.item(), episode)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 글로벌 모델과 동기화
        with torch.no_grad():
            for shared_param, local_param in zip(shared_model.parameters(), local_model.parameters()):
                shared_param.data = local_param.data.clone()
        
        if episode % 3 == 0:
            torch.save(shared_model.state_dict(), 
                     os.path.join(save_path, f'model_{episode}.pth'))
            print(f'Worker {rank}, Episode {episode}, Average Reward: {avg_reward:.2f}')
    
    writer.close()
    env.close()

def train_a3c():
    # 공유 모델 생성
    shared_model = ActorCritic().to(device)
    shared_model.share_memory()
    
    processes = []
    num_workers = 1#mp.cpu_count()
    
    for rank in range(num_workers):
        p = mp.Process(target=worker_process, args=(rank, shared_model, save_path, device))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == '__main__':
    date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/A3C_only_pos_BUILD_EasyStage1_Only_Pos/{date_time}"
    
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능 여부에 따라 device 설정
    game = r"C:\AllProgramming\MyProjects\UnityProjects\RL_TheHardestGame\BUILD_EasyStage1_Visual_Position0"
             
    os.makedirs(save_path, exist_ok=True)
    train_a3c()