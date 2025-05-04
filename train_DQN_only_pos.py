import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
import cv2
import time 
from tqdm import tqdm

train_mode = True
load_model = not train_mode

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

state_size = [3, 128, 200]
action_size = (2,3)

batch_size = 128 # 한번 모델을 학습할 때 리플레이 메모리에서 꺼내는 경험 데이터 수
mem_maxlen = 10000 # 만약 10000개 이상 쌓이면 가장 오래된 데이터를 제거
discount_factor = 0.9 # gamma
learning_rate = 0.00025 # 네트워크의 learning rate
        
run_step = 50000 if train_mode else 0 # 학습 모드에서 진행할 스텝 수 
test_step = 5000 # 평가모드에서 진행할 스텝 수
train_start_step = 5000 # 학습 시작 전에 리플레이 메모리에 충분한 데이터를 모으기 위해 몇 스텝 동안 임의의 행동으로 게임 진행할 것인지 
target_update_step = 500 # 타겟 네트워크를 몇 스텝 주기로 갱신할 것인지 

print_interval = 10 # 텐서보드에 기록할 주기 설정
save_interval = 10 # 학습 모델을 저장할 에피소드 주기 설정

epsilon_eval = 0.05 # 평가모드의 epsilon 값. 평가모드에서는 5%의 확률로 랜덤하게 이동한다. (탐색한다)
epsilon_init = 1.0 if train_mode else epsilon_eval # 초기 epsioon 값. 학습모드 일때 처음에 탐색하는 비율.
epsilon_min = 0.1
explore_step = run_step * 0.8 # epsilon이 감소되는 구간
epsilon_delta = (epsilon_init - epsilon_min) / explore_step if train_mode else 0 # 한 스텝당 감소하는 epsilon 변화량

VISUAL_OBS = 0 
OBS = VISUAL_OBS
# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/DQN_only_pos_BUILD_EasyStage1_Only_Pos/{date_time}"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
load_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/DQN_only_pos_BUILD_EasyStage1_Only_Pos/20241210023135"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능 여부에 따라 device 설정

game = r"C:\AllProgramming\MyProjects\UnityProjects\RL_TheHardestGame\BUILD_EasyStage1_Visual_Position"

    
class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        ## input shape: (1, 16)
        input_dim = 2
        hidden_dim = 128
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
    def forward(self, x, Traing=False):
        
        shared_features = self.shared_layers(x[:,[-4,-2]])
        prob_x = self.actor_x(shared_features)
        shared_features = self.shared_layers(x[:,[-3,-1]])
        prob_y = self.actor_x(shared_features)
        out = torch.stack([prob_x, prob_y], dim = -2)
        return out
    
    #    self.fc_out1 = torch.nn.Linear(16, 512)
    #    self.bath_norm1 = torch.nn.BatchNorm1d(512)
    #    self.fc_out2 = torch.nn.Linear(512, 256)
    #    self.bath_norm2 = torch.nn.BatchNorm1d(256)
    #    self.fc_out3 = torch.nn.Linear(256, 32)
    #    self.bath_norm3 = torch.nn.BatchNorm1d(32)
    #    self.fc_out4 = torch.nn.Linear(32, 6)
    #def forward(self, x, Training = False):
    #    #print( x.shape)
    #    x1 = self.fc_out1(x)
    #    x1 = F.relu(x1)
    #    x1 = self.bath_norm1(x1)
    #    x2 = self.fc_out2(x1)
    #    x2 = F.relu(x2)
    #    x2 = self.bath_norm2(x2)
    #    x3 = self.fc_out3(x2)
    #    x3 = F.relu(x3)
    #    x3 = self.bath_norm3(x3)
    #    x4 = self.fc_out4(x3)
#
    #    x4 = x4.view(x4.size(0), action_size[0], action_size[1])
    #    return x4
    
class DQNAgent:
    def __init__(self):
        self.dqn = DQN().to(device)
        self.target_dqn = copy.deepcopy(self.dqn).to(device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen )
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)

        if load_model:
            self.load_model(load_path)
    def get_q_value( self, state, training =True):
        self.dqn.train(training)
        epsilon = self.epsilon if training else epsilon_eval

        if random.random() <= epsilon:
            q_values = np.random.rand( state.shape[0],action_size[0], action_size[1])
        else:
            state = torch.FloatTensor(state).to(device)
            q_values = self.dqn(state).cpu().detach().numpy()
        return q_values
    def get_action(self,  state, training =True):
        #self.dqn.train(training)
        epsilon = self.epsilon if training else epsilon_eval
        self.dqn.train(training)
        #epsilon = self.epsilon
        
        if random.random() <= epsilon:
            q_values = np.random.rand( state.shape[0],action_size[0], action_size[1])
        else:
            state = torch.FloatTensor(state).to(device)
            q_values = self.dqn(state).cpu().detach().numpy()

        action = q_values.argmax(-1).reshape(-1,action_size[0])
        return action
    def append_sample(self, state, action, reward, next_state, done ):
        self.memory.append((state, action, reward, next_state, done))
    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state = np.stack([sample[0] for sample in batch], axis=0)
        action = np.stack([sample[1] for sample in batch], axis=0)
        reward = np.stack([sample[2] for sample in batch], axis=0)
        next_state = []
        for sample in batch :
            #print('sample[3].shape: ', sample[3].shape )
            next_state.append( sample[3])
        next_state = np.array(next_state, np.uint8)
        done = np.stack([sample[4] for sample in batch], axis=0)

        state, action, reward, next_state, done = map( lambda x :
                                         torch.FloatTensor(x).to(device), 
                                [state, action, reward, next_state, done])
        eye = torch.eye(action_size[1]).to(device)

        one_hot_action1 = eye[action[:,0].view(-1).long()]
        one_hot_action2 = eye[action[:,1].view(-1).long()]
        q = self.dqn(state, True)
        #print('q.shape: ', q.shape)
        q1 = (q[:,0,:] * one_hot_action1).sum(1, keepdim=True)
        #print('q1.shape: ', q1.shape, q1)
        q2 = (q[:,1,:] * one_hot_action2).sum(1, keepdim=True)
        #print('reward.shape: ', reward.shape)
        #print('done.shape: ', done.shape)
        with torch.no_grad():
            target_q = self.target_dqn(next_state, True)
            #print('target_q.shape: ',  target_q.shape, target_q[:,0,:].max(dim= -1)[0].shape)
            target_q1  =  reward.reshape(-1,1)  +((1 - done) * discount_factor )\
                                  * target_q[:,0,:].max(dim= -1)[0].reshape(-1,1)
            #print('target_q1.shape: ', target_q1.shape, target_q1)
            target_q2  =  reward.reshape(-1,1)+((1 - done) * discount_factor )\
                                  *target_q[:,1,:].max(dim= -1)[0].reshape(-1,1)
        
        loss1 = F.smooth_l1_loss(q1, target_q1) 
        #print('loss1: ', loss1)
        loss2 = F.smooth_l1_loss(q2, target_q2)
        loss = loss1 + loss2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)
        return loss1.item(), loss2.item(), loss.item()
    def update_target_model(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
    def load_model(self, path):
        print(f"Load Model from {path}/ckpt")
        checkpoint = torch.load(f"{path}/ckpt", map_location=device)
        self.dqn.load_state_dict(checkpoint['dqn'])
        self.target_dqn.load_state_dict(checkpoint['dqn'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        pass
    def save_model(self, path):
        print(f"Save Model to {path}/ckpt")
        torch.save({
            'dqn': self.dqn.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"{path}/ckpt")
        pass    
    def write_summary(self, score, loss1,loss2,loss, step ):
        self.writer.add_scalar("score", score, step)
        self.writer.add_scalar("loss1", loss1, step)
        self.writer.add_scalar("loss2", loss2, step)
        self.writer.add_scalar("loss", loss, step)
        self.writer.add_scalar("epsilon", self.epsilon, step)
        
    
if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel() # Create the engine configuration channel and set the base resolution of the brain to 32x32
    engine_configuration_channel.set_configuration_parameters(width=512, height=256, target_frame_rate=60)
    env = UnityEnvironment(file_name = game, side_channels=[engine_configuration_channel]
                           , no_graphics=False) # 유니티 환경, side_channel은 해상도, timesclae, graphic quality 등을 설정할 때 사용
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0] # 브레인 이름 설정
    spec = env.behavior_specs[behavior_name] # 브레인 스펙 설정 - input/output
    agent = DQNAgent()
    dec, term = env.get_steps(behavior_name) # 환경에서 관측 정보를 가져옴
    loss1es,loss2es,losses, scores, episode, score = [],[],[], [], 0, 0  # 손실, 점수, 에피소드 기록 리스트
    engine_configuration_channel.set_configuration_parameters(time_scale = 1.0) # 환경 속도 설정
    for step in tqdm(range(run_step + test_step)):
        
        if step == run_step: # 학습 스텝 완료 시 모델 저장&Test mode 시작
            if train_mode:
                agent.save_model(save_path)
            print( 'Test mode starts...')
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale = 1.0) # 환경 속도 설정
        
        state = dec.obs[1]
        action = agent.get_action(state, not train_mode)
        
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        env.set_action_for_agent(behavior_name, 0, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name) # 환경에서 관측 정보를 가져옴
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        score += reward[0]
        next_state = dec.obs[1]
        if next_state.shape[0] == 0:
            next_state = np.ones_like(state)
            next_state[:,-2:] = 0
            #print('state.shape: ', state.shape)
            #print('next_state.shape: ', next_state.shape)

        if train_mode: # train 시 리플레이 메모리에 데이터 저장
            agent.append_sample(state[0], action[0], reward[0], next_state[0], [done])
            if step >train_start_step: # 일정량 메모리 저장시 학습 시작
                loss1,loss2,loss = agent.train_model()
                losses.append(loss)
                loss1es.append(loss1)
                loss2es.append(loss2)
                if step % target_update_step == 0: # 타겟 네트워크 업데이트
                    agent.update_target_model()

        if done: # 에피소드 종료 시
            scores.append(score)
            score = 0
            episode += 1
            if True:#episode % print_interval == 0:
                print(f"step: {step}, episode: {episode}, score: {np.mean(scores):.2f},\
                        loss1: {np.mean(loss1es):.4f}, loss2:{np.mean(loss2es):.4f}, loss: {np.mean(losses):.4f}, epsilon: {agent.epsilon:.2f}")
                agent.write_summary(np.mean(scores),np.mean(loss1es),np.mean(loss1es), np.mean(losses), step)
                scores, loss1es, loss2es,losses = [], [],[],[]
            if episode % save_interval == 0 and train_mode:
                agent.save_model(save_path)
            env.reset()
            dec, term = env.get_steps(behavior_name)

    env.close()