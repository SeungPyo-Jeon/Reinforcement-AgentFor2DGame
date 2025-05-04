import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import cv2 as cv
# UnityEnvironment : 유니티로 만든 환경을 불러올 때 사용
# ActionTuple : 액션을 환경에 전달하기 위한 행동 객체
# EngineConfigurationChannel : 유니티 환경 타임 스케일 조절

state_size = [3, 128, 200] # DQN 에이전트의 입력으로 사용할 상태의 크기
action_size = [2,3] # DQN 에이전트의 출력으로 사용할 행동의 크기

load_model = False
train_mode =  True

batch_size = 32 # 한번 모델을 학습할 때 리플레이 메모리에서 꺼내는 경험 데이터 수
mem_maxlen = 10000 # 만약 10000개 이상 쌓이면 가장 오래된 데이터를 제거
discount_factor = 0.9 # gamma
learning_rate = 0.00025 # 네트워크의 learning rate
        
run_step = 50000 if train_mode else 0 # 학습 모드에서 진행할 스텝 수 
test_step = 5000 # 평가모드에서 진행할 스텝 수
train_start_step = 5000 # 학습 시작 전에 리플레이 메모리에 충분한 데이터를 모으기 위해 몇 스텝 동안 임의의 행동으로 게임 진행할 것인지 
target_update_step = 500 # 타겟 네트워크를 몇 스텝 주기로 갱신할 것인지 

print_interval = 10 # 텐서보드에 기록할 주기 설정
save_interval = 100 # 학습 모델을 저장할 에피소드 주기 설정

epsilon_eval = 0.05 # 평가모드의 epsilon 값. 평가모드에서는 5%의 확률로 랜덤하게 이동한다. (탐색한다)
epsilon_init = 1.0 if train_mode else epsilon_eval # 초기 epsioon 값. 학습모드 일때 처음에 탐색하는 비율.
epsilon_min = 0.1
explore_step = run_step * 0.8 # epsilon이 감소되는 구간
epsilon_delta = (epsilon_init - epsilon_min) / explore_step if train_mode else 0 # 한 스텝당 감소하는 epsilon 변화량


VISUAL_OBS = 0 # 시각적 관측 인덱스
#GOAL_OBS = 1 # 목적지 관측 인덱스
#VECTOR_OBS = 2 # 수치적 관측 인덱스
OBS = VISUAL_OBS # DQN에서는 시각적 관측 인덱스를 사용함으로써 VISUAL_OBS로 설정

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/DQN/{date_time}"
import os
#if os.path.exists(save_path) == False:
#    os.makedirs(save_path)
load_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/DQN/20241201080721"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능 여부에 따라 device 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3) # 32개의 8x8 필터를 stride 4로 적용
            # 256,128,3 -> 64,32,32
        self.batch_norm1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1) # 64개의 4x4 필터를 stride 2로 적용
            # 64,32,32 -> 32,16,64
        self.batch_norm2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1) # 64개의 3x3 필터를 stride 1로 적용
            # 32,16,64 -> 32,16,128
        self.batch_norm3 = torch.nn.BatchNorm2d(64)

        self.fc1 = torch.nn.Linear(32 * 16 * 128, 2048) # 7x7x64 크기의 입력을 512 크기의 출력으로 변환
        self.fc2_h = torch.nn.Linear(512, action_size[1]) # 512 크기의 입력을 action_size 크기의 출력으
        self.fc2_v = torch.nn.Linear(512, action_size[1])
        self.q = torch.nn.Linear(512, action_size[0]*action_size[1])

    def forward(self, x):
        #print('init x.shape: ', x.shape )
        x = x #/ 255#x.permute(0, 3, 1, 2).float() / 255.0 # 이미지 데이터를 0~1 사이의 값으로 정규화
        x = F.relu(self.conv1(x)) # ReLU 활성화 함수를 적용
        x = self.batch_norm1(x)
        #print('conv1 x.shape: ', x.shape )
        x = F.relu(self.conv2(x)) # ReLU 활성화 함수를 적용
        x = self.batch_norm2(x)
        #print('conv2 x.shape: ', x.shape )
        x = F.relu(self.conv3(x)) # ReLU 활성화 함수를 적용
        x = self.batch_norm3(x)
        #print('conv3 x.shape: ', x.shape )
        x = x.view(x.size(0), -1) # 1차원 텐서로 변환
        x = F.relu(self.fc1(x))
        #print('FCN x.shape: ', x.shape )
        #x_v = self.fc2_h(x)
        #x_h = self.fc2_v(x)
        return self.q(x).reshape(x.shape[0], action_size[0], action_size[1])
        
class DQNAgent:
    def __init__(self):
        self.network = DQN().to(device) # DQN 네트워크 생성
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate) # Adam 옵티마이저 생성
        self.memory = deque(maxlen=mem_maxlen) # 리플레이 메모리 생성
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)

        if load_model:
            print(f'... Load model from {load_path}/ckpt...')
            checkpoint = torch.load(load_path+'/ckpt',map_location=device)
            self.network.load_state_dict(checkpoint['network'])
            self.target_network.load_state_dict(checkpoint['network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def get_action(self, state, training = True ):
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval

        if epsilon > random.random():
            action = np.random.randint(0, action_size[1],size=[ state.shape[0] if state.shape[0] else 1,action_size[0]] )
            #print('random action: ', action.shape, action)
        else:
            q = self.network(torch.FloatTensor(state).to(device))
            #print('q: ', q.shape, q)
            q = q.reshape(q.shape[0], action_size[0], action_size[1])
            #q_h = q[:,0,:]
            #print('q_h: ', q_h.shape, q_h)
            #q_v = q[:,1,:]
            #q = (q_h+q_v).sum(-1, keepdim=True)
            #print('q: ', q.shape, q)
            action = q.argmax(-1).cpu().numpy().reshape(-1,action_size[0])
            #print('model action: ', action.shape, action)
        return action
    
    def append_sample(self, state, action, reward, next_state, done):
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
        #print( 'next_state.shape: ',next_state.shape)
        #next_state = np.stack([sample[3] for sample in batch], axis=0)
        done = np.stack([sample[4] for sample in batch], axis=0)

        state, action, reward, next_state, done = map( lambda x :
                                         torch.FloatTensor(x).to(device), 
                                [state, action, reward, next_state, done])
        eye = torch.eye(action_size[1]).to(device)
        #print('eye', eye.shape, eye)
        #print( 'action.shape: ', action.shape)
        #print('action[:,0]: ', action[:,0])
        #print( 'action[:,0].view(-1).long(): ', action[:,0].view(-1).long())
        one_hot_action_h = eye[action[:,0].view(-1).long()]
        #print( 'one_hot_action_h', one_hot_action_h.shape, one_hot_action_h)
        eye2 = torch.eye(action_size[1]).to(device)
        #print( 'action[:,1].view(-1).long(): ', action[:,1].view(-1).long())
        one_hot_action_v = eye2[action[:,1].view(-1).long()]
        #print( 'one_hot_action_v', one_hot_action_v.shape, one_hot_action_v)
        network_out = self.network(state)
        network_out = network_out.reshape( network_out.shape[0], action_size[0], action_size[1] )
        #print('network_out: ', network_out.shape, network_out)
        #print('network_out[:,0,: ]: ', network_out[:,0,: ].squeeze(1).shape, network_out[:,0,: ].squeeze(1))  
        """ ver1 
        q_h = network_out[:,0,: ].squeeze(1) * one_hot_action_h
        #print('q_h: ', q_h.shape, q_h)
        q_v = network_out[:,1,: ].squeeze(1) * one_hot_action_v
        q = (q_h+q_v).sum(-1, keepdim=True)
        #print('q: ', q.shape, q)
        #q = ( network_out.reshape( network_out.shape[0], action_size[0], action_size[1] )
        #            *np.stack([one_hot_action_h,one_hot_action_v])).sum((-1,-2), keepdim=True)
        """
        q_h = network_out[:,0,: ].squeeze(1) * one_hot_action_h
        q_v = network_out[:,1,: ].squeeze(1) * one_hot_action_v
        
        with torch.no_grad():
            """ ver1 
            next_q = self.target_network(next_state)
            next_q = next_q.reshape(next_q.shape[0],action_size[0],action_size[1])
            target_q = reward+ (next_q[:,0,:]+next_q[:,1,:]).squeeze(1).max( -1, keepdims=True).values*((1-done) *discount_factor)
            """
            next_q = self.target_network(next_state)
            next_q = next_q.reshape(next_q.shape[0],action_size[0],action_size[1])
            target_q_h_index = next_q[:,0,:].squeeze(1).argmax( -1, keepdims=True)
            eye = torch.eye(action_size[1]).to(device)
            one_hot_action_h = eye[target_q_h_index.view(-1).long()] * reward
            target_q_h = (next_q[:,0,:].squeeze(1) + one_hot_action_h)*((1-done) *discount_factor)
            target_q_v_index = next_q[:,1,:].squeeze(1).argmax( -1, keepdims=True)
            eye2 = torch.eye(action_size[1]).to(device)
            one_hot_action_v = eye2[target_q_v_index.view(-1).long()] * reward
            target_q_v = (next_q[:,1,:].squeeze(1) + one_hot_action_v)*((1-done) *discount_factor)
        loss = F.smooth_l1_loss(q_h, target_q_v) + F.smooth_l1_loss(q_v, target_q_v)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)

        return loss.item()
    
    def update_target_model( self ):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self):
        print(f'... Save model at {save_path}/ckpt...')
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, save_path+'/ckpt')

    def write_summary(self, score, loss, step):
        self.writer.add_scalar('score', score, step)
        self.writer.add_scalar('loss', loss, step)
        self.writer.add_scalar('epsilon', self.epsilon, step)

# main 함수
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel() # 유니티 엔진 설정 채널
    env = UnityEnvironment(file_name = r"C:\AllProgramming\MyProjects\UnityProjects\RL_TheHardestGame\BUILD", side_channels=[engine_configuration_channel]) # 유니티 환경, side_channel은 해상도, timesclae, graphic quality 등을 설정할 때 사용
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0] # 브레인 이름 설정
    #print( 'list(env.behavior_specs)_len: ',env.behavior_specs.keys()) 
    spec = env.behavior_specs[behavior_name] # 브레인 스펙 설정
    #print( 'spec.observation_shapes: ',spec ) # (12, 128, 200)
    #action_spec : discrete_branches=(3, 3)
    dec, term = env.get_steps(behavior_name) # 환경에서 관측 정보를 가져옴

    agent = DQNAgent() # DQN 에이전트 생성

    losses, scores, episode, score = [], [], 0, 0  # 손실, 점수, 에피소드 기록 리스트
    for step in range(run_step + test_step):
        engine_configuration_channel.set_configuration_parameters(time_scale = 1.0) # 환경 속도 설정
        if step == run_step:
            if train_mode:
                agent.save_model() # 학습 모드에서 학습이 끝나면 모델을 저장
            print('Test mode')
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale = 1.0) # 평가 모드에서 환경 속도 설정
        state = dec.obs[OBS][:,:3,:,:]
        #img = state.reshape(4,3,128,200)
        #print('state:', state.shape )
        action = agent.get_action(state, train_mode) # DQN 에이전트로부터 행동을 가져옴
        print( 'action: ',action.shape, action)
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        env.set_action_for_agent(behavior_name, 0, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name) # 환경에서 관측 정보를 가져옴
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = dec.obs[OBS][:,:3,:,:]
        #print('reward: ', reward)
        score += reward[0]

        if train_mode:
            #print( 'state: ', state.shape, 'action: ', action.shape, 'reward: ', reward, 'next_state: ', next_state.shape, 'done: ', done)
            if done:
                agent.append_sample(state[0], action[0], reward, np.zeros(state.shape[1:]), [done])
            else:
                agent.append_sample(state[0], action[0], reward, next_state[0], [done])
        if train_mode and step > max( batch_size, train_start_step):
            loss = agent.train_model()
            losses.append(loss)

            if step % target_update_step == 0:
                agent.update_target_model()
        if done:
            scores.append(score)
            episode += 1
            score = 0
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_loss = np.mean(losses)
                agent.write_summary(mean_score, mean_loss, step)
                losses,scores = [],[]

                print('step: ', step, '/ episode: ', episode,
                    f'/ score:  {mean_score:.2f}',f'loss: {mean_loss:.4f}',
                    f'epsilon: {agent.epsilon:.4f}')
            if train_mode and step % save_interval == 0:
                agent.save_model()
            env.reset()
            dec, term = env.get_steps(behavior_name) # 환경에서 관측 정보를 가져옴


env.close()