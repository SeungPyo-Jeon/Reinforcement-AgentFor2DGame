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

train_mode = not True
load_model = train_mode

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

state_size = [3, 128, 200]
action_size = (2,3)

batch_size = 64 # 한번 모델을 학습할 때 리플레이 메모리에서 꺼내는 경험 데이터 수
mem_maxlen = 10000 # 만약 10000개 이상 쌓이면 가장 오래된 데이터를 제거
discount_factor = 0.9 # gamma
learning_rate = 0.0025 # 네트워크의 learning rate
        
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

VISUAL_OBS = 0 
OBS = VISUAL_OBS
# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/A2C_EasyStage1_Visual_Position/{date_time}"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
load_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/A2C_EasyStage1_Visual_Position/20241209030707"
torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능 여부에 따라 device 설정

game = r"C:\AllProgramming\MyProjects\UnityProjects\RL_TheHardestGame\BUILD_EasyStage1_Visual_Position"

class A2C(nn.Module):
    def __init__(self, input_shape=(2, 128, 256), lstm_hidden_size=256, lstm_num_layers=1):
        super(A2C, self).__init__()
        
        # 1. Depthwise Separable Convolution 사용
        self.conv1_dw = nn.Sequential(
            nn.Conv2d(input_shape[0], input_shape[0], kernel_size=3, stride=1, groups=input_shape[0]),
            nn.Conv2d(input_shape[0], 16, kernel_size=1)  # 채널 수 감소 (32 -> 16)
        )
        
        self.conv2_dw = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, groups=16),
            nn.Conv2d(16, 32, kernel_size=1)  # 채널 수 감소 (64 -> 32)
        )
        
        self.conv3_dw = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, groups=32),
            nn.Conv2d(32, 64, kernel_size=1)  # 채널 수 감소 (128 -> 64)
        )
        
        # 2. Global Average Pooling 유지 (파라미터가 없는 레이어)
        #self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        h = ((input_shape[1]-2)//2 - 1)//2 - 1
        w = ((input_shape[2]-2)//2 - 1)//2 - 1
        self.cnn_output_size = 64 * h * w
        #print(self.cnn_output_size)
        self.fc_cnn = nn.Linear(self.cnn_output_size, 128)  # CNN 출력을 LSTM 입력 크기로 조정

        # 3. LSTM 크기 축소
        self.lstm = nn.LSTM(
            input_size=128,  # CNN 출력 채널 수와 맞춤
            hidden_size=lstm_hidden_size,  # 128 -> 64
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        # 4. FC 레이어에 작은 중간 레이어 추가하여 파라미터 감소
        self.fc_common = nn.Linear(lstm_hidden_size, 128)  # 공통 피처 추출
        self.fc_common2 = nn.Linear(128, 32)  # 공통 피처 추출
        self.fc_pi1 = nn.Linear(32, 3)
        self.fc_pi2 = nn.Linear(32, 3)
        self.fc_value = nn.Linear(32, 1)
        
        # 5. 추가: BatchNorm과 Dropout으로 일반화 성능 향상
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        time_steps = T
        assert C % 2 == 0, "Input channels must be divisible by 3."
        
        #x = x.view(B, time_steps, 2, H, W)
        
        cnn_features = []
        for t in range(time_steps):
            xt = x[:, t, :, :, :]
            
            # CNN with BatchNorm and ReLU
            xt = F.relu(self.bn1(self.conv1_dw(xt)))
            xt = F.relu(self.bn2(self.conv2_dw(xt)))
            xt = F.relu(self.bn3(self.conv3_dw(xt)))
            
            #xt = self.global_avg_pool(xt)
            #xt = xt.view(B, -1)
            #cnn_features.append(xt)
            xt = xt.view(B, -1)
            cnn_features.append(self.fc_cnn(xt))

        lstm_input = torch.stack(cnn_features, dim=1)
        
        # LSTM과 Dropout
        lstm_output, _ = self.lstm(lstm_input)
        lstm_last_output = lstm_output[:, -1, :]
        lstm_last_output = self.dropout(lstm_last_output)
        
        # 공통 피처 추출
        common_features =  F.relu(self.fc_common(lstm_last_output))
        common_features =  F.relu(self.fc_common2(common_features))
        common_features = self.dropout(common_features)
        
        
        # Policy와 Value 출력
        pi1 = F.softmax(self.fc_pi1(common_features), dim=1)
        pi2 = F.softmax(self.fc_pi2(common_features) , dim=1)
        v = self.fc_value(common_features)
        
        return pi1, pi2, v
class A2CAgent:
    def __init__(self):
        self.a2c = A2C().to(device)
        self.optimizer = torch.optim.Adam(self.a2c.parameters(), lr=learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=run_step, eta_min=0)
        self.memory = deque(maxlen=mem_maxlen )
        self.writer = SummaryWriter( save_path )

        if load_model:
            print(f"Load Model from {load_path}/ckpt ... ")
            checkpoint = torch.load(f"{load_path}/ckpt", map_location=device)
            self.a2c.load_state_dict(checkpoint['network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def get_action(self, state, traing=True):
        self.a2c.train(traing)
        state = torch.FloatTensor(state).to(device)
        pi1, pi2, v = self.a2c(state)

        action = np.stack([torch.multinomial(pi1, 1).cpu().numpy(), torch.multinomial(pi2, 1).cpu().numpy()], axis=-1).reshape(1,2)
        return action
    
    def append_sample(self, state, action, reward, next_state, done):
        if reward[0] > 0:
            self.memory.append((state, action, reward, next_state, done))
        else:
            if random.random() < 0.3:
                self.memory.append((state, action, reward, next_state, done))
    def train_model_batch(self):
        if len(self.memory) < batch_size:
            return 0, 0, 0
        batch = random.sample(self.memory, batch_size)
        state = np.stack([sample[0] for sample in batch], axis=0)
        action = np.stack([sample[1] for sample in batch], axis=0)
        reward = np.stack([sample[2] for sample in batch], axis=0)
        next_state = []
        for sample in batch :
            next_state.append( sample[3])
        next_state = np.array(next_state, np.uint8)
        done = np.stack([sample[4] for sample in batch], axis=0)

        state, action, reward, next_state, done = map( lambda x :
                                         torch.FloatTensor(x).to(device), 
                                [state, action, reward, next_state, done])
        pi1, pi2, value = self.a2c( state.squeeze() )
        with torch.no_grad():
            _,_,next_value = self.a2c( next_state.squeeze() )
            target = reward + (1-done) * discount_factor * next_value
        critic_loss = F.mse_loss(value, target)
        eye = torch.eye(action_size[1]).to(device)
        one_hot_action1 = eye[action[:,:,0].view(-1).long()]
        one_hot_action2 = eye[action[:,:,1].view(-1).long()]
        advantage = (target - value).detach()
        actor_loss1 = -(torch.log((one_hot_action1*pi1).sum(1)) * advantage).mean()
        actor_loss2 = -(torch.log((one_hot_action2*pi2).sum(1)) * advantage).mean()
        loss = critic_loss + actor_loss1 + actor_loss2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), actor_loss1.item(), actor_loss2.item()
    
    def train_model(self, state, action, reward, next_state, done):
        self.append_sample(state, action, reward, next_state, done)
        state, action, reward, next_state, done = map( lambda x : torch.FloatTensor(x).to(device),
                                                       [state, action, reward, next_state, done])
        
        pi1,pi2, value = self.a2c( state)
        with torch.no_grad():
            _,_,next_value = self.a2c( next_state )
            target = reward + (1-done) * discount_factor * next_value
        critic_loss = F.mse_loss(value, target)

        eye = torch.eye(action_size[1]).to(device)
        one_hot_action1 = eye[action[:,0].view(-1).long()]
        one_hot_action2 = eye[action[:,1].view(-1).long()]
        advantage = (target - value).detach()
        actor_loss1 = -(torch.log((one_hot_action1*pi1).sum(1)) * advantage).mean()
        actor_loss2 = -(torch.log((one_hot_action2*pi2).sum(1)) * advantage).mean()
        loss = critic_loss + actor_loss1 + actor_loss2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return critic_loss.item(), actor_loss1.item(), actor_loss2.item()
    
    def save_model(self,save_path):
        print(f"Save Model to {save_path}/ckpt")
        torch.save({
            'network': self.a2c.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"{save_path}/ckpt")

    def write_summary(self, score, actor_loss1, actor_loss2 ,critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss1", actor_loss1, step)
        self.writer.add_scalar("model/actor_loss2", actor_loss2, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)
    def write_batch( self, actor_loss1, actor_loss2 ,critic_loss, step):
        self.writer.add_scalar("batch/actor_loss1", actor_loss1, step)
        self.writer.add_scalar("batch/actor_loss2", actor_loss2, step)
        self.writer.add_scalar("batch/critic_loss", critic_loss, step)

# a 공간 범위
a_min = np.array([0.8, 0.8])
a_max = np.array([20.2, 12.2])   
# b 공간 범위
b_shape = (128, 256)  # 크기 (128, 256)
b_min = np.array([0, 0])
b_max = np.array([b_shape[1] - 1, b_shape[0] - 1])  # (127, 255)
         
def bilinear_interpolation_to_array(x, y):
    # a -> b 좌표 변환
    b_coord = (np.array([x, y]) - a_min) / (a_max - a_min) * (b_max - b_min) + b_min

    # b 공간에서 정수 좌표
    b_x1, b_y1 = np.floor(b_coord).astype(int)
    b_x2, b_y2 = np.ceil(b_coord).astype(int)

    # 보간 가중치 계산
    dx = b_coord[0] - b_x1
    dy = b_coord[1] - b_y1

    weights = {
        (b_x1, b_y1): (1 - dx) * (1 - dy),
        (b_x1, b_y2): (1 - dx) * dy,
        (b_x2, b_y1): dx * (1 - dy),
        (b_x2, b_y2): dx * dy,
    }
    
    # 출력 배열 생성
    result = np.zeros(b_shape, dtype= np.float32)
    for (bx, by), weight in weights.items():
        # b 공간의 범위를 벗어나는 경우 무시
        if 0 <= bx < b_shape[1] and 0 <= by < b_shape[0]:
            result[by, bx] = weight

    return result
imgs = []
cache_valid = False
if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel() # Create the engine configuration channel and set the base resolution of the brain to 32x32
    engine_configuration_channel.set_configuration_parameters(width=256, height=128)
    env = UnityEnvironment(file_name = game, side_channels=[engine_configuration_channel]
                                , no_graphics=True) # 유니티 환경, side_channel은 해상도, timesclae, graphic quality 등을 설정할 때 사용
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0] # 브레인 이름 설정
    spec = env.behavior_specs[behavior_name] # 브레인 스펙 설정 - input/output
    engine_configuration_channel.set_configuration_parameters(time_scale = 12.0) # 환경 속도 설정
    dec, term = env.get_steps(behavior_name) # 환경에서 관측 정보를 가져옴
    
    agent = A2CAgent()
    actor1_losses,actor2_losses,critic_losses, scores, episode, score = [],[], [], [], 0, 0  # 손실, 점수, 에피소드 기록 리스트
    batch_actor1_losses, batch_actor2_losses, batch_critic_losses = [], [], []
    for step in tqdm(range(run_step + test_step)):
        if step == run_step: # 학습 스텝 완료 시 모델 저장&Test mode 시작
            if train_mode:
                agent.save_model(save_path)
            print( 'Test mode starts...')
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale = 1.0) # 환경 속도 설정
        
        state = np.transpose(dec.obs[OBS][:,:,:,:],(0,2,3,1))
        position = dec.obs[1][:,:]
        
        if cache_valid:
            img = []
            imgs = imgs[-3:]
            img.append(cv2.cvtColor(state[0,:,:,-3:], cv2.COLOR_BGR2GRAY))
            pos1 = bilinear_interpolation_to_array(position[0,-4], position[0,-3])
            pos2 = bilinear_interpolation_to_array(position[0,-2], position[0,-1])*2
            img.append(pos1+pos2)
            imgs.append(img)
        else:
            imgs = []
            for i in range(4):
                img = []
                img.append(cv2.cvtColor(state[0,:,:,i:i+3], cv2.COLOR_BGR2GRAY))
                pos1 = bilinear_interpolation_to_array(position[0,0+(i*4)], position[0,1+(i*4)])
                pos2 = bilinear_interpolation_to_array(position[0,2+(i*4)], position[0,3+(i*4)])*2
                img.append(pos1+pos2)
                imgs.append(img)

        state = np.stack([imgs], axis=0)

        action = agent.get_action(state, train_mode)
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        env.set_action_for_agent(behavior_name, 0, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name) # 환경에서 관측 정보를 가져옴
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        score += reward[0]

        next_state = np.transpose(dec.obs[OBS][:,:,:,:],(0,2,3,1))
        position = dec.obs[1][:,:]
        imgs = imgs[-3:]
        img = []
        if next_state.shape[0] == 0:
            img.append(np.zeros((128,256)))
            pos1 = np.zeros((128,256))
            pos2 = np.zeros((128,256))
        else:
            img.append(cv2.cvtColor(next_state[0,:,:,-3:], cv2.COLOR_BGR2GRAY))
            pos1 = bilinear_interpolation_to_array(position[0,-4], position[0,-3])
            pos2 = bilinear_interpolation_to_array(position[0,-2], position[0,-1])*2
        img.append(pos1+pos2)
        imgs.append(img)
        next_state = np.stack([imgs], axis=0)
        cache_valid = True
        if train_mode: 
            critic_loss, actor1_loss, actor2_loss  = agent.train_model(state, action, [reward[0]], next_state, [done])
            actor1_losses.append(actor1_loss)
            actor2_losses.append(actor2_loss)
            critic_losses.append(critic_loss)
            if step != 0 and step % (batch_size*3) == 0:
                #critic_loss, actor1_loss, actor2_loss = agent.train_model_batch()
                #agent.write_batch(actor1_loss, actor2_loss, critic_loss, step)
                pass
        if done: # 에피소드 종료 시
            scores.append(score)
            score = 0
            episode += 1
            if True:#episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_actor1_loss = np.mean(actor1_losses) if len(actor1_losses) > 0 else 0
                mean_actor2_loss = np.mean(actor2_losses) if len(actor2_losses) > 0 else 0
                mean_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0
                agent.write_summary(mean_score, mean_actor1_loss, mean_actor2_loss, mean_critic_loss, step)
                actor1_losses, actor2_losses, critic_lossesm, scores = [], [], [], []
                print(f"step: {step}, episode: {episode}, score: {mean_score:.2f},\
                       actor1_loss: {mean_actor1_loss:.4f}, actor2_loss: {mean_actor2_loss:.4f}, critic_loss: {mean_critic_loss:.4f}")
            if episode % (save_interval//10) == 0 and train_mode:
                agent.save_model(save_path)
            env.reset()
            dec, term = env.get_steps(behavior_name)
            cache_valid = False
    env.close()