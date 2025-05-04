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
import os
import cv2
import time 
from tqdm import tqdm

train_mode = False
load_model = not train_mode

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

state_size = [3, 128, 200]
action_size = (2,3)

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

VISUAL_OBS = 0 
OBS = VISUAL_OBS
# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/DQN/{date_time}"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
load_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/DQN/20241206185504"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능 여부에 따라 device 설정

game = r"C:\AllProgramming\MyProjects\UnityProjects\RL_TheHardestGame\BUILD_EasyStage1_Only_Visual"

class ResidualBloack(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBloack, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding= padding)
        self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, 1, padding= 1)
        self.batch_norm2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, 3, 1, padding= 1)
        self.batch_norm3 = torch.nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.silu(self.batch_norm1(x1))

        x1_2 = self.conv2(x1)
        x1_2 = F.silu(self.batch_norm2(x1_2))

        x1_2_3 = self.conv3(x1+x1_2)
        x1_2_3 = F.silu(self.batch_norm3(x1_2_3))
    
        return x1_2_3 + x1_2
    
class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        ## input shape: (3, 128, 256)
        self.residual_block1 = ResidualBloack(3, 32, 3, 1, 1) #( 32, 128, 256)
        self.residual_block2 = ResidualBloack(32, 64, 3, 2, 1) #( 64, 64, 128)
        self.residual_block3 = ResidualBloack(64, 64, 3, 2, 1) #( 128, 32, 64)
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1,1)) #(128, 1, 1)
        self.fc = torch.nn.Linear(64, 32)
        self.fc_out1 = torch.nn.Linear(32, 3)
        self.fc_out2 = torch.nn.Linear(32, 3)
    def forward(self, x, Training = False):
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.global_avg_pool(x)
        x = x.view( x.shape[:-2], -1)
        x = self.fc(x)
        x1 = self.fc_out1(x)
        x2 = self.fc_out2(x)
        #print(x1.shape, x2.shape)
        x = torch.stack([x1, x2], dim=-2)
        #print(x.shape)
        return x

class DQNAgent:
    def __init__(self):
        self.dqn = DQN().to(device)
        self.target_dqn = copy.deepcopy(self.dqn).to(device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen )
        self.epsilon = epsilon_init
        self.writer = SummaryWriter()
        self.loss1 = torch.nn.KLDivLoss( reduction='batchmean')
        self.loss2 = torch.nn.KLDivLoss( reduction='batchmean')

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
            #print( torch.is_tensor(q_values))
            #action = q_value.argmax(-1).cpu().numpy().reshape(-1,action_size[0])
        return q_values
    def get_action(self, q_values):
        action = q_values.argmax(-1).reshape(-1,action_size[0])
        return action
    def append_sample(self, state, q_values, reward, next_state, done ):
        self.memory.append((state, q_values, reward, next_state, done))
    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state = np.stack([sample[0] for sample in batch], axis=0)
        q_values = np.stack([sample[1] for sample in batch], axis=0)
        reward = np.stack([sample[2] for sample in batch], axis=0)
        next_state = []
        for sample in batch :
            #print('sample[3].shape: ', sample[3].shape )
            next_state.append( sample[3])
        next_state = np.array(next_state, np.uint8)
        done = np.stack([sample[4] for sample in batch], axis=0)

        state, q_values, reward, next_state, done = map( lambda x :
                                         torch.FloatTensor(x).to(device), 
                                [state, q_values, reward, next_state, done])
        q = self.dqn(state)
        ones = torch.ones(action_size).to(device)
        reward = torch.stack([ ones*reward[i] for i in range(reward.shape[0])]).to(device)
        reward = (reward - reward.mean())
        done = torch.stack([ ones*done[i] for i in range(done.shape[0])]).to(device)
        #print('reward: ', reward.shape , reward)
        #print('done: ', done.shape, done)

        #print( 'state: ', state.shape, state)
        #print('q: ', q.shape, q)
        #print( 'q_values: ', q_values.shape, q_values)
        with torch.no_grad():
            target_q = self.target_dqn(next_state)
            #print('target_q: ', target_q.shape, target_q)
            #print("target_q.shape", target_q.shape)
            #print("done.shape", done.shape)
            #print("reward.shape", reward.shape)
            #print( reward.shape)
            target  =  (1 - done) * discount_factor 
            #print('1:', target.shape, target_q.shape)
            target = target * target_q
            target = reward+ target
        #k1_loss1_1 = self.loss1( torch.log(q[:,:,:]), target[:,:,:])
        #k1_loss1_2 = self.loss2( torch.log(target[:,:,:]), q[:,:,:])
        #loss1 = k1_loss1_1 + k1_loss1_2
        #k1_loss2_1 = self.loss1( torch.log(q[:,1,:]), target[:,1,:])
        #k1_loss2_2 = self.loss2( torch.log(target[:,1,:]), q[:,1,:])
        #loss2 = k1_loss2_1 + k1_loss2_2
        #loss = loss1 #(loss1 + loss2)/4
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        #k1_loss1_1.backward()
        #k1_loss1_2.backward()
        #k1_loss2_1.backward()
        #k1_loss2_2.backward()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)
        #print('loss_1: ', k1_loss1_1.shape, k1_loss1_1.item(), k1_loss1_1)
        #print('loss_2: ', k1_loss1_2.shape, k1_loss1_2.item(), k1_loss1_2)
        #print('loss:', loss.shape, loss.item(), loss )
        return loss.item()
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
    def write_summary(self, score, loss, step ):
        self.writer.add_scalar("score", score, step)
        self.writer.add_scalar("loss", loss, step)
        self.writer.add_scalar("epsilon", self.epsilon, step)
        self.writer.flush()
    
if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel() # Create the engine configuration channel and set the base resolution of the brain to 32x32
    engine_configuration_channel.set_configuration_parameters(width=256, height=256)
    env = UnityEnvironment(file_name = game, side_channels=[engine_configuration_channel]) # 유니티 환경, side_channel은 해상도, timesclae, graphic quality 등을 설정할 때 사용
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0] # 브레인 이름 설정
    spec = env.behavior_specs[behavior_name] # 브레인 스펙 설정 - input/output
    agent = DQNAgent()
    
    dec, term = env.get_steps(behavior_name) # 환경에서 관측 정보를 가져옴
    losses, scores, episode, score = [], [], 0, 0  # 손실, 점수, 에피소드 기록 리스트
    engine_configuration_channel.set_configuration_parameters(time_scale = 10.0) # 환경 속도 설정
    for step in tqdm(range(run_step + test_step)):
        
        if step == run_step: # 학습 스텝 완료 시 모델 저장&Test mode 시작
            if train_mode:
                agent.save_model(save_path)
            print( 'Test mode starts...')
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale = 1.0) # 환경 속도 설정
        
        state = dec.obs[OBS][:,:3,:,:]

        action = np.array([[1,2]]) #action:  (1, 2) [[2 2]]
        q_values = agent.get_q_value(state, train_mode)
        action = agent.get_action(q_values)
        #print(action.shape ,action )
        
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action)
        env.set_action_for_agent(behavior_name, 0, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name) # 환경에서 관측 정보를 가져옴
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        score += reward[0]
        next_state = dec.obs[OBS][:,:3,:,:]
        if next_state.shape[0] == 0:
            next_state = np.ones_like(state)
        
        #img = (255*next_state.reshape(next_state.shape[2],next_state.shape[3],next_state.shape[1])).astype(np.uint8)
        #for i in range(4):
        #    cv2.imwrite(f'img{i}.png',cv2.cvtColor(img[:,:,i*3:i*3+3],cv2.COLOR_BGR2RGB)) 
        
        if train_mode: # train 시 리플레이 메모리에 데이터 저장
            agent.append_sample(state[0], q_values[0], reward[0], next_state[0], [done])

            if step > train_start_step: # 일정량 메모리 저장시 학습 시작
                loss = agent.train_model()
                losses.append(loss)
                if step % target_update_step == 0: # 타겟 네트워크 업데이트
                    agent.update_target_model()
        if done: # 에피소드 종료 시
            scores.append(score)
            score = 0
            episode += 1
            if True:#episode % print_interval == 0:
                print(f"step: {step}, episode: {episode}, score: {np.mean(scores):.2f},\
                       loss: {np.mean(losses):.4f}, epsilon: {agent.epsilon:.2f}")
                agent.write_summary(np.mean(scores), np.mean(losses), step)
                scores, losses = [], []
            if episode % save_interval == 0 and train_mode:
                agent.save_model(save_path)
            env.reset()
            dec, term = env.get_steps(behavior_name)

    env.close()