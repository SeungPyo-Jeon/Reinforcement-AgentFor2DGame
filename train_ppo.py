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

# 환경 설정
train_mode = True
load_model = not train_mode

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

state_size = [3, 128, 200]
action_size = (2, 3)

batch_size = 32
mem_maxlen = 10000
discount_factor = 0.99
learning_rate = 0.0003

run_step = 50000 if train_mode else 0
test_step = 5000
train_start_step = 5000
update_step = 512
epsilon_clip = 0.2  # PPO 클립 손실용

print_interval = 10
save_interval = 100

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.8

VISUAL_OBS = 0
OBS = VISUAL_OBS
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/PPO/{date_time}"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
load_path = f"C:/AllProgramming/MyProjects/UnityProjects/RL_TheHardestGame/saved_models/TheHardestGame/PPO/20241207234744"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = r"C:\AllProgramming\MyProjects\UnityProjects\RL_TheHardestGame\BUILD_EasyStage3_Visual_Position1"


class PPO(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PPO, self).__init__(**kwargs)
        # Input shape: (3, 128, 256)
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, stride=2)  # (64, 61, 125)
        self.conv3 = torch.nn.Conv2d(64, 64, 5, stride=2)  # (128, 29, 60)
        self.d1 = torch.nn.Linear(1769 * 64, 512)
        self.dropout = torch.nn.Dropout(0.2)
        self.d2 = torch.nn.Linear(512, 256)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.d3 = torch.nn.Linear(256, 128)
        self.pi1 = torch.nn.Linear(128, 3)
        self.pi2 = torch.nn.Linear(128, 3)
        self.v = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.d1(x))
        x = self.dropout(x)
        x = F.relu(self.d2(x))
        x = self.dropout2(x)
        x = F.relu(self.d3(x))
        pi1 = F.softmax(self.pi1(x), dim=-1)
        pi2 = F.softmax(self.pi2(x), dim=-1)
        v = self.v(x)
        return pi1, pi2, v


class PPOAgent:
    def __init__(self):
        self.ppo = PPO().to(device)
        self.optimizer = torch.optim.Adam(self.ppo.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(save_path)
        if load_model:
            print(f"Load Model from {load_path}/ckpt ... ")
            checkpoint = torch.load(f"{load_path}/ckpt", map_location=device)
            self.ppo.load_state_dict(checkpoint['network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.memory = deque(maxlen=mem_maxlen)
        self.clip_ratio = 0.2  # PPO clip range
        self.num_epochs = 4  # Number of epochs to optimize policy
        self.gae_lambda = 0.95  # GAE lambda

    def get_action(self, state, train=True):
        self.ppo.train(train)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        pi1, pi2, _ = self.ppo(state)
        action1 = torch.multinomial(pi1, 1).cpu().numpy()[0]
        action2 = torch.multinomial(pi2, 1).cpu().numpy()[0]
        return np.array([action1, action2], dtype=np.int32).reshape(1, 2)

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def train_model(self):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)

        # Compute GAE and returns
        _, _, values = self.ppo(states)
        _, _, next_values = self.ppo(next_states)
        td_target = rewards + (1 - dones) * discount_factor * next_values.squeeze()
        delta = td_target - values.squeeze()
        advantages = self.compute_gae(delta, dones)

        # Optimize policy for multiple epochs
        for _ in range(self.num_epochs):
            _, _, new_values = self.ppo(states)
            pi1, pi2, _ = self.ppo(states)

            log_probs1 = torch.log(pi1.gather(1, actions[:, 0].unsqueeze(1)))
            log_probs2 = torch.log(pi2.gather(1, actions[:, 1].unsqueeze(1)))
            new_log_probs = (log_probs1 + log_probs2).squeeze()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(new_values.squeeze(), td_target)
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()

    def compute_gae(self, delta, dones):
        advantages = []
        advantage = 0
        for d in reversed(range(len(delta))):
            advantage = delta[d] + discount_factor * self.gae_lambda * advantage * (1 - dones[d])
            advantages.insert(0, advantage)
        return torch.FloatTensor(advantages).to(device)

    def save_model(self, save_path):
        print(f"Save Model to {save_path}/ckpt")
        torch.save({
            'network': self.ppo.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"{save_path}/ckpt")


if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration_parameters(width=256, height=128)
    env = UnityEnvironment(file_name=game, base_port=5004+2,side_channels=[engine_configuration_channel], no_graphics=True)
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec, term = env.get_steps(behavior_name)

    agent = PPOAgent()
    scores, episode, score = [], 0, 0

    state = dec.obs[OBS][0, -3:, :, :]
    #state = np.transpose(state, (1, 2, 0))

    for step in tqdm(range(run_step + test_step)):
        if step == run_step:  # 학습 완료 후 테스트 모드로 전환
            if train_mode:
                agent.save_model(save_path)
            print('Switching to test mode...')
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        action = agent.get_action(state, train=train_mode)
        print(action.shape, action)
        action_tuple = ActionTuple()
        action_tuple.add_discrete(discrete=action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        # 환경으로부터 상태 업데이트
        dec, term = env.get_steps(behavior_name)
        done = len(term) > 0  # 종료 조건
        reward = term.reward[0] if done else dec.reward[0]
        next_state = term.obs[OBS][0, -3:, :, :] if done else dec.obs[OBS][0, -3:, :, :]
        #next_state = np.transpose(next_state, (1, 2, 0))

        # 저장 및 학습 처리
        log_prob1 = torch.log(torch.FloatTensor(agent.ppo(state)[0][0, action[0:0]]))
        log_prob2 = torch.log(torch.FloatTensor(agent.ppo(state)[0][1, action[0:1]]))
        log_prob = log_prob1 + log_prob2

        agent.store_transition(state, action, reward, next_state, done, log_prob.item())

        if train_mode and step >= train_start_step and step % update_step == 0:
            agent.train_model()

        # 다음 상태 업데이트 및 점수 누적
        state = next_state
        score += reward

        if done:
            scores.append(score)
            score = 0
            episode += 1
            if episode % print_interval == 0:
                avg_score = np.mean(scores[-print_interval:])
                print(f"Episode: {episode}, Step: {step}, Avg Score: {avg_score:.2f}")
                agent.writer.add_scalar("Average Score", avg_score, global_step=step)

            env.reset()
            dec, term = env.get_steps(behavior_name)
            state = dec.obs[OBS][0, -3:, :, :]
            state = np.transpose(state, (1, 2, 0))

        # 모델 저장
        if train_mode and step % save_interval == 0:
            agent.save_model(save_path)

    env.close()
    print("Training completed.")

 