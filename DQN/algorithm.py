import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# DQN网络结构 由三层128的全连接层构成
class DQNNet(nn.Module):
    # 定义时需要输入调整的特征个数和输出的动作个数
    def __init__(self, states, actions):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(states, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)  # 初始化initialization
        self.fc3 = nn.Linear(128, 128)
        self.fc3.weight.data.normal_(0, 0.1)  # 初始化initialization
        self.out = nn.Linear(128, actions)
        self.out.weight.data.normal_(0, 0.1)  # 初始化initialization
    # 前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# DQN类
class DQN(object):
    # 所使用的网络模型 动作维度 gamma值和学习率
    def __init__(self, model, action_dim=None, gamma=None, lr=None):
        # 使用GPU进行训练
        self.device = torch.device("cuda:0")
        # 构建两个相同的网络模型
        self.model = model.to(self.device)
        self.target_model = model.to(self.device)
        # 检查参数类型是否正确
        assert isinstance(action_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        # 传递参数
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        # 选择初始时的策略随机度和策略的随机性下降值
        self.e_greed = 0.9
        self.e_greed_decay = 0.001
        self.epoch = 0
        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)
    # target网络同步网络参数
    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    # 使用模型预测阶段
    def predict(self, obs):
        # 将状态传递到GPU预测再传回CPU
        out = self.model(torch.Tensor(obs).to(self.device))
        out = out.cpu().detach().numpy()
        return out
    # 学习函数
    def learn(self, obs, action, reward, next_obs, done):
        # 每隔200代同步target网络和训练网络的参数
        if self.epoch % 50 == 0:
            self.sync_target()
        self.epoch += 1
        # 已经被置为1的位置不能再次被选取
        invalid_set = np.squeeze(np.argwhere(next_obs == 1))
        #将经验池中取到的经验取到GPu中
        obs, action, reward, next_obs = torch.Tensor(obs).to(self.device), torch.Tensor(
            action).reshape(-1, 1).long().to(self.device), torch.Tensor(reward).to(self.device), torch.Tensor(
            next_obs).to(self.device)

        # 使用target网络预测
        next_pred_value = self.target_model(next_obs).cpu().detach().numpy()
        # 将不可行的位置置为较大负值
        next_pred_value[invalid_set] = -9999
        next_pred_value = torch.Tensor(next_pred_value).to(self.device)
        # 寻找最大Q值位置
        best_v = torch.max(next_pred_value, dim=1).values
        done = done.astype(int)
        done = torch.Tensor(done).to(self.device)
        # 根据公式计算反馈
        target = (reward + (1 - done) * self.gamma * best_v).reshape(-1, 1)
        # 使用原始模型预测动作
        pred_value = self.model(obs).gather(1, action)
        # 计算损失
        loss = self.criterion(pred_value, target)
        self.optimizer.zero_grad()
        #反向传播及优化
        loss.backward()
        self.optimizer.step()
        # 保存模型
        torch.save(self.model,'./DQN/result/dqn_model.pth')
        torch.save(self.target_model,'./DQN/result/dqn_target_model.pth')
    # 采样函数
    def sample(self, obs):
        rand = np.random.rand()
        # 如果随机值小于e则随机采样
        if rand < self.e_greed:
            actionset = np.squeeze(np.argwhere(obs == 0))
            action = random.sample(list(actionset), 1)[0]
        # 否则使用模型预测下一步的动作
        else:
            invalid_set = np.squeeze(np.argwhere(obs == 1))
            action = self.predict(obs)
            action[invalid_set] = -9999
            action = np.argmax(action)
        # e值策略衰减
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decay)

        return action
