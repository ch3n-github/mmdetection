import numpy as np
import os
from gym import spaces
# 环境类
class SignEnv():
    # 初始化
    def __init__(self, n):
        # 设定使用的动作空间
        self.n = n
        # 状态空间
        self.state = np.zeros(n)
        self.action_space = spaces.Discrete(n)
        self.reward = 0
        # 结束标记符
        self.done = False
        # 记录使用的特征组及其反馈
        self.feature_reward_dict = {}
    # 环境重设
    def reset(self):
        # 将状态归零
        self.state = np.zeros(self.n)
        self.done = False
        # 反馈置零
        self.reward = 0
        return self.state
    # 空间行动
    def step(self, action):
        # 将对应行为的选取状态置1
        self.state[action] = 1
        # 设置终止条件即选取两个特征时结束选取
        if np.sum(self.state) == 2:
            # self.reward = self.get_reward()
            self.done = True
        # 得到反馈
        self.reward = self.get_reward()
        # print(self.state,self.done)
        return self.state, self.reward, self.done, {}
    # 反馈函数
    def get_reward(self):
        # 调用改动参数函数获得使用的特征
        features = self.change_detection_config()
        # 查询是否已经获得数据
        if features in self.feature_reward_dict:
            reward = self.feature_reward_dict[features]
        else:
            # 使用选择的特征数据训练模型
            os.system('python tools/train.py --config DQN/dqn_config.py')
            os.system('python tools/test.py --config DQN/dqn_config.py --checkpoint DQN/dqn_exp/latest.pth --eval bbox')
            # 获取反馈值
            reward = np.load('./DQN/dqn_exp/epoch_reward.npy')
            # reward = sum(i*self.state[i] for i in range(self.n))
            # 写入记录字典
            self.feature_reward_dict[features] = reward
        print(features,reward)
        return reward
    # 修改使用的特征参数函数
    def change_detection_config(self):
        features = ''
        # 对状态进行编码得到字符串类型数据
        for i in range(len(self.state)):
            if self.state[i]:
                features += '\'f_'+str(i)+'\','
        # 修改配置文件内容
        file = './DQN/dqn_config.py'
        file_data = ""
        # 根据文件内的关键词修改使用的特征
        with open(file, "r", encoding = "utf-8") as f:
            for line in f:
                if "feature_type" in line:
                    line ="\t\t\tfeature_type=["+features+"],\n"
                file_data+=line
        # 保存文件
        with open(file, "w", encoding = "utf-8") as f:
            f.write(file_data)
        return features
