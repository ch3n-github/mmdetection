import collections
import numpy as np
import random

# 经验池
class replyMemory(object):
    def __init__(self, max_size):
        # 设定最大的经验池尺寸
        self.max_size = max_size
        # 使用队列完成经验池缓存区
        self.buffer = collections.deque(maxlen=max_size)
    # 向经验池中添加经验
    def append(self, exp):
        self.buffer.append(exp)
    # 经验池随机取样
    def sample(self, batch):
        # 随机取样batch大小
        minibatch = random.sample(self.buffer, batch)
        # 根据储存经验池划分内容
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        # 从经验池中分发经验
        for experience in minibatch:
            obs, action, reward, next_obs, done = experience
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
        # 返回各组内容
        return np.array(obs_batch).astype('float32'), np.array(action_batch).astype('float32'), \
               np.array(reward_batch).astype('float32'), np.array(next_obs_batch).astype('float32'), np.array(done_batch)

    def __len__(self):
        return len(self.buffer)




