import replymemory
import algorithm
import tqdm
import numpy as np
import SignEnv
import copy

# 需要选取的状态的数量
num_of_state = 8
# 创建环境
env = SignEnv.SignEnv(num_of_state)
# 建立DQN算法模型
dqn = algorithm.DQN(algorithm.DQNNet(num_of_state, num_of_state), num_of_state, 0.9, 0.01)
# 建立经验回放池
ReplyMemory = replymemory.replyMemory(500)
allreward = []
# 最大迭代次数
max_epoch = 1000
# 训练的batch
batch = 128
# 开始训练
for i_episode in tqdm.tqdm(range(max_epoch)):
    # 获取初始状态
    state = env.reset()
    ep_r = 0 
    while True:
        # 保存上一个状态
        ob_state = copy.deepcopy(state)
        # 推测行动
        action = dqn.sample(state)
        # 执行行动
        next_state, reward, done, info = env.step(action)
        # 计入经验池
        memory = copy.deepcopy([ob_state, action, reward, next_state, done])
        ReplyMemory.append(memory)
        ep_r += reward
        # 如果储存了足够多的经验则进行采样和训练
        if len(ReplyMemory) >= batch:
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = ReplyMemory.sample(batch)
            dqn.learn(obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)
        # 如果推测结束跳出循环
        if done: 
            break
        # 状态变更
        state = next_state
    print('epoch:', i_episode, '       reward:', ep_r)

    allreward.append([next_state, ep_r])
# 保存反馈变化
np.save('./DQN/result/reward.npy', np.array(allreward))
