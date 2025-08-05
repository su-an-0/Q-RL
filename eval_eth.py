import os
import time
from tqdm import tqdm  # 导入进度条库
from pre_processing import processed_data_dict
from agent import DQN
from env import Action, TradingEnvironment
import matplotlib.pyplot as plt
import numpy as np
import torch

# 本地路径设置
PREDS_DIR_PATH = "./preds"
CHECKPOINTS_DIR = "./checkpoints"

# 获取处理后的数据字典
input_data_dict = processed_data_dict()
ticker_name = 'ETH'

print(f'使用的数据集是 {ticker_name}')


def add_label(df):
    df['Action'] = Action.HOLD.value  # 默认为观望
    # 记录当前持仓状态（用于连续操作逻辑，0=空仓，1=多头，-1=空头）
    position = 0  # 初始为空仓
    
    for i in range(3, len(df)):
        # 计算过去3天的价格变化率加权和（原逻辑不变，用于判断趋势）
        three_days_pred = (
            0.4 * (df['Close'][i-2] - df['Close'][i-3]) / df['Close'][i-3] +
            0.32 * (df['Close'][i-1] - df['Close'][i-2]) / df['Close'][i-2] +
            0.28 * (df['Close'][i] - df['Close'][i-1]) / df['Close'][i-1]
        )
        
        # 1. 趋势判断：大幅上涨（适合平多或做空）
        if three_days_pred > 0.01:
            if position == 1:
                # 若持有多头，此时应平多（SELL）
                df.iloc[i, df.columns.get_loc('Action')] = Action.SELL.value
                position = 0  # 平仓后为空仓
            elif position == 0:
                # 若空仓，此时应做空（SHORT）
                df.iloc[i, df.columns.get_loc('Action')] = Action.SHORT.value
                position = -1  # 持有空头
            else:
                # 若已持有空头，继续观望（HOLD）
                df.iloc[i, df.columns.get_loc('Action')] = Action.HOLD.value
        
        # 2. 趋势判断：大幅下跌（适合平空或做多）
        elif three_days_pred < -0.01:
            if position == -1:
                # 若持有空头，此时应平空（COVER）
                df.iloc[i, df.columns.get_loc('Action')] = Action.COVER.value
                position = 0  # 平仓后为空仓
            elif position == 0:
                # 若空仓，此时应做多（Long）
                df.iloc[i, df.columns.get_loc('Action')] = Action.Long.value
                position = 1  # 持有多头
            else:
                # 若已持有多头，继续观望（HOLD）
                df.iloc[i, df.columns.get_loc('Action')] = Action.HOLD.value
        
        # 3. 趋势平缓（无操作，保持当前持仓）
        else:
            df.iloc[i, df.columns.get_loc('Action')] = Action.HOLD.value
    
    return df

# 数据处理与划分
df_eth = input_data_dict[ticker_name]
df_eth = add_label(df_eth)

# 训练集：2017年8月17日 至 2023年8月1日
df_eth_train = df_eth[
    (df_eth['Date'] >= '2017-08-17') & 
    (df_eth['Date'] <= '2023-08-01')
].reset_index(drop=True)

# 测试集：2023年8月2日 至今
df_eth_test = df_eth[
    df_eth['Date'] >= '2023-08-02'
].reset_index(drop=True)

env = TradingEnvironment(df_eth_train)

# 模型参数
state_size = env.state_size
action_size = env.action_space
hidden_size = 128
buffer_size = 10000
batch_size = 256
gamma = 0.95
lr = 0.001
initial_exploration = 1.0
exploration_decay = 0.9
min_exploration = 0.001
max_iter_episode = 100
target_update_frequency = 2
num_layers = 4
num_heads = state_size
while_used = True
loss_fn_used = 'KL Div Loss'
optimizer_used = 'RMSProp'

dqn_agent = DQN(state_size, action_size, lr, gamma, num_layers, num_heads, optimizer_type='AdamW')
print(f'运行设备: {dqn_agent.device.type.upper()}')

# 加载模型检查点
def load_model():
    if os.path.exists(CHECKPOINTS_DIR):
        checkpoints = os.listdir(CHECKPOINTS_DIR)
        if len(checkpoints) == 0:
            print('***** 无可用检查点 *****')
            return 0
        print('***** 加载模型 *****')
        sorted_files = sorted(checkpoints, key=lambda x: os.path.getmtime(f'{CHECKPOINTS_DIR}/{x}'), reverse=True)
        latest_checkpoint = sorted_files[0]
        print(f'最新检查点: {latest_checkpoint}')
        last_saved_episode = int(latest_checkpoint.split('_')[1].split('.')[0])
        print(f'最后保存的episode: {last_saved_episode}')
        dqn_agent.q_network.load_state_dict(torch.load(f'{CHECKPOINTS_DIR}/{latest_checkpoint}'))
        dqn_agent.target_network.load_state_dict(torch.load(f'{CHECKPOINTS_DIR}/{latest_checkpoint}'))
        print('***** 模型加载完成 *****')
        return last_saved_episode + 1

# 初始化训练参数
load_checkpoint = False
start_episode = load_model() if load_checkpoint else 0
num_episodes = start_episode + 100 if start_episode else 10000

# 训练指标记录
loss = []
avg_reward = []
profit_list = []
accuracy = []
total_reward = 0

# ********** 核心修改：添加总进度条 **********
# 创建总训练轮次的进度条，显示episode进度
with tqdm(total=num_episodes - start_episode, desc="总训练进度", unit="episode") as pbar_total:
    # 开始训练循环
    for episode in range(start_episode, num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        profit = 0
        epsilon = max(min_exploration, initial_exploration * (exploration_decay ** episode))
        count = 0

        # ********** 核心修改：添加单轮episode进度条 **********
        # 估算单轮最大步数（使用训练集长度），提升进度条准确性
        max_steps = len(df_eth_train)
        with tqdm(total=max_steps, desc=f"Episode {episode+1}/{num_episodes}", unit="step", leave=False) as pbar_step:
            while not done:
                action = dqn_agent.decide_action(state, epsilon)
                next_state, reward, done, labels = env.step(action)
                dqn_agent.remember(state, action, reward, next_state, done, labels)
                dqn_agent.train(state, action, reward, next_state, done, labels, count)
                episode_reward += reward
                state = next_state
                count += 1
                
                # 更新单步进度条
                pbar_step.update(1)
                # 实时显示单轮奖励
                pbar_step.set_postfix({"当前奖励": f"{episode_reward:.2f}", "当前损失": f"{np.mean(dqn_agent.loss) if dqn_agent.loss else 0:.6f}"})
                
                # 防止无限循环（极端情况保护）
                if count >= max_steps:
                    break

        # 更新目标网络
        if episode % target_update_frequency == 0:
            dqn_agent.update_target_network()

        # 保存模型
        if episode % 10 == 0:
            print('***** 保存模型 *****')
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            torch.save(dqn_agent.q_network.state_dict(), f'{CHECKPOINTS_DIR}/checkpoint_{episode}.pth')
            print('***** 模型保存完成 *****')
        
        # 记录训练指标
        total_reward += episode_reward
        avg_reward.append(total_reward)
        loss.append(np.mean(dqn_agent.loss) if dqn_agent.loss else 0)
        mean_acc = np.mean(dqn_agent.accuracy) if dqn_agent.accuracy else 0
        accuracy.append(mean_acc)
        
        # 更新总进度条信息
        pbar_total.update(1)
        pbar_total.set_postfix({
            "累计奖励": f"{total_reward:.2f}",
            "平均准确率": f"{mean_acc*100:.2f}%",
            "当前损失": f"{loss[-1]:.6f}"
        })

# 训练结果统计
average_reward = total_reward / num_episodes
print(f'\n平均奖励: {average_reward:.2f}')
print(f'最终准确率: {accuracy[-1]*100:.2f}%')
print(f'最终损失: {loss[-1]:.6f}')

# 保存训练结果
print('***** 保存结果文件 *****')
curr_time = time.strftime("%Y_%m_%d_%H_%M_%S")

def save_plots():
    plt.subplot(3, 1, 1)
    plt.plot(avg_reward)
    plt.xlabel('Episode')
    plt.ylabel('总奖励')
    plt.title('训练总奖励曲线')
    
    plt.subplot(3, 1, 2)
    plt.plot(loss)
    plt.xlabel('Episode')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')

    plt.subplot(3, 1, 3)
    plt.plot([x*100 for x in accuracy])
    plt.xlabel('Episode')
    plt.ylabel('准确率(%)')
    plt.title('训练准确率曲线')

    plt.tight_layout()
    plt.savefig(f'{PREDS_DIR_PATH}/reward_loss_acc_graph_{curr_time}.png')
    plt.show()

def create_log_file():
    with open(f'{PREDS_DIR_PATH}/training_log_{curr_time}.txt', 'w') as f:
        f.write(f'加密货币: {ticker_name}\n')
        f.write(f'训练集时间范围: 2017-08-17 至 2023-08-01\n')
        f.write(f'测试集时间范围: 2023-08-02 至今\n')
        f.write(f'总训练轮次: {num_episodes}\n')
        f.write(f'平均奖励: {average_reward:.2f}\n')
        f.write(f'最终准确率: {accuracy[-1]*100:.2f}%\n')
        f.write(f'最终损失: {loss[-1]:.6f}\n')

# 创建结果目录
os.makedirs(PREDS_DIR_PATH, exist_ok=True)
if not os.path.exists(f'{PREDS_DIR_PATH}/{curr_time}'):
    PREDS_DIR_PATH += '/' + curr_time
    os.makedirs(PREDS_DIR_PATH, exist_ok=True)

save_plots()
create_log_file()
print('***** 训练完成 *****')