from enum import Enum
import pandas as pd

class Action(Enum):
    Long = 0       # 做多（买入）
    SELL = 1       # 平多（卖出多头）
    HOLD = 2       # 观望
    SHORT = 3      # 做空（借入卖出）
    COVER = 4      # 平空（买回偿还）

action_space = [action.value for action in Action]

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.close_prices = self.data['Close'].to_numpy()
        
        # 核心属性：跟踪多头、空头头寸及账户状态
        self.current_step = 0
        self.balance = initial_balance  # 现金余额
        self.stock_owned = 0  # 多头持仓数量
        self.short_position = 0  # 空头持仓数量（借入的股票数量）
        self.short_entry_price = 0.0  # 做空时的价格（用于计算平仓收益）
        self.net_worth = initial_balance  # 总资产（现金+多头市值-空头负债）
        
        # 动作空间和状态空间
        self.action_space = len(action_space)
        # 状态扩展为：[现金余额, 多头市值, 当前价格, 空头数量, 空头持仓价值]
        self.state_size = 5  # 新增空头相关状态，提升决策准确性

    def reset(self):
        """重置环境到初始状态"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.short_position = 0
        self.short_entry_price = 0.0
        self.net_worth = self.initial_balance
        self.done = False
        return self._get_state()

    def _get_state(self):
        """获取当前状态（包含多头和空头信息）"""
        current_price = self.data.loc[self.current_step, 'Close']
        # 多头市值 = 持有数量 * 当前价格
        long_value = self.stock_owned * current_price
        # 空头负债 = 做空数量 * 做空时的价格（平仓时需偿还的基准）
        short_value = self.short_position * self.short_entry_price if self.short_position > 0 else 0
        # 状态包含：现金余额、多头市值、当前价格、空头数量、空头负债
        state = [
            self.balance,
            long_value,
            current_price,
            self.short_position,
            short_value
        ]
        return state

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        current_price = self.close_prices[self.current_step]
        fee_rate = 0.0005  # 手续费率：0.05%
        reward = 0
        earnings = 0

        # 记录上一步净值（用于计算增长奖励）
        previous_net_worth = self.net_worth
        # 计算当前净值（总资产）
        self.net_worth = self.balance + (self.stock_owned * current_price) - (self.short_position * self.short_entry_price if self.short_position > 0 else 0)
        # 净值变化（核心奖励来源之一：鼓励持续增长）
        net_worth_change = (self.net_worth - previous_net_worth) / self.initial_balance  # 放大100倍

        # 1. 做多（Long）：成功开仓给予小正向奖励，鼓励尝试
        if action == Action.Long.value:
            if self.stock_owned == 0 and self.short_position == 0 and self.balance > current_price * (1 + fee_rate):
                fee = current_price * fee_rate
                self.stock_owned += 1
                self.balance -= (current_price + fee)
                reward = 0.1 + net_worth_change  # 基础奖励+净值变化（鼓励开仓后净值增长）
            else:
                reward = -0.1  # 无效开仓（如已有持仓）给予负奖励

        # 2. 平多（SELL）：盈利/亏损奖励差异放大
        elif action == Action.SELL.value:
            if self.stock_owned > 0:
                revenue = current_price - (current_price * fee_rate)
                self.stock_owned -= 1
                self.balance += revenue
                # 用实际买入成本计算收益（需记录每次买入价，这里优化为记录平均成本）
                # 新增：记录多头持仓成本（替代之前的近似值）
                avg_buy_price = (self.initial_balance - self.balance - (self.short_position * self.short_entry_price if self.short_position > 0 else 0)) / (self.stock_owned + 1) if (self.stock_owned + 1) > 0 else current_price
                earnings = revenue - avg_buy_price
                # 盈利奖励放大10倍，亏损惩罚放大5倍
                reward = (earnings / self.initial_balance) + net_worth_change  # 放大1000倍
            else:
                reward = -0.1  # 无效平仓

        # 3. 做空（SHORT）：成功开仓给予小正向奖励，与Long对称
        elif action == Action.SHORT.value:
            if self.stock_owned == 0 and self.short_position == 0 and self.balance > current_price * fee_rate:
                self.short_position += 1
                self.short_entry_price = current_price
                self.balance -= current_price * fee_rate
                reward = 0.1 + net_worth_change  # 基础奖励+净值变化
            else:
                reward = -0.1  # 无效开仓

        # 4. 平空（COVER）：盈利/亏损奖励差异放大
        elif action == Action.COVER.value:
            if self.short_position > 0:
                cover_cost = current_price + (current_price * fee_rate)
                self.short_position -= 1
                self.balance -= cover_cost
                earnings = self.short_entry_price - cover_cost
                # 与平多对称，放大奖励尺度
                reward = (earnings / self.initial_balance) + net_worth_change  # 放大1000倍
            else:
                reward = -0.1  # 无效平仓

        # 5. 观望（HOLD）：奖励为0，但低于有效开仓的0.1，避免模型过度观望
        elif action == Action.HOLD.value:
            reward = 0 + net_worth_change  # 仅跟随净值变化

        # 处理未定义动作（严格惩罚）
        else:
            reward = -0.5  # 明显低于其他无效动作，禁止模型尝试

        # 限制奖励极端值（避免梯度爆炸）
        reward = max(min(reward, 5.0), -5.0)  # 奖励范围[-5, 5]

        state = self._get_state()
        labels = self.data.loc[self.current_step, 'Action'] if 'Action' in self.data.columns else -1

        return state, reward, done, labels