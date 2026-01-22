import torch

class Config:
    def __init__(self):
        # ============ 基础训练参数 ============
        self.total_timesteps = 10_000_000  # 总训练步数
        self.num_envs = 16                 # 并行环境数 (实际 Player 数量是 2 * num_envs)
        self.num_steps = 2048              # 每个环境采样步数
        self.batch_size = 2 * self.num_envs * self.num_steps  # PPO Batch Size
        self.minibatch_size = 4096         # PPO Update Minibatch
        self.num_minibatches = self.batch_size // self.minibatch_size
        self.update_epochs = 4             # 每次更新的 epoch 数

        # ============ PPO 超参数 ============
        self.learning_rate = 2.5e-4        # 初始学习率 (会被调度器调整)
        self.gamma = 0.99                  # 折扣因子
        self.gae_lambda = 0.95             # GAE lambda
        self.clip_coef = 0.2               # PPO clip 系数
        self.ent_coef = 0.01               # 熵正则化系数
        self.vf_coef = 0.5                 # 价值函数损失系数
        self.max_grad_norm = 0.5           # 梯度裁剪

        # ============ 高级训练选项 ============
        self.target_kl = 0.02              # KL 散度早停阈值 (None 禁用)
        self.use_reward_shaping = True     # 启用奖励塑形
        self.use_opponent_pool = False     # 启用对手池 (自博弈改进)

        # ============ 设备 ============
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # 计算训练步数
        self.train_steps = self.total_timesteps // self.batch_size

        # ============ 评估与可视化 ============
        self.eval_interval = 25            # 每 N 次 update 评估一次
        self.save_interval = 100           # 每 N 次 update 保存检查点
        self.eval_opponent = "self"        # 评估对手: "rule" 或 "self"


class FastConfig(Config):
    """快速训练配置 - 用于测试"""
    def __init__(self):
        super().__init__()
        self.total_timesteps = 1_000_000
        self.num_steps = 1024
        self.batch_size = 2 * self.num_envs * self.num_steps
        self.train_steps = self.total_timesteps // self.batch_size


class LongConfig(Config):
    """长时间训练配置 - 追求最强模型"""
    def __init__(self):
        super().__init__()
        self.total_timesteps = 50_000_000  # 5000万步
        self.num_envs = 32                 # 更多并行环境
        self.num_steps = 2048
        self.batch_size = 2 * self.num_envs * self.num_steps
        self.minibatch_size = 8192
        self.learning_rate = 1e-4          # 更低的学习率
        self.ent_coef = 0.005              # 更低的熵系数
        self.use_opponent_pool = True      # 启用对手池
        self.train_steps = self.total_timesteps // self.batch_size
