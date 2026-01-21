import torch

class Config:
    def __init__(self):
        # 训练参数
        self.total_timesteps = 5_000_000
        self.num_envs = 16 # 并行环境数 (注意：实际 Player 数量是 2 * num_envs)
        self.num_steps = 1024 # 每个环境采样步数
        self.batch_size = 2 * self.num_envs * self.num_steps # PPO Batch Size (P1 + P2)
        self.minibatch_size = 2048 # PPO Update Minibatch
        self.num_minibatches = self.batch_size // self.minibatch_size
        self.update_epochs = 4
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_coef = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        self.train_steps = self.total_timesteps // self.batch_size
        
        # 评估与可视化
        self.eval_interval = 25 # 每 25 次 update 评估一次
        self.eval_opponent = "self" # "rule" or "self"
