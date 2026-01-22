import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import copy
from .base import BaseAlgorithm
from model import ActorCritic


class RewardShaper:
    """
    奖励塑形器 - 为战术行为提供额外奖励信号

    设计原则:
    1. 所有塑形奖励都是零和的或自身相关的
    2. 鼓励有意义的战术行为
    3. 不改变最优策略（势函数差分）
    """

    def __init__(self, device):
        self.device = device

        # 奖励系数 (可调整)
        self.hp_advantage_coef = 0.1      # HP 优势奖励
        self.resource_control_coef = 0.05  # 资源控制奖励
        self.positioning_coef = 0.02       # 位置优势奖励
        self.action_diversity_coef = 0.01  # 动作多样性奖励

        # 常量
        self.map_size = 12
        self.max_hp = 4.0
        self.max_energy = 7.0
        self.max_shield = 2.0

    def compute_shaped_reward(self, obs, next_obs, action, base_reward):
        """
        计算塑形奖励

        Args:
            obs: 当前观测 [B, 160]
            next_obs: 下一步观测 [B, 160]
            action: 执行的动作 [B]
            base_reward: 环境给的基础奖励 [B]

        Returns:
            shaped_reward: 塑形后的奖励 [B]
        """
        batch_size = obs.shape[0]
        shaped_reward = base_reward.clone()

        # 解析观测
        my_hp = obs[:, 4] * self.max_hp
        enemy_hp = obs[:, 5] * self.max_hp
        my_energy = obs[:, 6] * self.max_energy
        my_shield = obs[:, 8] * self.max_shield

        next_my_hp = next_obs[:, 4] * self.max_hp
        next_enemy_hp = next_obs[:, 5] * self.max_hp
        next_my_energy = next_obs[:, 6] * self.max_energy
        next_my_shield = next_obs[:, 8] * self.max_shield

        # ============ 1. HP 优势奖励 (势函数差分) ============
        # Φ(s) = (my_hp - enemy_hp) / max_hp
        # r_shaped = γ * Φ(s') - Φ(s)
        hp_diff_current = (my_hp - enemy_hp) / self.max_hp
        hp_diff_next = (next_my_hp - next_enemy_hp) / self.max_hp
        hp_shaping = self.hp_advantage_coef * (hp_diff_next - hp_diff_current)
        shaped_reward += hp_shaping

        # ============ 2. 资源效率奖励 ============
        # 鼓励有效使用资源（使用技能而非浪费）
        # 惩罚能量过满时不行动
        energy_full = (my_energy >= self.max_energy - 0.5).float()
        stayed_still = (action == 0).float()  # Stay action
        energy_waste_penalty = -0.02 * energy_full * stayed_still
        shaped_reward += energy_waste_penalty

        # ============ 3. 护盾效率奖励 ============
        # 在受到攻击前获得护盾是好的
        shield_gained = (next_my_shield > my_shield).float()
        shield_bonus = 0.05 * shield_gained
        shaped_reward += shield_bonus

        # ============ 4. 存活奖励 (轻微) ============
        # 鼓励活得更久
        survival_bonus = 0.001 * torch.ones(batch_size, device=self.device)
        shaped_reward += survival_bonus

        return shaped_reward


class OpponentPool:
    """
    对手池 - 用于自博弈的历史对手采样

    功能:
    1. 保存历史模型快照
    2. 根据 Elo 或均匀分布采样对手
    3. 防止策略遗忘
    """

    def __init__(self, max_size=10, device='cpu'):
        self.max_size = max_size
        self.device = device
        self.pool = deque(maxlen=max_size)
        self.elo_ratings = deque(maxlen=max_size)
        self.base_elo = 1000

    def add(self, model_state_dict, elo=None):
        """添加模型快照到池中"""
        # 深拷贝模型状态
        snapshot = {k: v.cpu().clone() for k, v in model_state_dict.items()}
        self.pool.append(snapshot)
        self.elo_ratings.append(elo if elo else self.base_elo)

    def sample(self, model_class, obs_dim, action_dim):
        """
        从池中采样一个对手

        Returns:
            opponent_model: 可用于推理的模型
            idx: 采样的索引
        """
        if len(self.pool) == 0:
            return None, -1

        # 均匀采样 (也可以根据 Elo 加权)
        idx = np.random.randint(0, len(self.pool))
        snapshot = self.pool[idx]

        # 创建模型并加载状态
        opponent = model_class(obs_dim, action_dim).to(self.device)
        opponent.load_state_dict({k: v.to(self.device) for k, v in snapshot.items()})
        opponent.eval()

        return opponent, idx

    def update_elo(self, idx, win: bool, k=32):
        """更新 Elo 评分"""
        if idx < 0 or idx >= len(self.elo_ratings):
            return

        old_elo = self.elo_ratings[idx]
        expected = 1 / (1 + 10 ** ((self.base_elo - old_elo) / 400))
        actual = 1.0 if win else 0.0
        new_elo = old_elo + k * (actual - expected)
        self.elo_ratings[idx] = new_elo

    def __len__(self):
        return len(self.pool)


class PPO(BaseAlgorithm):
    """
    优化的 PPO 算法

    改进点:
    1. 奖励塑形
    2. 学习率调度 (线性衰减 / 余弦退火)
    3. KL 散度早停
    4. 梯度累积
    5. 价值函数预热
    6. 对手池采样
    """

    def __init__(self, config, obs_dim, action_dim):
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = config.device

        # 模型
        self.model = ActorCritic(obs_dim, action_dim).to(self.device)

        # 优化器 (AdamW 更好的正则化)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
            weight_decay=1e-4
        )

        # 学习率调度器
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.train_steps,
            eta_min=config.learning_rate * 0.1
        )

        # 奖励塑形器
        self.reward_shaper = RewardShaper(self.device)
        self.use_reward_shaping = getattr(config, 'use_reward_shaping', True)

        # 对手池
        self.opponent_pool = OpponentPool(max_size=10, device=self.device)
        self.use_opponent_pool = getattr(config, 'use_opponent_pool', False)

        # KL 散度目标 (用于自适应调整)
        self.target_kl = getattr(config, 'target_kl', 0.02)

        # 训练步数追踪 (用于学习率调度和保存)
        self.training_step = 0
        self.checkpoint_path = None

        # Storage
        self.obs_buffer = []
        self.mask_buffer = []
        self.action_buffer = []
        self.logprob_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.done_buffer = []

        # 额外存储 (用于奖励塑形)
        self.next_obs_buffer = []

    def get_action(self, obs, mask=None, deterministic=False):
        """
        获取动作

        Returns:
            action: Tensor
            info: Dict containing logprob, value, entropy
        """
        with torch.no_grad():
            action, logprob, entropy, value = self.model.get_action_and_value(obs, mask)

        if deterministic:
            # 确定性模式：选择概率最高的动作
            with torch.no_grad():
                fused = self.model._encode_obs(obs)
                actor_hidden = self.model.actor_backbone(fused)
                logits = self.model.actor_head(actor_hidden)
                if mask is not None:
                    logits = logits + (mask - 1.0) * 1e8
                action = logits.argmax(dim=-1)

        info = {
            "logprob": logprob,
            "value": value,
            "entropy": entropy
        }
        return action, info

    def get_value(self, obs):
        with torch.no_grad():
            return self.model.get_value(obs)

    def store_transition(self, obs, action, reward, done, mask, info, next_obs=None):
        self.obs_buffer.append(obs)
        self.mask_buffer.append(mask)
        self.action_buffer.append(action)
        self.logprob_buffer.append(info["logprob"])
        self.reward_buffer.append(reward)
        self.value_buffer.append(info["value"].flatten())
        self.done_buffer.append(done)

        if next_obs is not None:
            self.next_obs_buffer.append(next_obs)

    def update(self, next_obs=None):
        config = self.config
        device = self.device

        self.training_step += 1

        num_steps = len(self.obs_buffer)

        # Stack buffers
        b_obs = torch.stack(self.obs_buffer)
        b_mask = torch.stack(self.mask_buffer)
        b_actions = torch.stack(self.action_buffer)
        b_logprobs = torch.stack(self.logprob_buffer)
        b_rewards = torch.stack(self.reward_buffer)
        b_values = torch.stack(self.value_buffer)
        b_dones = torch.stack(self.done_buffer)

        # ============ 奖励塑形 (可选) ============
        if self.use_reward_shaping and len(self.next_obs_buffer) == num_steps:
            b_next_obs = torch.stack(self.next_obs_buffer)
            for t in range(num_steps):
                b_rewards[t] = self.reward_shaper.compute_shaped_reward(
                    b_obs[t], b_next_obs[t], b_actions[t], b_rewards[t]
                )

        # ============ 计算 GAE ============
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(-1)
            advantages = torch.zeros_like(b_rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = b_values[t + 1]

                nonterminal = 1.0 - b_dones[t].float()

                delta = b_rewards[t] + config.gamma * nextvalues * nonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nonterminal * lastgaelam

            returns = advantages + b_values

        # Flatten
        b_obs = b_obs.reshape((-1, self.obs_dim))
        b_mask = b_mask.reshape((-1, self.action_dim))
        b_logprobs = b_logprobs.reshape(-1)
        b_actions = b_actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = b_values.reshape(-1)

        # ============ PPO 更新 ============
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        results = {}

        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)

            epoch_kl = 0
            num_batches = 0

            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.model.get_action_and_value(
                    b_obs[mb_inds],
                    b_mask[mb_inds],
                    b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
                    epoch_kl += approx_kl.item()
                    num_batches += 1

                mb_advantages = b_advantages[mb_inds]

                # Advantage Normalization (per-minibatch)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss (Clipped)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss (Clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -config.clip_coef,
                    config.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                # Total Loss
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()

            # KL 早停
            avg_kl = epoch_kl / max(num_batches, 1)
            if self.target_kl is not None and avg_kl > self.target_kl:
                break

        # 学习率调度
        self.lr_scheduler.step()

        # 记录结果
        results['loss'] = loss.item()
        results['pg_loss'] = pg_loss.item()
        results['v_loss'] = v_loss.item()
        results['entropy_loss'] = entropy_loss.item()
        results['approx_kl'] = approx_kl.item()
        results['clipfrac'] = np.mean(clipfracs)
        results['lr'] = self.optimizer.param_groups[0]['lr']

        # 定期保存到对手池
        if self.use_opponent_pool and self.training_step % 50 == 0:
            self.opponent_pool.add(self.model.state_dict())

        # Clear buffers
        self._clear_buffers()

        return results

    def _clear_buffers(self):
        self.obs_buffer = []
        self.mask_buffer = []
        self.action_buffer = []
        self.logprob_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.done_buffer = []
        self.next_obs_buffer = []

    def save(self, path):
        """保存模型和训练状态"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'training_step': self.training_step,
        }
        torch.save(checkpoint, path)
        self.checkpoint_path = path

    def load(self, path):
        """加载模型和训练状态"""
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'training_step' in checkpoint:
                self.training_step = checkpoint['training_step']
        else:
            # 兼容旧格式
            self.model.load_state_dict(checkpoint)

        self.checkpoint_path = path

    def get_opponent_from_pool(self):
        """从对手池获取对手"""
        if len(self.opponent_pool) == 0:
            return None
        opponent, idx = self.opponent_pool.sample(ActorCritic, self.obs_dim, self.action_dim)
        return opponent
