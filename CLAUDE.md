# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作提供指导。

## 构建与运行命令

```bash
# 构建 Rust 环境（需要 maturin）
pip install maturin
maturin develop --release

# 安装 Python 依赖
pip install torch numpy

# 运行训练
python train.py
```

**训练输出：**
- 每 10 次更新输出进度日志（FPS、回报、胜率、伤害统计）
- 每 25 次更新在 `replays/` 目录生成回放文件
- 模型检查点：`ppo_final.pth` 或 `ppo_interrupted.pth`

## 架构概述

这是一个 Rust+Python 混合强化学习框架，用于使用带动作掩码的 PPO 进行对称自我博弈训练。

### Rust 环境层 (`src/lib.rs`)

**GameEnv Trait** - 游戏实现的通用接口：
```rust
pub trait GameEnv: Send + Sync + Clone {
    fn reset(&mut self) -> (obs_p1, obs_p2, mask_p1, mask_p2);
    fn step(&mut self, action_p1, action_p2) -> (obs, rewards, done, masks, info);
}
```

**SimpleDuel** - 实现 GameEnv 的 12x12 战术对战游戏：
- 13 个离散动作：停留、移动（4方向）、近战、射击、闪避、护盾、冲刺、AOE、治疗、换弹
- 160 维观测：16 个状态值 + 144 个网格单元
- 使用 Bresenham 算法计算远程攻击的视线
- 对称地形和道具生成确保公平游戏

**VectorizedEnv** - 使用 Rayon 并行化的 PyO3 封装：
- 管理 N 个并行游戏实例
- 返回堆叠的观测 `[2*N, 160]`（P1 占前 N 行，P2 占后 N 行）
- 回合结束时自动重置

### 游戏参数

| 项目 | 值 |
|------|------|
| 地图尺寸 | 12x12 |
| 动作数量 | 13 |
| 观测维度 | 160 |
| 生命值 | 4 |
| 能量值 | 7 |
| 护盾 | 最大2层 |
| 弹药 | 最大6发 |
| 最大回合 | 60 |
| 突然死亡 | 第40回合后停止能量恢复 |

### 动作空间

| ID | 动作 | 能量消耗 | 说明 |
|----|------|---------|------|
| 0 | Stay | 0 | 原地不动 |
| 1-4 | Move | 1 | 上下左右移动（水域+1） |
| 5 | Attack | 2 | 近战攻击（范围1） |
| 6 | Shoot | 3 | 远程射击（范围4，消耗1弹药） |
| 7 | Dodge | 2 | 下回合免疫1次攻击 |
| 8 | Shield | 3 | 获得1层护盾 |
| 9 | Dash | 3 | 向远离敌人方向移动2格 |
| 10 | AOE | 4 | 对周围1格内敌人造成伤害 |
| 11 | Heal | 4 | 恢复1HP（5步冷却） |
| 12 | Reload | 0 | 恢复3发弹药 |

### 观测空间

```
[0-7]:   mx, my, ex, ey, mhp, ehp, meng, eeng (归一化位置和属性)
[8-11]:  my_shield, enemy_shield, my_ammo, enemy_ammo
[12-15]: my_dodge, enemy_dodge, heal_cd, step_progress
[16-159]: 12x12 地形网格 (编码地形类型和道具)
```

### 地形系统

| 类型 | 编码值 | 效果 |
|------|--------|------|
| 空地 | 0.0 | 正常通行 |
| 墙体 | 0.25 | 不可通过、阻挡视线 |
| 水域 | 0.5 | 移动消耗+1能量、阻挡射击 |
| 高地 | 0.75 | 射击范围+1、被远程攻击50%闪避 |

### 道具系统

- 地图对称放置2-3个道具点
- 道具被拾取后10步刷新
- 类型：血包(+1HP)、能量球(+3能量)、弹药箱(+2弹药)、护盾(+1层)

### 核心设计：对称变换

两个玩家共享同一个神经网络。这通过对 P2 的观测进行变换来实现：
- **坐标翻转**：`(MAP_SIZE-1-x, MAP_SIZE-1-y)`（180° 旋转）
- **动作翻转**：P2 的上↔下、左↔右在物理空间中互换
- **非方向动作**：Dodge, Shield, Dash, AOE, Heal, Reload 不需要翻转
- **效果**：两个玩家在观测中都"先看到自己"

这是核心设计原则——任何新的游戏实现都必须保持这种对称性。

### Python 算法层

- `algorithms/ppo.py` - 带 GAE 的 PPO，通过 `logits += (mask - 1.0) * 1e8` 实现动作掩码
- `model.py` - ActorCritic：MLP(160→256→256)，具有独立的 actor/critic 输出头
- `train.py` - 自我博弈循环：32 个玩家（16 个环境 × 2），每次 rollout 2048 步
- `agents/rule_based.py` - 基于 BFS 的启发式对手，支持新动作

### 数据流

```
train.py
  ↓ VectorizedEnv.reset()
  ↓ → 并行 Rust 环境重置 → [32, 160] 观测
  ↓ PPO.get_action(obs, mask)
  ↓ → 拆分动作：P1=前 16 个，P2=后 16 个
  ↓ VectorizedEnv.step(actions_p1, actions_p2)
  ↓ → 并行 Rust 环境执行 → 新观测、奖励、结束标志
  ↓ 存储转移数据
  ↓ 每 2048 步后：PPO.update() 使用 65K 样本更新
  ↓ 重复直到 1000 万时间步
```

## 关键实现细节

- **动作掩码在 Rust 中计算**以提高性能，而非在 Python 中
- **零和奖励**：命中 +1/-1，获胜 +5/-5
- **伤害系统**：闪避完全免疫→高地50%闪避远程→护盾吸收→实际伤害
- **能量恢复**在第 40 步停止（进入突然死亡阶段）
- **游戏结束**条件：生命值 ≤ 0 或达到 60 步
- 观测和掩码全程使用 `Vec<f32>` / numpy 数组

## 训练参数

| 参数 | 值 |
|------|------|
| total_timesteps | 10,000,000 |
| num_envs | 16 |
| num_steps | 2048 |
| learning_rate | 1e-4 |
| entropy_coef | 0.02 |
| batch_size | 65,536 |
| minibatch_size | 4096 |
