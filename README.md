# RL Self-Play

基于 Rust + Python 的高性能对称自博弈强化学习框架。

## 特性

- **高性能**: Rust 实现的向量化环境，使用 Rayon 并行处理
- **对称自博弈**: 单一神经网络同时控制双方玩家
- **动作掩码**: 环境端计算合法动作，保证策略有效性
- **战术深度**: 12x12 地图，13种动作，多样化地形和道具系统

## 快速开始

### 环境要求

- Python 3.10+
- Rust (Edition 2021)
- maturin

### 安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install maturin torch numpy

# 构建 Rust 环境
maturin develop --release
```

### 运行训练

```bash
python train.py
```

训练过程中会输出：
- 每 10 次更新显示 FPS、胜率、伤害统计
- 每 25 次更新在 `replays/` 目录生成回放文件
- `Ctrl+C` 中断保存 `ppo_interrupted.pth`
- 训练完成保存 `ppo_final.pth`

### 观看对战

使用 `watch.py` 在终端实时观看对战动画：

```bash
# PPO vs PPO 对战 (使用当前训练中的模型)
python watch.py

# PPO vs 规则智能体
python watch.py --p2 rule

# 加载已保存的模型
python watch.py --model ppo_final.pth

# 调整播放速度 (默认 0.3 秒/帧)
python watch.py --delay 0.5

# 规则智能体 vs 规则智能体
python watch.py --p1 rule --p2 rule
```

回放文件会显示对战双方的类型（PPO 模型 / Rule-Based）和模型版本信息。

## 游戏规则

### SimpleDuel - 12x12 战术对战

两名玩家在 12x12 的战术地图上对战，通过攻击、防御和资源管理击败对手。

#### 基础属性

| 属性 | 最大值 | 说明 |
|------|--------|------|
| 生命值 (HP) | 4 | 降至 0 则失败 |
| 能量 (Energy) | 7 | 执行动作消耗，每回合恢复 1 |
| 护盾 (Shield) | 2 | 吸收伤害 |
| 弹药 (Ammo) | 6 | 远程射击消耗 |

#### 动作空间

| 动作 | 能量 | 说明 |
|------|------|------|
| Stay | 0 | 原地不动 |
| Move (4方向) | 1 | 移动一格（水域 +1） |
| Attack | 2 | 近战攻击（范围 1） |
| Shoot | 3 | 远程射击（范围 4，消耗 1 弹药） |
| Dodge | 2 | 下回合免疫一次攻击 |
| Shield | 3 | 获得 1 层护盾 |
| Dash | 3 | 向远离敌人方向移动 2 格 |
| AOE | 4 | 对周围 1 格内敌人造成伤害 |
| Heal | 4 | 恢复 1 HP（5 步冷却） |
| Reload | 0 | 恢复 3 发弹药 |

#### 地形系统

| 地形 | 效果 |
|------|------|
| 空地 | 正常通行 |
| 墙体 | 不可通过，阻挡视线 |
| 水域 | 移动 +1 能量，阻挡远程攻击 |
| 高地 | 射击范围 +1，被远程攻击 50% 闪避 |

#### 道具系统

地图对称放置道具点，被拾取后 10 步刷新：

- 血包: +1 HP
- 能量球: +3 能量
- 弹药箱: +2 弹药
- 护盾: +1 层护盾

#### 游戏流程

- 最大回合数: 60
- 第 40 回合后进入突然死亡阶段（停止能量恢复）
- 胜利条件: 对手 HP 降至 0，或回合结束时 HP 更高

## 架构

```
┌─────────────────────────────────────────────────────┐
│                    Python Layer                      │
│  ┌─────────┐  ┌─────────┐  ┌──────────────────────┐ │
│  │ train.py│  │ model.py│  │ algorithms/ppo.py    │ │
│  └────┬────┘  └────┬────┘  └──────────┬───────────┘ │
│       │            │                   │             │
│       └────────────┼───────────────────┘             │
│                    ▼                                 │
│         ┌──────────────────┐                        │
│         │  VectorizedEnv   │  (PyO3 Binding)        │
│         └────────┬─────────┘                        │
└──────────────────┼──────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│                    Rust Layer                        │
│  ┌──────────────────────────────────────────────┐   │
│  │              src/lib.rs                       │   │
│  │  ┌────────────┐  ┌─────────────────────────┐ │   │
│  │  │ GameEnv    │  │ SimpleDuel              │ │   │
│  │  │  (Trait)   │  │  - 12x12 战术地图        │ │   │
│  │  └────────────┘  │  - 13 种动作            │ │   │
│  │                  │  - 地形/道具系统         │ │   │
│  │                  └─────────────────────────┘ │   │
│  │  ┌────────────────────────────────────────┐  │   │
│  │  │ Rayon 并行处理 N 个环境实例             │  │   │
│  │  └────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 核心设计：对称变换

两个玩家共享同一个神经网络，通过观测变换实现：

```
P1 视角 (物理坐标):          P2 视角 (变换后):
  ┌───────────┐               ┌───────────┐
  │ P1    P2  │    180° 旋转   │ P2'   P1' │
  │  ●     ○  │  ──────────▶  │  ●     ○  │
  │           │               │           │
  └───────────┘               └───────────┘
  "我在左下"                   "我在左下"
```

- 坐标翻转: `(x, y) → (MAP_SIZE-1-x, MAP_SIZE-1-y)`
- 方向动作翻转: 上↔下, 左↔右
- 非方向动作保持不变

### 数据流

```
1. env.reset() → [32, 160] 观测 (16环境 × 2玩家)
2. model.forward(obs, mask) → [32] 动作
3. 拆分: actions_p1 = actions[:16], actions_p2 = actions[16:]
4. env.step(actions_p1, actions_p2) → 新观测、奖励、结束标志
5. 存储转移数据
6. 每 2048 步: PPO 更新
7. 重复至 1000 万步
```

## 项目结构

```
RL-self-play/
├── src/
│   └── lib.rs          # Rust 环境实现
├── algorithms/
│   ├── base.py         # 算法基类
│   └── ppo.py          # PPO 算法
├── agents/
│   └── rule_based.py   # 规则智能体
├── model.py            # 神经网络模型
├── train.py            # 训练入口
├── watch.py            # 观看对战回放
├── config.py           # 训练配置
├── utils.py            # 工具函数
├── Cargo.toml          # Rust 依赖
├── CLAUDE.md           # Claude Code 指导
└── README.md           # 本文件
```

## 训练参数

| 参数 | 值 | 说明 |
|------|------|------|
| total_timesteps | 10,000,000 | 总训练步数 |
| num_envs | 16 | 并行环境数 |
| num_steps | 2048 | 每次 rollout 步数 |
| batch_size | 65,536 | PPO 批次大小 |
| learning_rate | 1e-4 | 学习率 |
| entropy_coef | 0.02 | 熵正则化系数 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE 参数 |

## 扩展

### 添加新游戏

实现 `GameEnv` trait：

```rust
pub trait GameEnv: Send + Sync + Clone {
    fn new() -> Self;
    fn reset(&mut self) -> (obs_p1, obs_p2, mask_p1, mask_p2);
    fn step(&mut self, action_p1, action_p2) -> (obs, rewards, done, masks, info);
    fn obs_dim() -> usize;
    fn action_dim() -> usize;
}
```

关键要求：
1. 观测必须对称（P2 视角需要变换）
2. 动作掩码在 Rust 端计算
3. 奖励设计为零和

## 许可证

MIT License
