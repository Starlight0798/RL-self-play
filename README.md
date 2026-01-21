这份文档重点强调了 **Rust 环境的通用性（基于 Trait 设计）**、**观测空间的对称性转化** 以及 **标准 PPO 算法** 的实现要求。

---

# 项目开发文档：基于 Rust (Generic) 与 Python 的对称自博弈 RL 环境

## 1. 项目概述

本项目旨在构建一个高性能、可扩展的强化学习环境框架。
核心需求如下：

1. **通用性 (Genericity)**：Rust 环境端需通过 Trait（特征）定义接口，允许未来轻松扩展新的游戏场景（如从 GridWorld 切换到连续空间对抗），而无需重写底层通信逻辑。
2. **对称性 (Symmetry)**：实现“相对视角”的观测转换。无论当前是 Player 1 还是 Player 2，输入给神经网络的观测数据（Observation）在语义上必须一致（例如永远是“我方在前，敌方在后”）。这使得同一个神经网络权重可以同时控制双方。
3. **极速交互**: 使用 Rust (2024 Edition) + Rayon 实现 CPU 端的环境大规模向量化（Vectorization）。
4. **算法**: 使用 Python 实现标准 PPO 算法，配合 Action Masking 进行 Self-Play 训练。

## 2. 技术栈

* **Rust**: Edition 2024。依赖：`pyo3`, `numpy` (Rust binding), `rayon`, `rand`, `thiserror`.
* **Python**: 3.10+。依赖：`torch`, `numpy`.
* **构建**: `maturin`。

## 3. 模块一：Rust 通用环境层 (`src/lib.rs`)

### 3.1 核心设计：`GameEnv` Trait

为了保证通过性，不要直接把游戏逻辑写死在 Python 接口类里。请定义一个 Trait，所有具体的游戏逻辑都实现这个 Trait。

```rust
pub trait GameEnv: Send + Sync + Clone {
    type Action;
    type Obs;
    
    // 创建新游戏
    fn new() -> Self;
    
    // 重置游戏，返回 (P1_Obs, P2_Obs, P1_Mask, P2_Mask)
    fn reset(&mut self) -> (Self::Obs, Self::Obs, Vec<f32>, Vec<f32>);
    
    // 执行一步，接收双方动作
    // 返回 (P1_Obs, P2_Obs, P1_Reward, P2_Reward, Done, P1_Mask, P2_Mask)
    fn step(&mut self, action_p1: Self::Action, action_p2: Self::Action) 
        -> (Self::Obs, Self::Obs, f32, f32, bool, Vec<f32>, Vec<f32>);
        
    // 获取观测数据的维度（用于 Python 初始化 tensor）
    fn obs_dim() -> usize;
    
    // 获取动作空间大小
    fn action_dim() -> usize;
}

```

### 3.2 具体实现：`SimpleDuel` 场景

实现上述 Trait 的一个具体例子：简化的对抗游戏。

* **状态**: 2D 网格 ()，包含 P1 和 P2 的位置、HP。
* **对称观测 (Canonical Observation)**:
* **关键点**: 不要返回绝对坐标。
* 对于 P1，返回 `[My_X, My_Y, Enemy_X, Enemy_Y, My_HP, Enemy_HP]`。
* 对于 P2，返回 `[My_X, My_Y, Enemy_X, Enemy_Y, My_HP, Enemy_HP]`（这里的 My 是 P2 自己，Enemy 是 P1）。
* **坐标系转换**: 建议将 P2 的视角进行物理翻转（例如 P1 视角看地图是左下原点，P2 视角看地图是右上原点），或者仅使用相对坐标（dx, dy），确保网络学到的是通用的“博弈策略”而非“地图死记硬背”。


* **动作空间 (Discrete)**:
* [停, 上, 下, 左, 右, 攻击] (6个离散动作)。


* **Action Mask**:
* 移动到边界/障碍物 = 非法。
* 攻击 CD 未好 = 非法。



### 3.3 向量化包装器 (`VectorizedEnv`)

这是一个暴露给 Python 的 PyClass。

* 内部持有 `Vec<Box<dyn GameEnv>>` 或具体泛型 `Vec<SimpleDuel>`。
* 利用 `rayon::par_iter` 并行处理 `step`。
* **数据布局**:
* Python 调用 `step(actions_p1_batch, actions_p2_batch)`。
* Rust 并行更新 N 个环境。
* Rust 将结果拼接成 Numpy 数组返回 Python。建议返回结构为：
* `obs_batch`: Shape `[2 * N, Obs_Dim]` (将 P1 和 P2 的观测堆叠，方便统一推理)
* `reward_batch`: Shape `[2 * N]`
* `done_batch`: Shape `[N]` (游戏结束是对等的)
* `mask_batch`: Shape `[2 * N, Action_Dim]`





## 4. 模块二：Python 算法层 (`train.py`)

### 4.1 网络架构 (Shared Policy)

由于 Rust 层已经做了**对称性处理**，Python 端只需要一个 Actor-Critic 网络。

* **Input**: `[Batch_Size, Obs_Dim]` (Batch_Size = 2 * Num_Envs)
* **Process**:
1. Forward Pass 拿到 Logits。
2. **Apply Mask**: `Logits[mask == 0] = -1e9`。
3. Softmax & Sample。


* **Output**: Actions `[Batch_Size]`。

### 4.2 训练循环 (Self-Play Logic)

不需要复杂的对手池逻辑，最简单的 Self-Play 即可验证系统。

1. **初始化**:
* Rust Env 初始化 N 个实例。
* PPO 模型 `model`。


2. **Rollout (采样)**:
* 从 Env 获取 `obs` (包含 P1 和 P2 的视角，共 2N 个数据)。
* 模型推理 `actions = model(obs)`。
* 将 `actions` 拆分为 `act_p1` (前 N 个) 和 `act_p2` (后 N 个) 传回 Rust。
* 存储 Trajectories。注意：P1 的对手是 P2，P2 的对手是 P1。


3. **PPO 更新**:
* 将收集到的所有数据（P1 的数据 + P2 的数据）混合在一起。
* 使用标准的 PPO Loss (Clipped Surrogate Objective) 进行更新。
* 计算 GAE (Generalized Advantage Estimation)。
* **不需要 Dual-Clip**。



### 4.3 奖励设计 (Zero-Sum)

* Rust 端返回的奖励应当是零和的，或者近似零和。
* 例如：P1 击中 P2 -> P1 得 +1 分，P2 得 -1 分。
* 这样模型会自然学会攻击（为了得分）和躲避（为了避免扣分）。

## 5. 交付文件清单

请编写以下代码文件：

1. **`Cargo.toml`**:
* 配置 `name = "high_perf_env"`。
* 配置 `crate-type = ["cdylib"]`。
* 添加 `pyo3 = { version = "0.20", features = ["extension-module"] }`。
* 添加 `numpy`, `rayon`, `rand`.


2. **`src/lib.rs`**:
* 定义 `trait GameEnv`。
* 实现 `struct SimpleDuel` (包含详细的 Action Mask 和 对称 Obs 逻辑)。
* 实现 `struct PyVectorizedEnv` (处理 Numpy 转换和 Rayon 并行)。


3. **`train.py`**:
* PPO 类 (ActorCritic, Mask 处理)。
* 主训练循环 (Rollout -> Learn)。
* 不依赖外部复杂库，纯 PyTorch 实现。



## 6. 特别注意 (Instructions for Coding)

* **对称性检查**: 在编写 `SimpleDuel::reset` 和 `step` 时，务必确保 `obs_p2` 是基于 P2 视角的“相对坐标”或“翻转坐标”，绝对不能直接返回 P2 的绝对坐标，否则单一网络无法同时学会两边的策略。
* **Masking**: 在 Rust 端计算 Mask，不要在 Python 端计算，以保证性能。
* **Rust 2024**: 使用最新的 Rust 语法特性（如有）。
* **注释**: 关键逻辑（尤其是对称转换部分）请保留中文注释。