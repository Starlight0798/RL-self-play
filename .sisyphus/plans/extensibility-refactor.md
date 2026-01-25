# 高可扩展性架构重构 - 多游戏 & 多算法支持

## Context

### Original Request
在保持高可扩展性和高解耦性的基础上，添加 Rust 侧不同游戏支持，Python 侧不同算法支持。

### Interview Summary
**Key Discussions**:
- Rust 侧扩展方案: 宏生成方案 (为每个游戏生成专用 VectorizedEnv)
- 示例新游戏: TicTacToe (井字棋)
- 示例新算法: DQN (Deep Q-Network)
- RewardShaper: 移入 Rust 侧
- 文件组织: 多文件模块化 (src/games/*.rs)
- Python 环境抽象: 创建 envs/registry.py
- 测试基础设施: pytest + cargo test

**Research Findings**:
- VectorizedEnv 硬编码使用 SimpleDuel (lib.rs line ~1745)
- Python 侧已有 algorithms/registry.py 和 models/registry.py
- train.py 直接导入 PPO，未使用 registry
- obs_dim=160, action_dim=13 硬编码在多处
- RewardShaper 与 SimpleDuel 观测空间强耦合

### Metis Review
**Identified Gaps** (addressed):
- RewardShaper 位置决策: 确认移入 Rust 侧，每个游戏自行处理
- 观测维度获取: 使用 env.obs_dim() 动态获取
- 宏生成策略: 同时生成 VectorizedEnv 和 VectorizedEnvZeroCopy
- 测试覆盖范围: 新代码 + 端到端集成测试

---

## Work Objectives

### Core Objective
重构 RL-self-play 框架，实现 Rust 侧多游戏支持和 Python 侧多算法支持，同时保持高可扩展性和解耦性。

### Concrete Deliverables
1. Rust 侧多文件模块化结构 (src/games/*.rs)
2. 宏生成的 VectorizedEnv 系统
3. TicTacToe 示例游戏
4. Python 环境注册表 (envs/registry.py)
5. DQN 示例算法
6. 重构后的 train.py (使用 registry)
7. 测试基础设施 (pytest + cargo test)

### Definition of Done
- [ ] `cargo check` 通过
- [ ] `cargo clippy` 无警告
- [ ] `maturin develop --release` 成功构建
- [ ] `python train.py --config fast` 正常运行 (向后兼容)
- [ ] `python train.py --env tictactoe --algo dqn` 可以运行
- [ ] 所有 pytest 测试通过
- [ ] 所有 cargo test 测试通过

### Must Have
- Rust 宏生成 VectorizedEnv 和 VectorizedEnvZeroCopy
- SimpleDuel 行为与重构前完全一致
- TicTacToe 可以完成一局游戏
- DQN 继承 BaseAlgorithm 并实现所有抽象方法
- 环境注册表支持 list_envs() 和 get_env()
- train.py 使用 env.obs_dim() 和 env.action_dim()

### Must NOT Have (Guardrails)
- **MUST NOT** 修改 GameEnv / GameEnvZeroCopy trait 签名
- **MUST NOT** 改变 SimpleDuel 的观测向量布局 (索引 0-159 的含义)
- **MUST NOT** 在 DQN 中硬编码游戏特定逻辑
- **MUST NOT** 删除现有的 VectorizedEnv 类名 (保持向后兼容)
- **MUST NOT** 添加 Double DQN 或 Prioritized Experience Replay
- **MUST NOT** 添加 TicTacToe 变体 (只实现标准 3x3)
- **MUST NOT** 添加 CI/CD 或 coverage 工具

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (需要添加)
- **User wants tests**: YES (TDD for new code)
- **Framework**: pytest (Python) + cargo test (Rust)

### Test Setup Task
- [ ] 0. Setup Test Infrastructure
  - Install: `pip install pytest pytest-cov`
  - Config: Create `pytest.ini` and `tests/` directory
  - Verify: `pytest --help` shows help
  - Rust: `cargo test` already available

### TDD Workflow
Each new feature follows RED-GREEN-REFACTOR:
1. **RED**: Write failing test first
2. **GREEN**: Implement minimum code to pass
3. **REFACTOR**: Clean up while keeping green

---

## Task Flow

```
Phase 1: Rust 重构 (不改变行为)
  Task 0 (Setup) → Task 1 (模块化) → Task 2 (宏系统) → Task 3 (验证)

Phase 2: Python 环境抽象
  Task 4 (registry) → Task 5 (train.py 重构) → Task 6 (验证)

Phase 3: 新游戏 (TicTacToe)
  Task 7 (游戏逻辑) → Task 8 (注册) → Task 9 (验证)

Phase 4: 新算法 (DQN)
  Task 10 (DQN 实现) → Task 11 (注册) → Task 12 (验证)

Phase 5: 测试基础设施
  Task 13 (pytest) → Task 14 (cargo test) → Task 15 (集成测试)
```

## Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| A | 7, 10 | TicTacToe 和 DQN 可以并行开发 (Phase 3 & 4) |
| B | 13, 14 | pytest 和 cargo test 可以并行设置 |

| Task | Depends On | Reason |
|------|------------|--------|
| 1 | 0 | 需要先设置测试基础设施 |
| 2 | 1 | 宏系统依赖模块化结构 |
| 4 | 3 | Python 抽象依赖 Rust 重构完成 |
| 7 | 4 | TicTacToe 需要环境注册表 |
| 10 | 5 | DQN 需要 train.py 重构完成 |

---

## TODOs

### Phase 0: 基础设施

- [x] 0. 设置测试基础设施

  **What to do**:
  - 创建 `pytest.ini` 配置文件
  - 创建 `tests/` 目录结构
  - 创建 `tests/__init__.py`
  - 创建 `tests/conftest.py` 共享 fixtures
  - 验证 `cargo test` 可以运行 (已内置)

  **Must NOT do**:
  - 不添加 coverage 工具
  - 不添加 CI/CD 配置

  **Parallelizable**: NO (其他任务依赖此任务)

  **References**:
  - `algorithms/registry.py` - 参考现有 Python 代码风格
  - `Cargo.toml` - 确认 Rust 测试配置

  **Acceptance Criteria**:
  - [ ] `pytest --help` 显示帮助信息
  - [ ] `pytest tests/` 运行成功 (即使没有测试)
  - [ ] `cargo test` 运行成功

  **Commit**: YES
  - Message: `chore: add test infrastructure (pytest + cargo test)`
  - Files: `pytest.ini`, `tests/__init__.py`, `tests/conftest.py`

---

### Phase 1: Rust 重构

- [ ] 1. 创建 Rust 多文件模块结构

  **What to do**:
  - 创建 `src/games/mod.rs` 模块入口
  - 创建 `src/games/traits.rs` 存放 GameEnv 和 GameEnvZeroCopy traits
  - 创建 `src/games/simple_duel.rs` 移入 SimpleDuel 实现
  - 更新 `src/lib.rs` 使用模块导入
  - 保持所有公开 API 不变

  **Must NOT do**:
  - 不修改 trait 签名
  - 不改变 SimpleDuel 行为
  - 不改变观测向量布局

  **Parallelizable**: NO (后续任务依赖)

  **References**:
  - `src/lib.rs:1-100` - GameEnv 和 GameEnvZeroCopy trait 定义
  - `src/lib.rs:100-1700` - SimpleDuel 完整实现
  - `Cargo.toml` - 确认模块配置

  **Acceptance Criteria**:
  - [ ] `cargo check` 通过
  - [ ] `cargo clippy` 无警告
  - [ ] `maturin develop --release` 成功
  - [ ] `python -c "from high_perf_env import VectorizedEnv; print('OK')"` 输出 OK

  **Commit**: YES
  - Message: `refactor(rust): modularize game code into src/games/`
  - Files: `src/lib.rs`, `src/games/mod.rs`, `src/games/traits.rs`, `src/games/simple_duel.rs`

---

- [ ] 2. 创建 VectorizedEnv 宏生成系统

  **What to do**:
  - 在 `src/lib.rs` 中创建 `make_vectorized_env!` 宏
  - 宏应生成 `VectorizedEnv` 和 `VectorizedEnvZeroCopy` 两个版本
  - 使用宏重新生成 SimpleDuel 的向量化环境
  - 确保生成的类名与原有类名一致 (向后兼容)

  **宏设计** (CRITICAL):
  ```rust
  // 宏定义
  macro_rules! make_vectorized_env {
      ($env_name:ident, $game_type:ty, $py_class_name:literal, $py_class_name_zc:literal) => {
          // 生成标准 VectorizedEnv
          #[pyclass(name = $py_class_name)]
          pub struct $env_name {
              envs: Vec<$game_type>,
              // ... 其他字段
          }
          
          #[pymethods]
          impl $env_name {
              #[new]
              fn new(num_envs: usize) -> Self { ... }
              fn reset(&mut self) -> ... { ... }
              fn step(&mut self, actions_p1: ..., actions_p2: ...) -> ... { ... }
              fn obs_dim(&self) -> usize { <$game_type>::obs_dim() }
              fn action_dim(&self) -> usize { <$game_type>::action_dim() }
          }
          
          // 生成 ZeroCopy 版本 (类似结构)
          // ...
      };
  }
  
  // 使用宏生成 SimpleDuel 环境 (保持向后兼容)
  make_vectorized_env!(
      SimpleDuelVecEnv, 
      SimpleDuel, 
      "VectorizedEnv",           // Python 类名保持不变
      "VectorizedEnvZeroCopy"    // Python 类名保持不变
  );
  
  // 使用宏生成 TicTacToe 环境
  make_vectorized_env!(
      TicTacToeVecEnv, 
      TicTacToe, 
      "TicTacToeEnv",            // 新的 Python 类名
      "TicTacToeEnvZeroCopy"     // 新的 Python 类名
  );
  ```

  **向后兼容策略**:
  - SimpleDuel 的 Python 类名保持 `VectorizedEnv` 和 `VectorizedEnvZeroCopy`
  - 新游戏使用新的类名 (如 `TicTacToeEnv`)
  - 现有代码 `from high_perf_env import VectorizedEnv` 继续工作

  **Must NOT do**:
  - 不改变 VectorizedEnv 的 Python API
  - 不删除任何现有方法

  **Parallelizable**: NO (依赖 Task 1)

  **References**:
  - `src/lib.rs:1744-1889` - 现有 VectorizedEnv 实现 (作为宏模板)
  - `src/lib.rs:1905-2078` - 现有 VectorizedEnvZeroCopy 实现 (作为宏模板)
  - `src/lib.rs:2080-2090` - PyO3 模块注册

  **Acceptance Criteria**:
  - [ ] 宏定义存在且可编译
  - [ ] `cargo check` 通过
  - [ ] `maturin develop --release` 成功
  - [ ] `python -c "from high_perf_env import VectorizedEnv, VectorizedEnvZeroCopy; print('OK')"` 输出 OK
  - [ ] `python watch.py` 可以正常运行游戏

  **Commit**: YES
  - Message: `feat(rust): add make_vectorized_env! macro for game extensibility`
  - Files: `src/lib.rs`

---

- [ ] 3. 验证 Rust 重构正确性

  **What to do**:
  - 运行 `python watch.py` 观察游戏行为
  - 运行 `python train.py --config fast` 验证训练
  - 对比重构前后的行为一致性
  - 添加 Rust 单元测试验证 SimpleDuel 逻辑

  **Must NOT do**:
  - 不修改任何代码 (只验证)

  **Parallelizable**: NO (依赖 Task 2)

  **References**:
  - `watch.py` - 游戏可视化脚本
  - `train.py` - 训练脚本
  - `config.py` - fast 配置

  **Acceptance Criteria**:
  - [ ] `python watch.py` 游戏正常进行
  - [ ] `python train.py --config fast` 完成训练无错误
  - [ ] `cargo test` 所有测试通过

  **Commit**: YES
  - Message: `test(rust): add unit tests for SimpleDuel game logic`
  - Files: `src/games/simple_duel.rs` (添加 #[cfg(test)] 模块)

---

### Phase 2: Python 环境抽象

- [ ] 4. 创建 Python 环境注册表

  **What to do**:
  - 创建 `envs/__init__.py`
  - 创建 `envs/registry.py` 实现 `@register_env`, `get_env()`, `list_envs()`
  - 创建 `envs/base.py` 定义 `BaseEnv` 抽象类
  - 注册 SimpleDuel 环境

  **Must NOT do**:
  - 不添加版本管理或热加载功能
  - 不修改 Rust 侧代码

  **Parallelizable**: NO (依赖 Task 3)

  **References**:
  - `algorithms/registry.py` - 参考现有注册表模式
  - `models/registry.py` - 参考现有注册表模式

  **Acceptance Criteria**:
  - [ ] `from envs import get_env, list_envs` 导入成功
  - [ ] `list_envs()` 返回 `['simple_duel']`
  - [ ] `get_env('simple_duel', num_envs=4)` 返回可用环境

  **Commit**: YES
  - Message: `feat(python): add environment registry (envs/registry.py)`
  - Files: `envs/__init__.py`, `envs/registry.py`, `envs/base.py`

---

- [ ] 5. 重构 train.py 使用 registry 和动态维度

  **What to do**:
  - 修改 `train.py` 使用 `get_env()` 获取环境
  - 修改 `train.py` 使用 `get_algorithm()` 获取算法
  - 使用 `env.obs_dim()` 和 `env.action_dim()` 替代硬编码
  - 在 `config.py` 中添加 `env_name` 和 `algorithm_name` 字段
  - 添加命令行参数 `--env` 和 `--algo`
  - **在 `algorithms/base.py` 中添加 `is_on_policy` 类属性** (默认 True)
  - **在 `algorithms/ppo.py` 的 PPO 类中添加 `is_on_policy = True`**
  - **支持 on-policy 和 off-policy 算法的训练循环**:
    - 训练循环根据 `algo.is_on_policy` 选择对应模式
    - on-policy 循环: rollout N steps → update → repeat (现有逻辑)
    - off-policy 循环: step → store → update every K steps → repeat

  **BaseAlgorithm 修改** (CRITICAL):
  ```python
  # algorithms/base.py
  class BaseAlgorithm(ABC):
      is_on_policy: bool = True  # 默认为 on-policy，DQN 覆盖为 False
      
      @abstractmethod
      def get_action(self, obs, mask, deterministic=False): ...
      # ... 其他方法
  ```

  **训练循环设计**:
  ```python
  # train.py 伪代码
  if algo.is_on_policy:
      # PPO 风格: 收集完整 rollout 后更新
      for _ in range(num_updates):
          rollout = collect_rollout(env, algo, num_steps)
          algo.update(rollout)
  else:
      # DQN 风格: 每步存储，定期更新
      for step in range(total_steps):
          action = algo.get_action(obs, mask)
          next_obs, reward, done, info = env.step(action)
          algo.store_transition(obs, action, reward, done, mask, info)
          if step % update_freq == 0 and step > warmup_steps:
              algo.update()
          obs = next_obs
  ```

  **Must NOT do**:
  - 不改变默认行为 (默认仍为 simple_duel + ppo)
  - 不删除现有功能

  **Parallelizable**: NO (依赖 Task 4)

  **References**:
  - `train.py:1-50` - 当前导入和初始化逻辑
  - `train.py:22` - 硬编码的 obs_dim 和 action_dim
  - `train.py:60-150` - 现有训练循环 (on-policy 风格)
  - `config.py` - Config dataclass
  - `algorithms/registry.py` - get_algorithm() 用法
  - `algorithms/base.py` - BaseAlgorithm 接口 (需要添加 is_on_policy)
  - `algorithms/ppo.py` - PPO 实现 (需要添加 is_on_policy = True)

  **Acceptance Criteria**:
  - [ ] `python train.py --config fast` 正常运行 (向后兼容)
  - [ ] `python train.py --env simple_duel --algo ppo --config fast` 正常运行
  - [ ] 无硬编码的 obs_dim=160 或 action_dim=13
  - [ ] 训练循环支持 `is_on_policy=True` 和 `is_on_policy=False` 两种模式
  - [ ] `BaseAlgorithm.is_on_policy` 属性存在
  - [ ] `PPO.is_on_policy == True`

  **Commit**: YES
  - Message: `refactor(python): use registry pattern in train.py`
  - Files: `train.py`, `config.py`, `algorithms/base.py`, `algorithms/ppo.py`

---

- [ ] 6. 验证 Python 重构正确性

  **What to do**:
  - 运行完整训练验证
  - 检查模型保存和加载
  - 验证 watch.py 仍然工作

  **Must NOT do**:
  - 不修改任何代码 (只验证)

  **Parallelizable**: NO (依赖 Task 5)

  **References**:
  - `train.py` - 训练脚本
  - `watch.py` - 可视化脚本

  **Acceptance Criteria**:
  - [ ] `python train.py --config fast` 完成训练
  - [ ] 模型文件 `ppo_checkpoint.pth` 生成
  - [ ] `python watch.py` 可以加载模型并运行

  **Commit**: NO (只验证)

---

### Phase 3: 新游戏 (TicTacToe)

- [ ] 7. 实现 TicTacToe 游戏逻辑 (Rust)

  **What to do**:
  - 创建 `src/games/tic_tac_toe.rs`
  - 实现 `TicTacToe` struct
  - 实现 `GameEnv` trait
  - 实现 `GameEnvZeroCopy` trait
  - 实现对称变换 (180° 旋转)
  - 使用 `make_vectorized_env!` 宏生成向量化环境

  **观测空间设计** (CRITICAL):
  - `obs_dim() = 27`: 3x3 棋盘 × 3 通道 (空/己方/对方)
  - 观测向量布局:
    - `[0-8]`: 空位 one-hot (1 = 空, 0 = 占用)
    - `[9-17]`: 己方棋子 one-hot (1 = 己方, 0 = 非己方)
    - `[18-26]`: 对方棋子 one-hot (1 = 对方, 0 = 非对方)
  - 棋盘索引映射:
    ```
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
    ```
  - `action_dim() = 9`: 对应 9 个格子的落子位置
  - action mask: 只有空位可落子

  **对称变换设计** (180° 旋转):
  - P2 视角 = P1 视角旋转 180°
  - 位置映射: `transform_pos(i) = 8 - i`
  - 观测变换: 交换己方/对方通道 + 位置旋转
  - 动作变换: `transform_action(a) = 8 - a`

  **奖励设计**:
  - 胜利: +1.0
  - 失败: -1.0
  - 平局: 0.0
  - 中间步骤: 0.0 (无 reward shaping)

  **Must NOT do**:
  - 不实现变体 (只标准 3x3)
  - 不添加复杂的 reward shaping

  **Parallelizable**: YES (与 Task 10 并行)

  **References**:
  - `src/games/simple_duel.rs` - 参考现有游戏实现模式
  - `src/games/traits.rs` - GameEnv 和 GameEnvZeroCopy trait 定义

  **Acceptance Criteria**:
  - [ ] `cargo check` 通过
  - [ ] `cargo test` TicTacToe 测试通过
  - [ ] `maturin develop --release` 成功
  - [ ] `python -c "from high_perf_env import TicTacToeEnv; print('OK')"` 输出 OK
  - [ ] `TicTacToeEnv.obs_dim()` 返回 27
  - [ ] `TicTacToeEnv.action_dim()` 返回 9

  **Commit**: YES
  - Message: `feat(rust): add TicTacToe game implementation`
  - Files: `src/games/tic_tac_toe.rs`, `src/games/mod.rs`, `src/lib.rs`

---

- [ ] 8. 注册 TicTacToe 到 Python 环境注册表

  **What to do**:
  - 在 `envs/registry.py` 中注册 TicTacToe
  - 创建 TicTacToe 的 Python wrapper (如需要)
  - 更新 `list_envs()` 返回值

  **Must NOT do**:
  - 不修改 TicTacToe 的 Rust 实现

  **Parallelizable**: NO (依赖 Task 7)

  **References**:
  - `envs/registry.py` - 环境注册表
  - `src/games/tic_tac_toe.rs` - TicTacToe 实现

  **Acceptance Criteria**:
  - [ ] `list_envs()` 返回 `['simple_duel', 'tictactoe']`
  - [ ] `get_env('tictactoe', num_envs=4)` 返回可用环境

  **Commit**: YES
  - Message: `feat(python): register TicTacToe in environment registry`
  - Files: `envs/registry.py`

---

- [ ] 9. 验证 TicTacToe 可训练

  **What to do**:
  - 使用 PPO 训练 TicTacToe
  - 验证训练循环正常运行
  - 观察学习曲线

  **Must NOT do**:
  - 不要求达到特定胜率

  **Parallelizable**: NO (依赖 Task 8)

  **References**:
  - `train.py` - 训练脚本

  **Acceptance Criteria**:
  - [ ] `python train.py --env tictactoe --algo ppo --config fast` 完成训练
  - [ ] 无运行时错误
  - [ ] 损失值正常下降

  **Commit**: NO (只验证)

---

### Phase 4: 新算法 (DQN)

- [ ] 10. 实现 DQN 算法

  **What to do**:
  - 创建 `algorithms/dqn.py`
  - 实现 `DQN` 类继承 `BaseAlgorithm`
  - 实现 experience replay buffer (在 `algorithms/dqn.py` 内部)
  - 实现 target network
  - 实现 epsilon-greedy 探索
  - 使用 `@register_algorithm('dqn')` 注册
  - **在 `config.py` 中添加 DQN 相关配置**

  **ReplayBuffer 设计** (CRITICAL):
  - 位置: 在 `algorithms/dqn.py` 内部实现 (不修改 buffers.py)
  - 接口:
    ```python
    class ReplayBuffer:
        def __init__(self, capacity: int):
            self.buffer = deque(maxlen=capacity)
            
        def add(self, obs, action, reward, next_obs, done, mask):
            self.buffer.append((obs, action, reward, next_obs, done, mask))
            
        def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
            batch = random.sample(self.buffer, batch_size)
            obs, action, reward, next_obs, done, mask = zip(*batch)
            return {
                "obs": torch.stack(obs),
                "action": torch.tensor(action),
                "reward": torch.tensor(reward),
                "next_obs": torch.stack(next_obs),
                "done": torch.tensor(done),
                "mask": torch.stack(mask)
            }
            
        def __len__(self) -> int:
            return len(self.buffer)
    ```

  **config.py DQN 配置** (CRITICAL):
  在 Config dataclass 中添加:
  ```python
  # DQN 相关配置
  buffer_size: int = 100000
  epsilon_start: float = 1.0
  epsilon_end: float = 0.01
  epsilon_decay_steps: int = 100000
  dqn_update_freq: int = 4
  dqn_warmup_steps: int = 10000
  target_update_freq: int = 1000
  dqn_batch_size: int = 32
  dqn_lr: float = 1e-4
  dqn_gamma: float = 0.99
  ```

  **BaseAlgorithm 接口适配** (CRITICAL):
  DQN 是 off-policy 算法，需要特殊适配 BaseAlgorithm 接口:

  ```python
  class DQN(BaseAlgorithm):
      is_on_policy = False  # 标识为 off-policy 算法
      
      def __init__(self, config, obs_dim, action_dim, device):
          self.replay_buffer = ReplayBuffer(capacity=config.buffer_size)
          self.q_network = DQNNetwork(obs_dim, action_dim)
          self.target_network = DQNNetwork(obs_dim, action_dim)
          self.epsilon = config.epsilon_start
          self.update_freq = config.dqn_update_freq
          self.warmup_steps = config.dqn_warmup_steps
          self.target_update_freq = config.target_update_freq
          
      def get_action(self, obs, mask, deterministic=False):
          # epsilon-greedy 探索
          if not deterministic and random.random() < self.epsilon:
              return random_valid_action(mask)
          return self.q_network(obs).argmax()
          
      def store_transition(self, obs, action, reward, done, mask, info):
          # 存入 replay buffer (不立即更新)
          # 注意: 需要存储 next_obs，从 info 或环境获取
          self.replay_buffer.add(obs, action, reward, next_obs, done, mask)
          
      def update(self, *args, **kwargs):
          # 从 replay buffer 采样并更新 Q-network
          if len(self.replay_buffer) < self.warmup_steps:
              return {}  # 预热期不更新
          batch = self.replay_buffer.sample(self.dqn_batch_size)
          loss = self._compute_td_loss(batch)
          # 定期更新 target network
          if self.step_count % self.target_update_freq == 0:
              self.target_network.load_state_dict(self.q_network.state_dict())
          return {"loss": loss}
  ```

  **Self-Play 适配**:
  - 两个玩家共享同一个 Q-network (对称 self-play)
  - Experience replay 存储双方的 transition
  - 对手采样: 从 replay buffer 中采样，对手是过去的自己

  **Must NOT do**:
  - 不实现 Double DQN
  - 不实现 Prioritized Experience Replay
  - 不实现 Dueling DQN
  - 不硬编码游戏特定逻辑
  - 不修改 buffers.py (ReplayBuffer 在 dqn.py 内部实现)

  **Parallelizable**: YES (与 Task 7 并行)

  **References**:
  - `algorithms/base.py` - BaseAlgorithm 抽象类 (Task 5 已添加 is_on_policy)
  - `algorithms/ppo.py` - 参考现有算法实现
  - `algorithms/registry.py` - 注册装饰器
  - `config.py` - 需要添加 DQN 配置

  **Acceptance Criteria**:
  - [ ] `from algorithms import get_algorithm` 导入成功
  - [ ] `get_algorithm('dqn', config=config, obs_dim=9, action_dim=9)` 返回 DQN 实例
  - [ ] DQN 实现所有 BaseAlgorithm 抽象方法
  - [ ] `DQN.is_on_policy == False`
  - [ ] ReplayBuffer 可以存储和采样 transition
  - [ ] config.py 包含所有 DQN 相关配置

  **Commit**: YES
  - Message: `feat(python): add DQN algorithm implementation`
  - Files: `algorithms/dqn.py`, `algorithms/__init__.py`, `config.py`

---

- [ ] 11. 创建 DQN 专用模型

  **What to do**:
  - 在 `model.py` 中添加 `DQNNetwork` 类
  - 使用 `@register_model('dqn')` 注册
  - 实现 Q-value 输出

  **Must NOT do**:
  - 不实现 Dueling 架构

  **Parallelizable**: NO (依赖 Task 10)

  **References**:
  - `model.py` - 现有模型实现
  - `models/registry.py` - 模型注册表

  **Acceptance Criteria**:
  - [ ] `get_model('dqn', obs_dim=9, action_dim=9)` 返回 DQNNetwork 实例
  - [ ] 模型输出形状正确 (batch_size, action_dim)

  **Commit**: YES
  - Message: `feat(python): add DQN network model`
  - Files: `model.py`

---

- [ ] 12. 验证 DQN 可训练

  **What to do**:
  - 使用 DQN 训练 TicTacToe
  - 验证训练循环正常运行
  - 观察学习曲线

  **Must NOT do**:
  - 不要求达到特定胜率

  **Parallelizable**: NO (依赖 Task 11)

  **References**:
  - `train.py` - 训练脚本

  **Acceptance Criteria**:
  - [ ] `python train.py --env tictactoe --algo dqn --config fast` 完成训练
  - [ ] 无运行时错误
  - [ ] Q-value 正常更新

  **Commit**: NO (只验证)

---

### Phase 5: 测试完善

- [ ] 13. 添加 Python 单元测试

  **What to do**:
  - 创建 `tests/test_envs.py` 测试环境注册表
  - 创建 `tests/test_algorithms.py` 测试算法注册表
  - 创建 `tests/test_models.py` 测试模型注册表
  - 创建 `tests/test_integration.py` 端到端测试

  **Must NOT do**:
  - 不添加 coverage 要求
  - 不添加性能测试

  **Parallelizable**: YES (与 Task 14 并行)

  **References**:
  - `envs/registry.py` - 环境注册表
  - `algorithms/registry.py` - 算法注册表
  - `models/registry.py` - 模型注册表

  **Acceptance Criteria**:
  - [ ] `pytest tests/` 所有测试通过
  - [ ] 测试覆盖注册表的基本功能

  **Commit**: YES
  - Message: `test(python): add unit tests for registries`
  - Files: `tests/test_envs.py`, `tests/test_algorithms.py`, `tests/test_models.py`, `tests/test_integration.py`

---

- [ ] 14. 添加 Rust 单元测试

  **What to do**:
  - 在 `src/games/simple_duel.rs` 添加 `#[cfg(test)]` 模块
  - 在 `src/games/tic_tac_toe.rs` 添加 `#[cfg(test)]` 模块
  - 测试游戏逻辑正确性
  - 测试对称变换正确性

  **Must NOT do**:
  - 不添加性能基准测试

  **Parallelizable**: YES (与 Task 13 并行)

  **References**:
  - `src/games/simple_duel.rs` - SimpleDuel 实现
  - `src/games/tic_tac_toe.rs` - TicTacToe 实现

  **Acceptance Criteria**:
  - [ ] `cargo test` 所有测试通过
  - [ ] 测试覆盖游戏核心逻辑

  **Commit**: YES
  - Message: `test(rust): add unit tests for game logic`
  - Files: `src/games/simple_duel.rs`, `src/games/tic_tac_toe.rs`

---

- [ ] 15. 最终集成验证

  **What to do**:
  - 运行所有测试
  - 验证所有组合可用 (simple_duel+ppo, simple_duel+dqn, tictactoe+ppo, tictactoe+dqn)
  - 更新 AGENTS.md 文档

  **Must NOT do**:
  - 不添加新功能

  **Parallelizable**: NO (最终验证)

  **References**:
  - `AGENTS.md` - 项目文档

  **Acceptance Criteria**:
  - [ ] `cargo test` 通过
  - [ ] `pytest tests/` 通过
  - [ ] 所有 4 种组合可以运行训练
  - [ ] AGENTS.md 更新完成

  **Commit**: YES
  - Message: `docs: update AGENTS.md with new extensibility features`
  - Files: `AGENTS.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 0 | `chore: add test infrastructure` | pytest.ini, tests/* | pytest --help |
| 1 | `refactor(rust): modularize game code` | src/games/*, src/lib.rs | cargo check |
| 2 | `feat(rust): add make_vectorized_env! macro` | src/lib.rs | maturin develop |
| 3 | `test(rust): add SimpleDuel unit tests` | src/games/simple_duel.rs | cargo test |
| 4 | `feat(python): add environment registry` | envs/* | python -c "from envs import list_envs" |
| 5 | `refactor(python): use registry in train.py` | train.py, config.py | python train.py --config fast |
| 7 | `feat(rust): add TicTacToe game` | src/games/tic_tac_toe.rs | cargo test |
| 8 | `feat(python): register TicTacToe` | envs/registry.py | list_envs() |
| 10 | `feat(python): add DQN algorithm` | algorithms/dqn.py | get_algorithm('dqn') |
| 11 | `feat(python): add DQN network model` | model.py | get_model('dqn') |
| 13 | `test(python): add unit tests` | tests/* | pytest tests/ |
| 14 | `test(rust): add game unit tests` | src/games/*.rs | cargo test |
| 15 | `docs: update AGENTS.md` | AGENTS.md | - |

---

## Success Criteria

### Verification Commands
```bash
# Rust 构建
cargo check           # Expected: no errors
cargo clippy          # Expected: no warnings
cargo test            # Expected: all tests pass
maturin develop --release  # Expected: successful build

# Python 测试
pytest tests/         # Expected: all tests pass

# 训练验证
python train.py --config fast                           # Expected: completes
python train.py --env simple_duel --algo ppo --config fast  # Expected: completes
python train.py --env tictactoe --algo ppo --config fast    # Expected: completes
python train.py --env tictactoe --algo dqn --config fast    # Expected: completes

# 可视化验证
python watch.py       # Expected: game runs normally
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Backward compatibility maintained
- [ ] AGENTS.md updated
