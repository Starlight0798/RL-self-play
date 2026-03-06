use std::collections::HashMap;

pub type GameStepInfo = HashMap<String, f32>;
pub type GameReset = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
pub type GameStep = (
    Vec<f32>,
    Vec<f32>,
    f32,
    f32,
    bool,
    Vec<f32>,
    Vec<f32>,
    GameStepInfo,
);

// ============================================================================
// 1. 核心 Trait 定义
// ============================================================================

/// 定义通用游戏环境接口
/// 所有的具体游戏逻辑（如 GridWorld, 连续控制等）都应实现此 Trait
pub trait GameEnv: Send + Sync + Clone {
    // 这里为了简化 Python 交互，我们假定 Action 是 usize (离散)，Obs 是 Vec<f32>
    // 如果需要连续动作，可以改为 Vec<f32>

    // 创建新游戏实例
    fn new() -> Self;

    // 重置游戏
    // 返回 (Obs_P1, Obs_P2, Mask_P1, Mask_P2)
    // Mask: 1.0 表示合法，0.0 表示非法
    fn reset(&mut self) -> GameReset;

    // 执行一步
    // 输入: P1 和 P2 的动作
    // 返回: (Obs_P1, Obs_P2, Reward_P1, Reward_P2, Done, Mask_P1, Mask_P2, Info)
    // Info: 统计信息，仅在 Done=true 时有内容，否则为空
    fn step(&mut self, action_p1: usize, action_p2: usize) -> GameStep;

    fn obs_dim() -> usize;
    fn action_dim() -> usize;
}

/// Zero-copy game environment interface for high-performance vectorized execution.
/// Implementations write directly into pre-allocated buffers.
pub trait GameEnvZeroCopy: Send + Sync + Clone {
    fn new() -> Self;

    /// Reset and write observations/masks into provided buffers
    fn reset_into(
        &mut self,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    );

    /// Step and write results into provided buffers
    /// Returns (reward_p1, reward_p2, done, info)
    fn step_into(
        &mut self,
        action_p1: usize,
        action_p2: usize,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) -> (f32, f32, bool, GameInfo);

    fn obs_dim() -> usize;
    fn action_dim() -> usize;
}

/// Typed game info struct - replaces HashMap<String, f32> for zero allocation
#[derive(Clone, Default, Copy)]
pub struct GameInfo {
    pub p1_win: f32,
    pub p2_win: f32,
    pub draw: f32,
    pub p1_attacks: f32,
    pub p2_attacks: f32,
    pub p1_damage: f32,
    pub p2_damage: f32,
    pub steps: f32,
    pub is_terminal: bool,
}

#[derive(Clone, Copy, Default)]
pub struct TerminalStats {
    pub p1_win: bool,
    pub p2_win: bool,
    pub draw: bool,
    pub p1_attacks: i32,
    pub p2_attacks: i32,
    pub p1_damage: i32,
    pub p2_damage: i32,
    pub steps: i32,
}

impl GameInfo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn terminal(stats: TerminalStats) -> Self {
        Self {
            p1_win: if stats.p1_win { 1.0 } else { 0.0 },
            p2_win: if stats.p2_win { 1.0 } else { 0.0 },
            draw: if stats.draw { 1.0 } else { 0.0 },
            p1_attacks: stats.p1_attacks as f32,
            p2_attacks: stats.p2_attacks as f32,
            p1_damage: stats.p1_damage as f32,
            p2_damage: stats.p2_damage as f32,
            steps: stats.steps as f32,
            is_terminal: true,
        }
    }
}
