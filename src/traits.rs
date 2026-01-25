use crate::GameInfo;
use std::collections::HashMap;

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
    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    // 执行一步
    // 输入: P1 和 P2 的动作
    // 返回: (Obs_P1, Obs_P2, Reward_P1, Reward_P2, Done, Mask_P1, Mask_P2, Info)
    // Info: 统计信息，仅在 Done=true 时有内容，否则为空
    fn step(
        &mut self,
        action_p1: usize,
        action_p2: usize,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        f32,
        f32,
        bool,
        Vec<f32>,
        Vec<f32>,
        HashMap<String, f32>,
    );

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

pub trait GameMetadata {
    fn name() -> &'static str;
    fn obs_dim() -> usize;
    fn action_dim() -> usize;
    fn description() -> &'static str;
}
