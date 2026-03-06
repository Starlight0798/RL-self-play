use crate::traits::*;
use crate::{
    ACTION_DIM, CONNECT4_ACTION_DIM, CONNECT4_OBS_DIM, Connect4, OBS_DIM, REVERSI_ACTION_DIM,
    REVERSI_OBS_DIM, Reversi, SimpleDuel, TICTACTOE_ACTION_DIM, TICTACTOE_OBS_DIM, TicTacToe,
};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Factory function type for creating game instances
pub type GameFactory = fn() -> GameEnvDispatch;

/// Game registry type: maps game name to (factory, obs_dim, action_dim)
pub type GameRegistry = HashMap<&'static str, (GameFactory, usize, usize)>;

/// Global game registry
pub static GAME_REGISTRY: LazyLock<GameRegistry> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    registry.insert(
        "simple_duel",
        (
            (|| GameEnvDispatch::SimpleDuel(<SimpleDuel as GameEnv>::new())) as GameFactory,
            OBS_DIM,
            ACTION_DIM,
        ),
    );
    registry.insert(
        "tictactoe",
        (
            (|| GameEnvDispatch::TicTacToe(<TicTacToe as GameEnv>::new())) as GameFactory,
            TICTACTOE_OBS_DIM,
            TICTACTOE_ACTION_DIM,
        ),
    );
    registry.insert(
        "connect4",
        (
            (|| GameEnvDispatch::Connect4(<Connect4 as GameEnv>::new())) as GameFactory,
            CONNECT4_OBS_DIM,
            CONNECT4_ACTION_DIM,
        ),
    );
    registry.insert(
        "reversi",
        (
            (|| GameEnvDispatch::Reversi(<Reversi as GameEnv>::new())) as GameFactory,
            REVERSI_OBS_DIM,
            REVERSI_ACTION_DIM,
        ),
    );
    registry
});

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum GameEnvDispatch {
    SimpleDuel(SimpleDuel),
    TicTacToe(TicTacToe),
    Connect4(Connect4),
    Reversi(Reversi),
}

#[allow(dead_code)]
impl GameEnvDispatch {
    pub fn reset(&mut self) -> GameReset {
        match self {
            GameEnvDispatch::SimpleDuel(env) => env.reset(),
            GameEnvDispatch::TicTacToe(env) => env.reset(),
            GameEnvDispatch::Connect4(env) => env.reset(),
            GameEnvDispatch::Reversi(env) => env.reset(),
        }
    }

    pub fn step(&mut self, action_p1: usize, action_p2: usize) -> GameStep {
        match self {
            GameEnvDispatch::SimpleDuel(env) => env.step(action_p1, action_p2),
            GameEnvDispatch::TicTacToe(env) => env.step(action_p1, action_p2),
            GameEnvDispatch::Connect4(env) => env.step(action_p1, action_p2),
            GameEnvDispatch::Reversi(env) => env.step(action_p1, action_p2),
        }
    }

    pub fn reset_into(
        &mut self,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) {
        match self {
            GameEnvDispatch::SimpleDuel(env) => env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2),
            GameEnvDispatch::TicTacToe(env) => env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2),
            GameEnvDispatch::Connect4(env) => env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2),
            GameEnvDispatch::Reversi(env) => env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2),
        }
    }

    pub fn step_into(
        &mut self,
        action_p1: usize,
        action_p2: usize,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) -> (f32, f32, bool, GameInfo) {
        match self {
            GameEnvDispatch::SimpleDuel(env) => {
                env.step_into(action_p1, action_p2, obs_p1, obs_p2, mask_p1, mask_p2)
            }
            GameEnvDispatch::TicTacToe(env) => {
                env.step_into(action_p1, action_p2, obs_p1, obs_p2, mask_p1, mask_p2)
            }
            GameEnvDispatch::Connect4(env) => {
                env.step_into(action_p1, action_p2, obs_p1, obs_p2, mask_p1, mask_p2)
            }
            GameEnvDispatch::Reversi(env) => {
                env.step_into(action_p1, action_p2, obs_p1, obs_p2, mask_p1, mask_p2)
            }
        }
    }

    pub fn obs_dim(&self) -> usize {
        match self {
            GameEnvDispatch::SimpleDuel(_) => <SimpleDuel as GameEnv>::obs_dim(),
            GameEnvDispatch::TicTacToe(_) => <TicTacToe as GameEnv>::obs_dim(),
            GameEnvDispatch::Connect4(_) => <Connect4 as GameEnv>::obs_dim(),
            GameEnvDispatch::Reversi(_) => <Reversi as GameEnv>::obs_dim(),
        }
    }

    pub fn action_dim(&self) -> usize {
        match self {
            GameEnvDispatch::SimpleDuel(_) => <SimpleDuel as GameEnv>::action_dim(),
            GameEnvDispatch::TicTacToe(_) => <TicTacToe as GameEnv>::action_dim(),
            GameEnvDispatch::Connect4(_) => <Connect4 as GameEnv>::action_dim(),
            GameEnvDispatch::Reversi(_) => <Reversi as GameEnv>::action_dim(),
        }
    }
}
