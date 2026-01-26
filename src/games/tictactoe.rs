use crate::traits::*;
use std::collections::HashMap;

pub const TICTACTOE_OBS_DIM: usize = 27;
pub const TICTACTOE_ACTION_DIM: usize = 9;

#[derive(Clone)]
pub struct TicTacToe {
    pub board: [i8; 9],
    pub current_player: i8,
    pub step_count: i32,
}

impl TicTacToe {
    fn check_winner(&self) -> Option<i8> {
        const WIN_PATTERNS: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for pattern in WIN_PATTERNS {
            let a = self.board[pattern[0]];
            let b = self.board[pattern[1]];
            let c = self.board[pattern[2]];
            if a != 0 && a == b && b == c {
                return Some(a);
            }
        }
        None
    }

    fn is_board_full(&self) -> bool {
        self.board.iter().all(|&cell| cell != 0)
    }

    fn get_obs_for_player(&self, player: i8) -> Vec<f32> {
        let mut obs = vec![0.0; TICTACTOE_OBS_DIM];
        for (i, &cell) in self.board.iter().enumerate() {
            let transformed_cell = if player == 1 { cell } else { -cell };
            let base = i * 3;
            match transformed_cell {
                0 => obs[base] = 1.0,
                1 => obs[base + 1] = 1.0,
                -1 => obs[base + 2] = 1.0,
                _ => {}
            }
        }
        obs
    }

    fn get_obs_into_for_player(&self, player: i8, obs: &mut [f32]) {
        obs.fill(0.0);
        for (i, &cell) in self.board.iter().enumerate() {
            let transformed_cell = if player == 1 { cell } else { -cell };
            let base = i * 3;
            match transformed_cell {
                0 => obs[base] = 1.0,
                1 => obs[base + 1] = 1.0,
                -1 => obs[base + 2] = 1.0,
                _ => {}
            }
        }
    }

    fn get_mask(&self) -> Vec<f32> {
        let mut mask = vec![0.0; TICTACTOE_ACTION_DIM];
        for (i, &cell) in self.board.iter().enumerate() {
            if cell == 0 {
                mask[i] = 1.0;
            }
        }
        mask
    }

    fn get_mask_into(&self, mask: &mut [f32]) {
        for (i, &cell) in self.board.iter().enumerate() {
            mask[i] = if cell == 0 { 1.0 } else { 0.0 };
        }
    }
}

impl GameEnv for TicTacToe {
    fn new() -> Self {
        TicTacToe {
            board: [0; 9],
            current_player: 1,
            step_count: 0,
        }
    }

    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.board = [0; 9];
        self.current_player = 1;
        self.step_count = 0;

        let obs_p1 = self.get_obs_for_player(1);
        let obs_p2 = self.get_obs_for_player(-1);
        let mask = self.get_mask();

        (obs_p1, obs_p2, mask.clone(), mask)
    }

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
    ) {
        self.step_count += 1;

        let action = if self.current_player == 1 {
            action_p1
        } else {
            action_p2
        };

        let mut r1 = 0.0;
        let mut r2 = 0.0;
        let mut done = false;
        let mut info = HashMap::new();

        if action < 9 && self.board[action] == 0 {
            self.board[action] = self.current_player;

            if let Some(winner) = self.check_winner() {
                done = true;
                if winner == 1 {
                    r1 = 1.0;
                    r2 = -1.0;
                    info.insert("p1_win".to_string(), 1.0);
                    info.insert("p2_win".to_string(), 0.0);
                } else {
                    r1 = -1.0;
                    r2 = 1.0;
                    info.insert("p1_win".to_string(), 0.0);
                    info.insert("p2_win".to_string(), 1.0);
                }
                info.insert("draw".to_string(), 0.0);
                info.insert("steps".to_string(), self.step_count as f32);
            } else if self.is_board_full() {
                done = true;
                info.insert("p1_win".to_string(), 0.0);
                info.insert("p2_win".to_string(), 0.0);
                info.insert("draw".to_string(), 1.0);
                info.insert("steps".to_string(), self.step_count as f32);
            }

            self.current_player = -self.current_player;
        }

        let obs_p1 = self.get_obs_for_player(1);
        let obs_p2 = self.get_obs_for_player(-1);
        let mask = self.get_mask();

        (obs_p1, obs_p2, r1, r2, done, mask.clone(), mask, info)
    }

    fn obs_dim() -> usize {
        TICTACTOE_OBS_DIM
    }

    fn action_dim() -> usize {
        TICTACTOE_ACTION_DIM
    }
}

impl GameEnvZeroCopy for TicTacToe {
    fn new() -> Self {
        <Self as GameEnv>::new()
    }

    fn reset_into(
        &mut self,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) {
        self.board = [0; 9];
        self.current_player = 1;
        self.step_count = 0;

        self.get_obs_into_for_player(1, obs_p1);
        self.get_obs_into_for_player(-1, obs_p2);
        self.get_mask_into(mask_p1);
        self.get_mask_into(mask_p2);
    }

    fn step_into(
        &mut self,
        action_p1: usize,
        action_p2: usize,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) -> (f32, f32, bool, GameInfo) {
        self.step_count += 1;

        let action = if self.current_player == 1 {
            action_p1
        } else {
            action_p2
        };

        let mut r1 = 0.0;
        let mut r2 = 0.0;
        let mut done = false;
        let mut info = GameInfo::new();

        if action < 9 && self.board[action] == 0 {
            self.board[action] = self.current_player;

            if let Some(winner) = self.check_winner() {
                done = true;
                if winner == 1 {
                    r1 = 1.0;
                    r2 = -1.0;
                    info = GameInfo::terminal(true, false, false, 0, 0, 0, 0, self.step_count);
                } else {
                    r1 = -1.0;
                    r2 = 1.0;
                    info = GameInfo::terminal(false, true, false, 0, 0, 0, 0, self.step_count);
                }
            } else if self.is_board_full() {
                done = true;
                info = GameInfo::terminal(false, false, true, 0, 0, 0, 0, self.step_count);
            }

            self.current_player = -self.current_player;
        }

        self.get_obs_into_for_player(1, obs_p1);
        self.get_obs_into_for_player(-1, obs_p2);
        self.get_mask_into(mask_p1);
        self.get_mask_into(mask_p2);

        (r1, r2, done, info)
    }

    fn obs_dim() -> usize {
        TICTACTOE_OBS_DIM
    }

    fn action_dim() -> usize {
        TICTACTOE_ACTION_DIM
    }
}
