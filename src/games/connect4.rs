use crate::traits::*;
use std::collections::HashMap;

// ============================================================================
// 4. Connect4 Implementation - 6x7 board game
// ============================================================================

pub const CONNECT4_ROWS: usize = 6;
pub const CONNECT4_COLS: usize = 7;
// obs_dim = 6*7*3 = 126 (3 channels: my pieces, opponent pieces, empty)
pub const CONNECT4_OBS_DIM: usize = CONNECT4_ROWS * CONNECT4_COLS * 3;
pub const CONNECT4_ACTION_DIM: usize = CONNECT4_COLS;

#[derive(Clone)]
pub struct Connect4 {
    // Board: 0 = empty, 1 = player 1, -1 = player 2
    pub board: [[i8; CONNECT4_COLS]; CONNECT4_ROWS],
    // Height of each column (next available row for each column)
    pub heights: [usize; CONNECT4_COLS],
    pub current_player: i8,
    pub step_count: i32,
}

impl Connect4 {
    /// Check if a column is valid for placing a piece
    fn is_valid_column(&self, col: usize) -> bool {
        col < CONNECT4_COLS && self.heights[col] < CONNECT4_ROWS
    }

    /// Drop a piece in the given column, returns the row where it landed
    fn drop_piece(&mut self, col: usize) -> Option<usize> {
        if !self.is_valid_column(col) {
            return None;
        }
        let row = self.heights[col];
        self.board[row][col] = self.current_player;
        self.heights[col] += 1;
        Some(row)
    }

    /// Check if the last move at (row, col) resulted in a win
    fn check_winner_at(&self, row: usize, col: usize) -> bool {
        let player = self.board[row][col];
        if player == 0 {
            return false;
        }

        // Check all 4 directions: horizontal, vertical, diagonal /, diagonal \
        let directions: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];

        for (dr, dc) in directions {
            let mut count = 1;

            // Count in positive direction
            let mut r = row as i32 + dr;
            let mut c = col as i32 + dc;
            while r >= 0
                && r < CONNECT4_ROWS as i32
                && c >= 0
                && c < CONNECT4_COLS as i32
                && self.board[r as usize][c as usize] == player
            {
                count += 1;
                r += dr;
                c += dc;
            }

            // Count in negative direction
            r = row as i32 - dr;
            c = col as i32 - dc;
            while r >= 0
                && r < CONNECT4_ROWS as i32
                && c >= 0
                && c < CONNECT4_COLS as i32
                && self.board[r as usize][c as usize] == player
            {
                count += 1;
                r -= dr;
                c -= dc;
            }

            if count >= 4 {
                return true;
            }
        }
        false
    }

    /// Check if the board is full (draw)
    fn is_board_full(&self) -> bool {
        self.heights.iter().all(|&h| h >= CONNECT4_ROWS)
    }

    /// Get observation for a player
    /// 3 channels: my pieces (1.0), opponent pieces (1.0), empty (1.0)
    /// Symmetric transform: flip piece colors for player -1
    fn get_obs_for_player(&self, player: i8) -> Vec<f32> {
        let mut obs = vec![0.0; CONNECT4_OBS_DIM];
        self.get_obs_into_for_player(player, &mut obs);
        obs
    }

    fn get_obs_into_for_player(&self, player: i8, obs: &mut [f32]) {
        obs.fill(0.0);
        for row in 0..CONNECT4_ROWS {
            for col in 0..CONNECT4_COLS {
                let cell = self.board[row][col];
                // Transform cell based on player perspective
                let transformed_cell = if player == 1 { cell } else { -cell };
                let base = (row * CONNECT4_COLS + col) * 3;
                match transformed_cell {
                    0 => obs[base] = 1.0,      // Empty channel
                    1 => obs[base + 1] = 1.0,  // My pieces channel
                    -1 => obs[base + 2] = 1.0, // Opponent pieces channel
                    _ => {}
                }
            }
        }
    }

    /// Get action mask (1.0 for valid columns, 0.0 for full columns)
    fn get_mask(&self) -> Vec<f32> {
        let mut mask = vec![0.0; CONNECT4_ACTION_DIM];
        self.get_mask_into(&mut mask);
        mask
    }

    fn get_mask_into(&self, mask: &mut [f32]) {
        for (col, slot) in mask.iter_mut().enumerate().take(CONNECT4_COLS) {
            *slot = if self.is_valid_column(col) { 1.0 } else { 0.0 };
        }
    }
}

impl GameEnv for Connect4 {
    fn new() -> Self {
        Connect4 {
            board: [[0; CONNECT4_COLS]; CONNECT4_ROWS],
            heights: [0; CONNECT4_COLS],
            current_player: 1,
            step_count: 0,
        }
    }

    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.board = [[0; CONNECT4_COLS]; CONNECT4_ROWS];
        self.heights = [0; CONNECT4_COLS];
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

        // Select action based on current player
        let action = if self.current_player == 1 {
            action_p1
        } else {
            action_p2
        };

        let mut r1 = 0.0;
        let mut r2 = 0.0;
        let mut done = false;
        let mut info = HashMap::new();

        // Try to drop piece
        if let Some(row) = self.drop_piece(action) {
            // Check for win
            if self.check_winner_at(row, action) {
                done = true;
                if self.current_player == 1 {
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
                // Draw
                done = true;
                info.insert("p1_win".to_string(), 0.0);
                info.insert("p2_win".to_string(), 0.0);
                info.insert("draw".to_string(), 1.0);
                info.insert("steps".to_string(), self.step_count as f32);
            }

            // Switch player
            self.current_player = -self.current_player;
        }

        let obs_p1 = self.get_obs_for_player(1);
        let obs_p2 = self.get_obs_for_player(-1);
        let mask = self.get_mask();

        (obs_p1, obs_p2, r1, r2, done, mask.clone(), mask, info)
    }

    fn obs_dim() -> usize {
        CONNECT4_OBS_DIM
    }

    fn action_dim() -> usize {
        CONNECT4_ACTION_DIM
    }
}

impl GameEnvZeroCopy for Connect4 {
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
        self.board = [[0; CONNECT4_COLS]; CONNECT4_ROWS];
        self.heights = [0; CONNECT4_COLS];
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

        if let Some(row) = self.drop_piece(action) {
            if self.check_winner_at(row, action) {
                done = true;
                if self.current_player == 1 {
                    r1 = 1.0;
                    r2 = -1.0;
                    info = GameInfo::terminal(TerminalStats {
                        p1_win: true,
                        steps: self.step_count,
                        ..Default::default()
                    });
                } else {
                    r1 = -1.0;
                    r2 = 1.0;
                    info = GameInfo::terminal(TerminalStats {
                        p2_win: true,
                        steps: self.step_count,
                        ..Default::default()
                    });
                }
            } else if self.is_board_full() {
                done = true;
                info = GameInfo::terminal(TerminalStats {
                    draw: true,
                    steps: self.step_count,
                    ..Default::default()
                });
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
        CONNECT4_OBS_DIM
    }

    fn action_dim() -> usize {
        CONNECT4_ACTION_DIM
    }
}
