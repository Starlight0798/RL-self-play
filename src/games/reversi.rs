use crate::traits::*;
use std::collections::HashMap;

pub const REVERSI_SIZE: usize = 8;
// obs_dim = 8*8*3 = 192 (3 channels: my pieces, opponent pieces, empty)
pub const REVERSI_OBS_DIM: usize = REVERSI_SIZE * REVERSI_SIZE * 3;
pub const REVERSI_ACTION_DIM: usize = REVERSI_SIZE * REVERSI_SIZE;

#[derive(Clone)]
pub struct Reversi {
    // Board: 0 = empty, 1 = player 1 (black), -1 = player 2 (white)
    board: [[i8; REVERSI_SIZE]; REVERSI_SIZE],
    current_player: i8,
    step_count: i32,
    // Track consecutive passes for game end detection
    consecutive_passes: i32,
}

impl Reversi {
    /// Convert (row, col) to action index
    fn pos_to_action(row: usize, col: usize) -> usize {
        row * REVERSI_SIZE + col
    }

    /// Convert action index to (row, col)
    fn action_to_pos(action: usize) -> (usize, usize) {
        (action / REVERSI_SIZE, action % REVERSI_SIZE)
    }

    /// Check if a position is within bounds
    fn is_valid_pos(row: i32, col: i32) -> bool {
        row >= 0 && row < REVERSI_SIZE as i32 && col >= 0 && col < REVERSI_SIZE as i32
    }

    /// Check if a move is valid for the given player
    /// A move is valid if it outflanks at least one opponent piece
    fn is_valid_move(&self, row: usize, col: usize, player: i8) -> bool {
        // Cell must be empty
        if self.board[row][col] != 0 {
            return false;
        }

        let opponent = -player;
        let directions: [(i32, i32); 8] = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];

        for (dr, dc) in directions {
            let mut r = row as i32 + dr;
            let mut c = col as i32 + dc;
            let mut found_opponent = false;

            // Move in direction while finding opponent pieces
            while Self::is_valid_pos(r, c) && self.board[r as usize][c as usize] == opponent {
                found_opponent = true;
                r += dr;
                c += dc;
            }

            // Check if we found our own piece after opponent pieces
            if found_opponent
                && Self::is_valid_pos(r, c)
                && self.board[r as usize][c as usize] == player
            {
                return true;
            }
        }

        false
    }

    /// Get all pieces that would be flipped by placing a piece at (row, col)
    fn get_flips(&self, row: usize, col: usize, player: i8) -> Vec<(usize, usize)> {
        let mut flips = Vec::new();
        let opponent = -player;
        let directions: [(i32, i32); 8] = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];

        for (dr, dc) in directions {
            let mut r = row as i32 + dr;
            let mut c = col as i32 + dc;
            let mut line_flips = Vec::new();

            // Collect opponent pieces in this direction
            while Self::is_valid_pos(r, c) && self.board[r as usize][c as usize] == opponent {
                line_flips.push((r as usize, c as usize));
                r += dr;
                c += dc;
            }

            // If we found our own piece at the end, add all collected pieces to flips
            if !line_flips.is_empty()
                && Self::is_valid_pos(r, c)
                && self.board[r as usize][c as usize] == player
            {
                flips.extend(line_flips);
            }
        }

        flips
    }

    /// Place a piece and flip captured pieces
    fn make_move(&mut self, row: usize, col: usize) -> bool {
        if !self.is_valid_move(row, col, self.current_player) {
            return false;
        }

        let flips = self.get_flips(row, col, self.current_player);

        // Place the piece
        self.board[row][col] = self.current_player;

        // Flip captured pieces
        for (fr, fc) in flips {
            self.board[fr][fc] = self.current_player;
        }

        true
    }

    /// Check if the current player has any valid moves
    fn has_valid_moves(&self, player: i8) -> bool {
        for row in 0..REVERSI_SIZE {
            for col in 0..REVERSI_SIZE {
                if self.is_valid_move(row, col, player) {
                    return true;
                }
            }
        }
        false
    }

    /// Count pieces for each player
    fn count_pieces(&self) -> (i32, i32) {
        let mut p1_count = 0;
        let mut p2_count = 0;
        for row in 0..REVERSI_SIZE {
            for col in 0..REVERSI_SIZE {
                match self.board[row][col] {
                    1 => p1_count += 1,
                    -1 => p2_count += 1,
                    _ => {}
                }
            }
        }
        (p1_count, p2_count)
    }

    /// Check if the game is over (both players have no valid moves)
    fn is_game_over(&self) -> bool {
        !self.has_valid_moves(1) && !self.has_valid_moves(-1)
    }

    /// Get observation for a player
    /// 3 channels: empty (1.0), my pieces (1.0), opponent pieces (1.0)
    /// Symmetric transform: flip piece colors for player -1
    fn get_obs_for_player(&self, player: i8) -> Vec<f32> {
        let mut obs = vec![0.0; REVERSI_OBS_DIM];
        self.get_obs_into_for_player(player, &mut obs);
        obs
    }

    fn get_obs_into_for_player(&self, player: i8, obs: &mut [f32]) {
        obs.fill(0.0);
        for row in 0..REVERSI_SIZE {
            for col in 0..REVERSI_SIZE {
                let cell = self.board[row][col];
                // Transform cell based on player perspective
                let transformed_cell = if player == 1 { cell } else { -cell };
                let base = (row * REVERSI_SIZE + col) * 3;
                match transformed_cell {
                    0 => obs[base] = 1.0,      // Empty channel
                    1 => obs[base + 1] = 1.0,  // My pieces channel
                    -1 => obs[base + 2] = 1.0, // Opponent pieces channel
                    _ => {}
                }
            }
        }
    }

    /// Get action mask for a player (1.0 for valid moves, 0.0 for invalid)
    fn get_mask_for_player(&self, player: i8) -> Vec<f32> {
        let mut mask = vec![0.0; REVERSI_ACTION_DIM];
        self.get_mask_into_for_player(player, &mut mask);
        mask
    }

    fn get_mask_into_for_player(&self, player: i8, mask: &mut [f32]) {
        for row in 0..REVERSI_SIZE {
            for col in 0..REVERSI_SIZE {
                let action = Self::pos_to_action(row, col);
                mask[action] = if self.is_valid_move(row, col, player) {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }
}

impl GameEnv for Reversi {
    fn new() -> Self {
        let mut board = [[0i8; REVERSI_SIZE]; REVERSI_SIZE];
        // Initial setup: 4 pieces in center
        // Standard Reversi starting position
        board[3][3] = -1; // White
        board[3][4] = 1; // Black
        board[4][3] = 1; // Black
        board[4][4] = -1; // White

        Reversi {
            board,
            current_player: 1, // Black moves first
            step_count: 0,
            consecutive_passes: 0,
        }
    }

    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.board = [[0i8; REVERSI_SIZE]; REVERSI_SIZE];
        // Initial setup
        self.board[3][3] = -1;
        self.board[3][4] = 1;
        self.board[4][3] = 1;
        self.board[4][4] = -1;
        self.current_player = 1;
        self.step_count = 0;
        self.consecutive_passes = 0;

        let obs_p1 = self.get_obs_for_player(1);
        let obs_p2 = self.get_obs_for_player(-1);
        let mask_p1 = self.get_mask_for_player(1);
        let mask_p2 = self.get_mask_for_player(-1);

        (obs_p1, obs_p2, mask_p1, mask_p2)
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

        // Check if current player has valid moves
        let has_moves = self.has_valid_moves(self.current_player);

        if has_moves {
            // Try to make the move
            let (row, col) = Self::action_to_pos(action);
            if row < REVERSI_SIZE && col < REVERSI_SIZE && self.make_move(row, col) {
                self.consecutive_passes = 0;
            }
            // If invalid move was attempted, we just skip (action mask should prevent this)
        } else {
            // Player must pass
            self.consecutive_passes += 1;
        }

        // Switch player
        self.current_player = -self.current_player;

        // Check if game is over
        if self.is_game_over() || self.consecutive_passes >= 2 {
            done = true;
            let (p1_count, p2_count) = self.count_pieces();

            if p1_count > p2_count {
                r1 = 1.0;
                r2 = -1.0;
                info.insert("p1_win".to_string(), 1.0);
                info.insert("p2_win".to_string(), 0.0);
                info.insert("draw".to_string(), 0.0);
            } else if p2_count > p1_count {
                r1 = -1.0;
                r2 = 1.0;
                info.insert("p1_win".to_string(), 0.0);
                info.insert("p2_win".to_string(), 1.0);
                info.insert("draw".to_string(), 0.0);
            } else {
                info.insert("p1_win".to_string(), 0.0);
                info.insert("p2_win".to_string(), 0.0);
                info.insert("draw".to_string(), 1.0);
            }
            info.insert("steps".to_string(), self.step_count as f32);
            info.insert("p1_pieces".to_string(), p1_count as f32);
            info.insert("p2_pieces".to_string(), p2_count as f32);
        }

        let obs_p1 = self.get_obs_for_player(1);
        let obs_p2 = self.get_obs_for_player(-1);
        let mask_p1 = self.get_mask_for_player(1);
        let mask_p2 = self.get_mask_for_player(-1);

        (obs_p1, obs_p2, r1, r2, done, mask_p1, mask_p2, info)
    }

    fn obs_dim() -> usize {
        REVERSI_OBS_DIM
    }

    fn action_dim() -> usize {
        REVERSI_ACTION_DIM
    }
}

impl GameEnvZeroCopy for Reversi {
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
        self.board = [[0i8; REVERSI_SIZE]; REVERSI_SIZE];
        self.board[3][3] = -1;
        self.board[3][4] = 1;
        self.board[4][3] = 1;
        self.board[4][4] = -1;
        self.current_player = 1;
        self.step_count = 0;
        self.consecutive_passes = 0;

        self.get_obs_into_for_player(1, obs_p1);
        self.get_obs_into_for_player(-1, obs_p2);
        self.get_mask_into_for_player(1, mask_p1);
        self.get_mask_into_for_player(-1, mask_p2);
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

        // Check if current player has valid moves
        let has_moves = self.has_valid_moves(self.current_player);

        if has_moves {
            let (row, col) = Self::action_to_pos(action);
            if row < REVERSI_SIZE && col < REVERSI_SIZE && self.make_move(row, col) {
                self.consecutive_passes = 0;
            }
        } else {
            self.consecutive_passes += 1;
        }

        // Switch player
        self.current_player = -self.current_player;

        // Check if game is over
        if self.is_game_over() || self.consecutive_passes >= 2 {
            done = true;
            let (p1_count, p2_count) = self.count_pieces();

            if p1_count > p2_count {
                r1 = 1.0;
                r2 = -1.0;
                info = GameInfo::terminal(TerminalStats {
                    p1_win: true,
                    p1_damage: p1_count,
                    p2_damage: p2_count,
                    steps: self.step_count,
                    ..Default::default()
                });
            } else if p2_count > p1_count {
                r1 = -1.0;
                r2 = 1.0;
                info = GameInfo::terminal(TerminalStats {
                    p2_win: true,
                    p1_damage: p1_count,
                    p2_damage: p2_count,
                    steps: self.step_count,
                    ..Default::default()
                });
            } else {
                info = GameInfo::terminal(TerminalStats {
                    draw: true,
                    p1_damage: p1_count,
                    p2_damage: p2_count,
                    steps: self.step_count,
                    ..Default::default()
                });
            }
        }

        self.get_obs_into_for_player(1, obs_p1);
        self.get_obs_into_for_player(-1, obs_p2);
        self.get_mask_into_for_player(1, mask_p1);
        self.get_mask_into_for_player(-1, mask_p2);

        (r1, r2, done, info)
    }

    fn obs_dim() -> usize {
        REVERSI_OBS_DIM
    }

    fn action_dim() -> usize {
        REVERSI_ACTION_DIM
    }
}
