import torch
import numpy as np
import high_perf_env
import time
import os

# --- Configuration Constants ---
# These should match the environment settings in src/lib.rs
MAP_SIZE = 8
MAX_HP = 3.0
MAX_ENERGY = 5.0
ACTION_NAMES = ["Stay", "Up", "Down", "Left", "Right", "Attack", "Shoot"]

class GameState:
    """
    Decoupled State Parser.
    Responsible for interpreting the raw observation vector into a structured, readable format.
    Assumes P1 perspective observation structure.
    """
    def __init__(self, obs_vector):
        # obs_vector: np.array shape (72,)
        self.raw = obs_vector
        
        # Parse P1 (My) info
        self.p1_pos = (
            self._denorm_pos(obs_vector[0]),
            self._denorm_pos(obs_vector[1])
        )
        self.p1_hp = self._denorm_hp(obs_vector[4])
        self.p1_eng = self._denorm_eng(obs_vector[6])
        
        # Parse P2 (Enemy) info
        self.p2_pos = (
            self._denorm_pos(obs_vector[2]),
            self._denorm_pos(obs_vector[3])
        )
        self.p2_hp = self._denorm_hp(obs_vector[5])
        self.p2_eng = self._denorm_eng(obs_vector[7])
        
        # Parse Walls
        # Walls are flattened 8x8 at the end (index 8 to 72)
        # Reshape to [Height, Width]
        self.walls = obs_vector[8:].reshape((MAP_SIZE, MAP_SIZE))

    def _denorm_pos(self, val):
        return int(np.clip(round(val * (MAP_SIZE - 1)), 0, MAP_SIZE - 1))
    
    def _denorm_hp(self, val):
        return int(round(val * MAX_HP))
    
    def _denorm_eng(self, val):
        return int(round(val * MAX_ENERGY))


class AsciiReplayLogger:
    """
    Handles the visualization and file I/O for game replays.
    Decoupled from the game loop and environment logic.
    """
    def __init__(self, output_dir="replays"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.filename = os.path.join(output_dir, f"replay_{timestamp}.txt")
        self.file = open(self.filename, "w", encoding="utf-8")
        
        self._write_header(timestamp)

    def _write_header(self, timestamp):
        self.file.write("\n" + "="*40 + "\n")
        self.file.write(f" >>> PLAYING TEST GAME @ {timestamp} <<<\n")
        self.file.write("="*40 + "\n\n")

    def log_step(self, step_idx, state: GameState, action_p1, action_p2, reward_p1, reward_p2, done, info=None):
        """
        Logs a single step of the game.
        """
        # 1. State Info
        self.file.write(f"Step {step_idx} | "
                        f"P1: {state.p1_pos} HP={state.p1_hp} E={state.p1_eng} | "
                        f"P2: {state.p2_pos} HP={state.p2_hp} E={state.p2_eng}\n")
        
        # 2. Render Map
        self._draw_map(state)
        
        # 3. Actions
        a1_str = self._get_action_name(action_p1)
        a2_str = self._get_action_name(action_p2)
        self.file.write(f"Action: P1={a1_str}, P2={a2_str}\n\n")
        
        # 4. Result if done
        if done:
            self.file.write(f"Game Over! Reward: P1={reward_p1:.1f}, P2={reward_p2:.1f}\n")
            if info:
                self.file.write(f"Info: {info}\n")

    def _draw_map(self, state: GameState):
        top_border = "+" + "---+" * MAP_SIZE
        self.file.write(top_border + "\n")
        
        for y in range(MAP_SIZE - 1, -1, -1):
            row_str = "|"
            for x in range(MAP_SIZE):
                content = "   "
                is_p1 = (x == state.p1_pos[0] and y == state.p1_pos[1])
                is_p2 = (x == state.p2_pos[0] and y == state.p2_pos[1])
                is_wall = (state.walls[y, x] > 0.5)
                
                if is_p1 and is_p2: content = " X "
                elif is_p1: content = " 1 "
                elif is_p2: content = " 2 "
                elif is_wall: content = "###"
                
                row_str += content + "|"
            self.file.write(row_str + "\n")
            self.file.write(top_border + "\n")

    def _get_action_name(self, action_idx):
        if 0 <= action_idx < len(ACTION_NAMES):
            return ACTION_NAMES[action_idx]
        return str(action_idx)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def run_evaluation_episode(agent_p1, agent_p2, env, device, logger=None, max_steps=50):
    """
    Core Game Loop for Evaluation.
    Decoupled from rendering logic.
    
    Args:
        agent_p1: Agent for Player 1
        agent_p2: Agent for Player 2
        env: VectorizedEnv instance
        device: torch device
        logger: Optional ReplayLogger instance
        max_steps: Maximum steps per episode
    """
    obs, mask = env.reset()
    
    # Obs/Mask are numpy arrays [2 * num_envs, dim]
    # For Eval with VectorizedEnv(1), shape is [2, dim]
    obs_t = torch.FloatTensor(obs).to(device)
    mask_t = torch.FloatTensor(mask).to(device)
    
    for step in range(max_steps):
        # Prepare inputs for agents (Batch Size = 1)
        obs_p1 = obs_t[0:1]
        mask_p1 = mask_t[0:1]
        
        obs_p2 = obs_t[1:2]
        mask_p2 = mask_t[1:2]
        
        # Get Actions
        with torch.no_grad():
            act1_t, _ = agent_p1.get_action(obs_p1, mask_p1, deterministic=True)
            act2_t, _ = agent_p2.get_action(obs_p2, mask_p2, deterministic=True)
        
        a1 = act1_t.item()
        a2 = act2_t.item()
        
        # Step Env
        obs_new, reward, done, mask_new, info_list = env.step([a1], [a2])
        
        # Log if logger is provided
        if logger:
            # Parse State from P1's perspective
            # obs[0] is P1's view: (MyPos, EnemyPos, ...)
            state = GameState(obs[0])
            
            info = info_list[0] if info_list else {}
            logger.log_step(step, state, a1, a2, reward[0], reward[1], done[0], info)
            
        if done[0]:
            break
            
        # Update for next step
        obs = obs_new
        mask = mask_new
        obs_t = torch.FloatTensor(obs).to(device)
        mask_t = torch.FloatTensor(mask).to(device)


def render_ascii_game_to_file(agent_p1, agent_p2, config, replay_dir="replays"):
    """
    Main entry point for generating replays.
    Uses the decoupled components to run and log a game.
    """
    # Create a temporary environment specifically for this replay
    env = high_perf_env.VectorizedEnv(1)
    
    logger = AsciiReplayLogger(replay_dir)
    try:
        run_evaluation_episode(agent_p1, agent_p2, env, config.device, logger)
    finally:
        logger.close()
