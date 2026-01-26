import torch
import numpy as np
import high_perf_env
import time
import os
import sys

# --- Configuration Constants ---
# These should match the environment settings in src/lib.rs
MAP_SIZE = 12
MAX_HP = 4.0
MAX_ENERGY = 7.0
MAX_SHIELD = 2.0
MAX_AMMO = 6.0

ACTION_NAMES = [
    "Stay",
    "Up",
    "Down",
    "Left",
    "Right",
    "Attack",
    "Shoot",
    "Dodge",
    "Shield",
    "Dash",
    "AOE",
    "Heal",
    "Reload",
]

# 地形字符映射
TERRAIN_CHARS = {
    0: ".",  # Empty
    1: "#",  # Wall
    2: "~",  # Water
    3: "^",  # High Ground
}

# 道具字符映射
ITEM_CHARS = {
    1: "H",  # Health
    2: "E",  # Energy
    3: "A",  # Ammo
    4: "S",  # Shield
}


# ANSI 颜色代码
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_BLUE = "\033[44m"
    BG_RED = "\033[41m"


def get_agent_info(agent) -> str:
    """获取智能体的描述信息"""
    class_name = agent.__class__.__name__
    if class_name == "RuleBasedAgent":
        return "Rule-Based"
    elif class_name == "PPO":
        # 尝试获取 checkpoint 信息
        checkpoint = getattr(agent, "checkpoint_path", None)
        step = getattr(agent, "training_step", None)
        if checkpoint:
            return f"PPO ({checkpoint})"
        elif step:
            return f"PPO (step {step})"
        else:
            return "PPO (current)"
    else:
        return class_name


class GameState:
    """
    Decoupled State Parser.
    Responsible for interpreting the raw observation vector into a structured, readable format.
    Assumes P1 perspective observation structure.
    """

    def __init__(self, obs_vector):
        # obs_vector: np.array shape (160,)
        self.raw = obs_vector

        # Parse P1 (My) info - [0-7]: positions and base stats
        self.p1_pos = (self._denorm_pos(obs_vector[0]), self._denorm_pos(obs_vector[1]))
        self.p1_hp = self._denorm_hp(obs_vector[4])
        self.p1_eng = self._denorm_eng(obs_vector[6])

        # Parse P2 (Enemy) info
        self.p2_pos = (self._denorm_pos(obs_vector[2]), self._denorm_pos(obs_vector[3]))
        self.p2_hp = self._denorm_hp(obs_vector[5])
        self.p2_eng = self._denorm_eng(obs_vector[7])

        # [8-11]: Shield and Ammo
        self.p1_shield = int(round(obs_vector[8] * MAX_SHIELD))
        self.p2_shield = int(round(obs_vector[9] * MAX_SHIELD))
        self.p1_ammo = int(round(obs_vector[10] * MAX_AMMO))
        self.p2_ammo = int(round(obs_vector[11] * MAX_AMMO))

        # [12-15]: Dodge, heal cooldown, step progress
        self.p1_dodge = obs_vector[12] > 0.5
        self.p2_dodge = obs_vector[13] > 0.5
        self.heal_cd = int(round(obs_vector[14] * 5))  # HEAL_COOLDOWN = 5
        self.step_progress = obs_vector[15]

        # Parse Terrain Grid
        # Grid is at index 16 to 159 (144 values for 12x12)
        # Values encode terrain + item info
        grid_data = obs_vector[16:]
        self.terrain = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)
        self.items = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)

        for y in range(MAP_SIZE):
            for x in range(MAP_SIZE):
                idx = y * MAP_SIZE + x
                val = grid_data[idx]

                # Decode terrain (main value)
                if val < 0.1:
                    self.terrain[y, x] = 0  # Empty
                elif val < 0.4:
                    self.terrain[y, x] = 1  # Wall (0.25)
                elif val < 0.6:
                    self.terrain[y, x] = 2  # Water (0.5)
                elif val < 0.9:
                    self.terrain[y, x] = 3  # High Ground (0.75)
                else:
                    self.terrain[y, x] = 0

                # Decode item (fractional part > 0.05 indicates item)
                frac = val - int(val * 4) / 4.0  # Get fractional above terrain base
                if frac > 0.05:
                    item_type = int(round(frac / 0.1))
                    if 1 <= item_type <= 4:
                        self.items[y, x] = item_type

    def _denorm_pos(self, val):
        return int(np.clip(round(val * (MAP_SIZE - 1)), 0, MAP_SIZE - 1))

    def _denorm_hp(self, val):
        return int(round(val * MAX_HP))

    def _denorm_eng(self, val):
        return int(round(val * MAX_ENERGY))


class LiveReplayPlayer:
    """
    在终端实时播放游戏回放
    """

    def __init__(self, p1_info: str, p2_info: str, delay: float = 0.5):
        self.p1_info = p1_info
        self.p2_info = p2_info
        self.delay = delay
        self.frames = []  # 存储所有帧用于回放

    def clear_screen(self):
        """清屏"""
        # 使用 ANSI 转义序列移动光标到顶部
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()

    def render_frame(
        self,
        step_idx: int,
        state: GameState,
        action_p1: int,
        action_p2: int,
        reward_p1: float,
        reward_p2: float,
        done: bool,
        info: dict = None,
    ):
        """渲染单帧到终端"""
        lines = []

        # 标题栏
        lines.append(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        lines.append(
            f"{Colors.BOLD}  LIVE BATTLE - Step {step_idx:3d}/60{Colors.RESET}"
        )
        lines.append(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

        # 对战信息
        lines.append(
            f"  {Colors.CYAN}P1 [{self.p1_info}]{Colors.RESET} vs {Colors.MAGENTA}P2 [{self.p2_info}]{Colors.RESET}"
        )
        lines.append("")

        # 玩家状态 - P1
        hp_bar_p1 = self._hp_bar(state.p1_hp, int(MAX_HP))
        eng_bar_p1 = self._eng_bar(state.p1_eng, int(MAX_ENERGY))
        shield_str_p1 = f"🛡{state.p1_shield}" if state.p1_shield > 0 else "  "
        ammo_str_p1 = f"💎{state.p1_ammo}"
        dodge_str_p1 = (
            f" {Colors.YELLOW}[DODGE]{Colors.RESET}" if state.p1_dodge else ""
        )

        lines.append(
            f"  {Colors.CYAN}P1{Colors.RESET} {hp_bar_p1} {eng_bar_p1} {shield_str_p1} {ammo_str_p1}{dodge_str_p1}"
        )

        # 玩家状态 - P2
        hp_bar_p2 = self._hp_bar(state.p2_hp, int(MAX_HP))
        eng_bar_p2 = self._eng_bar(state.p2_eng, int(MAX_ENERGY))
        shield_str_p2 = f"🛡{state.p2_shield}" if state.p2_shield > 0 else "  "
        ammo_str_p2 = f"💎{state.p2_ammo}"
        dodge_str_p2 = (
            f" {Colors.YELLOW}[DODGE]{Colors.RESET}" if state.p2_dodge else ""
        )

        lines.append(
            f"  {Colors.MAGENTA}P2{Colors.RESET} {hp_bar_p2} {eng_bar_p2} {shield_str_p2} {ammo_str_p2}{dodge_str_p2}"
        )
        lines.append("")

        # 地图
        lines.append(self._render_map(state))

        # 动作
        a1_str = (
            ACTION_NAMES[action_p1]
            if 0 <= action_p1 < len(ACTION_NAMES)
            else str(action_p1)
        )
        a2_str = (
            ACTION_NAMES[action_p2]
            if 0 <= action_p2 < len(ACTION_NAMES)
            else str(action_p2)
        )
        lines.append(
            f"  {Colors.CYAN}P1: {a1_str:8s}{Colors.RESET}  |  {Colors.MAGENTA}P2: {a2_str:8s}{Colors.RESET}"
        )

        # 奖励变化
        if reward_p1 != 0 or reward_p2 != 0:
            r1_color = (
                Colors.GREEN
                if reward_p1 > 0
                else Colors.RED
                if reward_p1 < 0
                else Colors.WHITE
            )
            r2_color = (
                Colors.GREEN
                if reward_p2 > 0
                else Colors.RED
                if reward_p2 < 0
                else Colors.WHITE
            )
            lines.append(
                f"  Reward: {r1_color}P1 {reward_p1:+.1f}{Colors.RESET}  |  {r2_color}P2 {reward_p2:+.1f}{Colors.RESET}"
            )

        # 结束信息
        if done:
            lines.append("")
            lines.append(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
            if state.p1_hp > state.p2_hp:
                lines.append(
                    f"  {Colors.CYAN}{Colors.BOLD}>>> P1 WINS! <<<{Colors.RESET}"
                )
            elif state.p2_hp > state.p1_hp:
                lines.append(
                    f"  {Colors.MAGENTA}{Colors.BOLD}>>> P2 WINS! <<<{Colors.RESET}"
                )
            else:
                lines.append(
                    f"  {Colors.YELLOW}{Colors.BOLD}>>> DRAW! <<<{Colors.RESET}"
                )
            lines.append(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

        lines.append("")
        lines.append(f"  {Colors.WHITE}[Press Ctrl+C to stop]{Colors.RESET}")

        return "\n".join(lines)

    def _hp_bar(self, current: int, max_val: int) -> str:
        """生成 HP 进度条"""
        filled = "█" * current
        empty = "░" * (max_val - current)
        color = (
            Colors.GREEN
            if current > 2
            else Colors.YELLOW
            if current > 1
            else Colors.RED
        )
        return f"HP[{color}{filled}{empty}{Colors.RESET}]"

    def _eng_bar(self, current: int, max_val: int) -> str:
        """生成能量进度条"""
        filled = "▓" * current
        empty = "░" * (max_val - current)
        return f"E[{Colors.BLUE}{filled}{empty}{Colors.RESET}]"

    def _render_map(self, state: GameState) -> str:
        """渲染地图"""
        lines = []
        top_border = "  +" + "--+" * MAP_SIZE

        lines.append(top_border)

        for y in range(MAP_SIZE - 1, -1, -1):
            row_str = "  |"
            for x in range(MAP_SIZE):
                is_p1 = x == state.p1_pos[0] and y == state.p1_pos[1]
                is_p2 = x == state.p2_pos[0] and y == state.p2_pos[1]
                terrain = state.terrain[y, x]
                item = state.items[y, x]

                if is_p1 and is_p2:
                    content = f"{Colors.RED}XX{Colors.RESET}"
                elif is_p1:
                    content = f"{Colors.CYAN}P1{Colors.RESET}"
                elif is_p2:
                    content = f"{Colors.MAGENTA}P2{Colors.RESET}"
                elif terrain == 1:  # Wall
                    content = f"{Colors.WHITE}██{Colors.RESET}"
                elif terrain == 2:  # Water
                    content = f"{Colors.BLUE}~~{Colors.RESET}"
                elif terrain == 3:  # High Ground
                    content = f"{Colors.YELLOW}^^{Colors.RESET}"
                elif item > 0:
                    item_char = ITEM_CHARS.get(item, "?")
                    content = f"{Colors.GREEN}{item_char} {Colors.RESET}"
                else:
                    content = ". "

                row_str += content + "|"
            lines.append(row_str)
            lines.append(top_border)

        return "\n".join(lines)

    def add_frame(
        self,
        step_idx: int,
        state: GameState,
        action_p1: int,
        action_p2: int,
        reward_p1: float,
        reward_p2: float,
        done: bool,
        info: dict = None,
    ):
        """添加一帧"""
        frame = self.render_frame(
            step_idx, state, action_p1, action_p2, reward_p1, reward_p2, done, info
        )
        self.frames.append((frame, done))

    def play(self):
        """播放所有帧"""
        try:
            for frame, done in self.frames:
                self.clear_screen()
                print(frame)
                if done:
                    break
                time.sleep(self.delay)
            # 结束后暂停显示结果
            time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n\nReplay stopped by user.")


class AsciiReplayLogger:
    """
    Handles the visualization and file I/O for game replays.
    Decoupled from the game loop and environment logic.
    """

    def __init__(
        self,
        output_dir: str = "replays",
        p1_info: str = "Unknown",
        p2_info: str = "Unknown",
    ):
        self.output_dir = output_dir
        self.p1_info = p1_info
        self.p2_info = p2_info

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(output_dir, f"replay_{timestamp}.txt")
        self.file = open(self.filename, "w", encoding="utf-8")

        self._write_header(timestamp)

    def _write_header(self, timestamp: str):
        self.file.write("\n" + "=" * 60 + "\n")
        self.file.write(f" >>> BATTLE REPLAY @ {timestamp} <<<\n")
        self.file.write("=" * 60 + "\n\n")
        self.file.write(f"P1: {self.p1_info}\n")
        self.file.write(f"P2: {self.p2_info}\n")
        self.file.write("-" * 60 + "\n\n")

    def log_step(
        self,
        step_idx: int,
        state: GameState,
        action_p1: int,
        action_p2: int,
        reward_p1: float,
        reward_p2: float,
        done: bool,
        info: dict = None,
    ):
        """
        Logs a single step of the game.
        """
        # 1. State Info
        self.file.write(
            f"Step {step_idx} | "
            f"P1: {state.p1_pos} HP={state.p1_hp} E={state.p1_eng} Sh={state.p1_shield} Am={state.p1_ammo} | "
            f"P2: {state.p2_pos} HP={state.p2_hp} E={state.p2_eng} Sh={state.p2_shield} Am={state.p2_ammo}\n"
        )

        if state.p1_dodge:
            self.file.write("  [P1 Dodge Active]\n")
        if state.p2_dodge:
            self.file.write("  [P2 Dodge Active]\n")

        # 2. Render Map
        self._draw_map(state)

        # 3. Actions
        a1_str = self._get_action_name(action_p1)
        a2_str = self._get_action_name(action_p2)
        self.file.write(f"Action: P1={a1_str}, P2={a2_str}\n\n")

        # 4. Result if done
        if done:
            self.file.write("-" * 60 + "\n")
            self.file.write(
                f"Game Over! Reward: P1={reward_p1:.1f}, P2={reward_p2:.1f}\n"
            )
            if state.p1_hp > state.p2_hp:
                self.file.write(f"Winner: P1 ({self.p1_info})\n")
            elif state.p2_hp > state.p1_hp:
                self.file.write(f"Winner: P2 ({self.p2_info})\n")
            else:
                self.file.write("Result: Draw\n")
            if info:
                self.file.write(f"Info: {info}\n")

    def _draw_map(self, state: GameState):
        # Use narrower cells for 12x12 map
        top_border = "+" + "--+" * MAP_SIZE
        self.file.write(top_border + "\n")

        for y in range(MAP_SIZE - 1, -1, -1):
            row_str = "|"
            for x in range(MAP_SIZE):
                content = "  "
                is_p1 = x == state.p1_pos[0] and y == state.p1_pos[1]
                is_p2 = x == state.p2_pos[0] and y == state.p2_pos[1]
                terrain = state.terrain[y, x]
                item = state.items[y, x]

                if is_p1 and is_p2:
                    content = "X "
                elif is_p1:
                    content = "1 "
                elif is_p2:
                    content = "2 "
                elif terrain == 1:  # Wall
                    content = "##"
                elif terrain == 2:  # Water
                    content = "~~"
                elif terrain == 3:  # High Ground
                    content = "^^"
                elif item > 0:
                    content = ITEM_CHARS.get(item, "?") + " "
                else:
                    content = ". "

                row_str += content + "|"
            self.file.write(row_str + "\n")
            self.file.write(top_border + "\n")

        # Legend
        self.file.write(
            "Legend: 1=P1, 2=P2, ##=Wall, ~~=Water, ^^=HighGround, H=Health, E=Energy, A=Ammo, S=Shield\n"
        )

    def _get_action_name(self, action_idx: int) -> str:
        if 0 <= action_idx < len(ACTION_NAMES):
            return ACTION_NAMES[action_idx]
        return str(action_idx)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def run_evaluation_episode(
    agent_p1, agent_p2, env, device, logger=None, live_player=None, max_steps=60
):
    """
    Core Game Loop for Evaluation.
    Decoupled from rendering logic.

    Args:
        agent_p1: Agent for Player 1
        agent_p2: Agent for Player 2
        env: VectorizedEnv instance
        device: torch device
        logger: Optional AsciiReplayLogger instance for file output
        live_player: Optional LiveReplayPlayer instance for terminal playback
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

        # Parse State
        state = GameState(obs[0])
        info = info_list[0] if info_list else {}

        # Log to file if logger is provided
        if logger:
            logger.log_step(step, state, a1, a2, reward[0], reward[1], done[0], info)

        # Add frame for live playback
        if live_player:
            live_player.add_frame(
                step, state, a1, a2, reward[0], reward[1], done[0], info
            )

        if done[0]:
            break

        # Update for next step
        obs = obs_new
        mask = mask_new
        obs_t = torch.FloatTensor(obs).to(device)
        mask_t = torch.FloatTensor(mask).to(device)


def render_ascii_game_to_file(agent_p1, agent_p2, config, replay_dir="replays"):
    """
    Main entry point for generating replays (file only).
    Uses the decoupled components to run and log a game.
    """
    # Create a temporary environment specifically for this replay
    env = high_perf_env.VectorizedEnv(1)

    # Get agent info
    p1_info = get_agent_info(agent_p1)
    p2_info = get_agent_info(agent_p2)

    logger = AsciiReplayLogger(replay_dir, p1_info, p2_info)
    try:
        run_evaluation_episode(agent_p1, agent_p2, env, config.device, logger=logger)
    finally:
        logger.close()


def play_live_game(agent_p1, agent_p2, device, delay: float = 0.3):
    """
    在终端实时播放一场对战

    Args:
        agent_p1: Player 1 的智能体
        agent_p2: Player 2 的智能体
        device: torch device
        delay: 每帧之间的延迟（秒）
    """
    env = high_perf_env.VectorizedEnv(1)

    p1_info = get_agent_info(agent_p1)
    p2_info = get_agent_info(agent_p2)

    live_player = LiveReplayPlayer(p1_info, p2_info, delay)

    print(f"\nPreparing battle: {p1_info} vs {p2_info}")
    print("Recording game...")

    # 先录制整场比赛
    run_evaluation_episode(agent_p1, agent_p2, env, device, live_player=live_player)

    print(f"Recorded {len(live_player.frames)} frames. Starting playback...\n")
    time.sleep(1)

    # 播放
    live_player.play()


def watch_replay(replay_file: str, delay: float = 0.3):
    """
    从文件加载并播放回放（简化版，直接逐步显示文件内容）

    注意：这个函数只是逐步显示文本文件，不是真正的动画回放。
    要获得动画效果，请使用 play_live_game() 函数。
    """
    if not os.path.exists(replay_file):
        print(f"Error: Replay file not found: {replay_file}")
        return

    with open(replay_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 按 Step 分割
    steps = content.split("Step ")

    print(steps[0])  # Header
    time.sleep(1)

    for step_content in steps[1:]:
        # 清屏
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()

        print(steps[0])  # 保留 header
        print("Step " + step_content.split("\n\n")[0] + "\n")

        # 找到地图部分并显示
        lines = step_content.split("\n")
        in_map = False
        for line in lines:
            if line.startswith("+--"):
                in_map = True
            if in_map:
                print(line)
            if "Legend:" in line:
                in_map = False
                break

        # 显示动作
        for line in lines:
            if line.startswith("Action:"):
                print("\n" + line)
                break

        time.sleep(delay)

        # 检查是否结束
        if "Game Over" in step_content:
            print("\n" + "-" * 60)
            for line in lines:
                if "Game Over" in line or "Winner" in line or "Draw" in line:
                    print(line)
            break

    print("\nReplay finished.")


from typing import TypedDict


class _BoardGameConfig(TypedDict):
    rows: int
    cols: int
    obs_dim: int
    channels: int
    symbols: dict[str, str]
    name: str


BOARD_GAME_CONFIG: dict[str, _BoardGameConfig] = {
    "connect4": {
        "rows": 6,
        "cols": 7,
        "obs_dim": 126,
        "channels": 3,
        "symbols": {"empty": ".", "self": "X", "opponent": "O"},
        "name": "Connect4",
    },
    "reversi": {
        "rows": 8,
        "cols": 8,
        "obs_dim": 192,
        "channels": 3,
        "symbols": {"empty": ".", "self": "B", "opponent": "W"},
        "name": "Reversi",
    },
}


class BoardGameState:
    """
    State parser for board games (Connect4, Reversi).
    Obs layout: obs[(row * cols + col) * 3 + channel] where channels are [empty, self, opponent].
    """

    def __init__(self, obs_vector, game_name: str):
        if game_name not in BOARD_GAME_CONFIG:
            raise ValueError(
                f"Unknown game: {game_name}. Supported: {list(BOARD_GAME_CONFIG.keys())}"
            )

        config = BOARD_GAME_CONFIG[game_name]
        self.game_name = game_name
        self.rows = int(config["rows"])
        self.cols = int(config["cols"])
        self.symbols = dict(config["symbols"])
        self.raw = obs_vector

        expected_dim = int(config["obs_dim"])
        if len(obs_vector) != expected_dim:
            raise ValueError(
                f"Expected obs_dim={expected_dim} for {game_name}, got {len(obs_vector)}"
            )

        self.board = np.zeros((self.rows, self.cols), dtype=int)

        for row in range(self.rows):
            for col in range(self.cols):
                base = (row * self.cols + col) * 3
                if obs_vector[base] > 0.5:
                    self.board[row, col] = 0
                elif obs_vector[base + 1] > 0.5:
                    self.board[row, col] = 1
                elif obs_vector[base + 2] > 0.5:
                    self.board[row, col] = 2
                else:
                    self.board[row, col] = 0

    def render(self) -> str:
        lines = []

        if self.game_name == "connect4":
            header = "  " + " ".join(str(i) for i in range(self.cols))
            lines.append(header)
        elif self.game_name == "reversi":
            header = "  " + " ".join("abcdefgh"[: self.cols])
            lines.append(header)

        if self.game_name == "connect4":
            for row in range(self.rows - 1, -1, -1):
                row_str = f"{row} "
                for col in range(self.cols):
                    cell = self.board[row, col]
                    if cell == 0:
                        row_str += self.symbols["empty"]
                    elif cell == 1:
                        row_str += self.symbols["self"]
                    else:
                        row_str += self.symbols["opponent"]
                    row_str += " "
                lines.append(row_str.rstrip())
        else:
            for row in range(self.rows):
                row_str = f"{row + 1} "
                for col in range(self.cols):
                    cell = self.board[row, col]
                    if cell == 0:
                        row_str += self.symbols["empty"]
                    elif cell == 1:
                        row_str += self.symbols["self"]
                    else:
                        row_str += self.symbols["opponent"]
                    row_str += " "
                lines.append(row_str.rstrip())

        return "\n".join(lines)

    def render_colored(self) -> str:
        lines = []

        if self.game_name == "connect4":
            header = "  " + " ".join(str(i) for i in range(self.cols))
            lines.append(f"{Colors.WHITE}{header}{Colors.RESET}")
        elif self.game_name == "reversi":
            header = "  " + " ".join("abcdefgh"[: self.cols])
            lines.append(f"{Colors.WHITE}{header}{Colors.RESET}")

        if self.game_name == "connect4":
            for row in range(self.rows - 1, -1, -1):
                row_str = f"{Colors.WHITE}{row}{Colors.RESET} "
                for col in range(self.cols):
                    cell = self.board[row, col]
                    if cell == 0:
                        row_str += (
                            f"{Colors.WHITE}{self.symbols['empty']}{Colors.RESET}"
                        )
                    elif cell == 1:
                        row_str += f"{Colors.CYAN}{self.symbols['self']}{Colors.RESET}"
                    else:
                        row_str += (
                            f"{Colors.MAGENTA}{self.symbols['opponent']}{Colors.RESET}"
                        )
                    row_str += " "
                lines.append(row_str.rstrip())
        else:
            for row in range(self.rows):
                row_str = f"{Colors.WHITE}{row + 1}{Colors.RESET} "
                for col in range(self.cols):
                    cell = self.board[row, col]
                    if cell == 0:
                        row_str += (
                            f"{Colors.WHITE}{self.symbols['empty']}{Colors.RESET}"
                        )
                    elif cell == 1:
                        row_str += f"{Colors.CYAN}{self.symbols['self']}{Colors.RESET}"
                    else:
                        row_str += (
                            f"{Colors.MAGENTA}{self.symbols['opponent']}{Colors.RESET}"
                        )
                    row_str += " "
                lines.append(row_str.rstrip())

        return "\n".join(lines)

    def count_pieces(self) -> tuple:
        self_count = np.sum(self.board == 1)
        opponent_count = np.sum(self.board == 2)
        return int(self_count), int(opponent_count)


def render_board_game(obs, game_name: str, colored: bool = False) -> str:
    obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
    state = BoardGameState(obs_array, game_name)

    if colored:
        return state.render_colored()
    return state.render()
