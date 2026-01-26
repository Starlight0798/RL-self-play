import torch
import numpy as np
import high_perf_env
import time
import os
import sys
from typing import Any, TypedDict

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


class TicTacToeState:
    def __init__(self, obs_vector):
        self.raw = obs_vector
        self.board = np.zeros((3, 3), dtype=int)
        for r in range(3):
            for c in range(3):
                base = (r * 3 + c) * 3
                if obs_vector[base] > 0.5:
                    self.board[r, c] = 0
                elif obs_vector[base + 1] > 0.5:
                    self.board[r, c] = 1
                elif obs_vector[base + 2] > 0.5:
                    self.board[r, c] = 2

    def render(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = []
        for r in range(3):
            lines.append(" ".join(symbols[self.board[r, c]] for c in range(3)))
        return "\n".join(lines)


class GameStateFactory:
    @staticmethod
    def create(game_name: str, obs: np.ndarray, config=None):
        if game_name == "simple_duel":
            return SimpleDuelState(obs, config)
        elif game_name == "tictactoe":
            return TicTacToeState(obs)
        elif game_name in ["connect4", "reversi"]:
            return BoardGameState(obs, game_name)
        else:
            raise ValueError(f"Unknown game: {game_name}")


class SimpleDuelState:
    def __init__(self, obs_vector, config=None):
        self.raw = obs_vector
        if config is None:
            from config import SimpleDuelConfig

            config = SimpleDuelConfig()
        self.config = config

        map_size = config.map_size

        self.p1_pos = (
            self._denorm_pos(obs_vector[0], map_size),
            self._denorm_pos(obs_vector[1], map_size),
        )
        self.p1_hp = self._denorm_hp(obs_vector[4], config.max_hp)
        self.p1_eng = self._denorm_eng(obs_vector[6], config.max_energy)

        self.p2_pos = (
            self._denorm_pos(obs_vector[2], map_size),
            self._denorm_pos(obs_vector[3], map_size),
        )
        self.p2_hp = self._denorm_hp(obs_vector[5], config.max_hp)
        self.p2_eng = self._denorm_eng(obs_vector[7], config.max_energy)

        self.p1_shield = int(round(obs_vector[8] * config.max_shield))
        self.p2_shield = int(round(obs_vector[9] * config.max_shield))
        self.p1_ammo = int(round(obs_vector[10] * config.max_ammo))
        self.p2_ammo = int(round(obs_vector[11] * config.max_ammo))

        self.p1_dodge = obs_vector[12] > 0.5
        self.p2_dodge = obs_vector[13] > 0.5
        self.heal_cd = int(round(obs_vector[14] * 5))
        self.step_progress = obs_vector[15]

        grid_data = obs_vector[16:]
        self.terrain = np.zeros((map_size, map_size), dtype=int)
        self.items = np.zeros((map_size, map_size), dtype=int)

        for y in range(map_size):
            for x in range(map_size):
                idx = y * map_size + x
                val = grid_data[idx]

                if val < 0.1:
                    self.terrain[y, x] = 0
                elif val < 0.4:
                    self.terrain[y, x] = 1
                elif val < 0.6:
                    self.terrain[y, x] = 2
                elif val < 0.9:
                    self.terrain[y, x] = 3
                else:
                    self.terrain[y, x] = 0

                frac = val - int(val * 4) / 4.0
                if frac > 0.05:
                    item_type = int(round(frac / 0.1))
                    if 1 <= item_type <= 4:
                        self.items[y, x] = item_type

    def _denorm_pos(self, val, map_size):
        return int(np.clip(round(val * (map_size - 1)), 0, map_size - 1))

    def _denorm_hp(self, val, max_hp):
        return int(round(val * max_hp))

    def _denorm_eng(self, val, max_energy):
        return int(round(val * max_energy))


def get_game_renderer(game_name: str, config=None):
    if game_name == "simple_duel":

        def render_simple_duel(state: SimpleDuelState):
            dummy = LiveReplayPlayer("P1", "P2", config=config)
            return dummy._render_map(state)

        return render_simple_duel
    elif game_name == "tictactoe":

        def render_tictactoe(state: TicTacToeState):
            return state.render()

        return render_tictactoe
    elif game_name in ["connect4", "reversi"]:

        def render_board_game(state: BoardGameState):
            return state.render()

        return render_board_game
    else:
        raise ValueError(f"Unknown game: {game_name}")


class LiveReplayPlayer:
    def __init__(self, p1_info: str, p2_info: str, delay: float = 0.5, config=None):
        self.p1_info = p1_info
        self.p2_info = p2_info
        self.delay = delay
        self.frames = []
        if config is None:
            from config import SimpleDuelConfig

            config = SimpleDuelConfig()
        self.config = config

    def clear_screen(self):
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()

    def render_frame(
        self,
        step_idx: int,
        state: Any,
        action_p1: int,
        action_p2: int,
        reward_p1: float,
        reward_p2: float,
        done: bool,
        info: dict | None = None,
    ):
        lines = []

        lines.append(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        lines.append(f"{Colors.BOLD}  LIVE BATTLE - Step {step_idx:3d}{Colors.RESET}")
        lines.append(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

        lines.append(
            f"  {Colors.CYAN}P1 [{self.p1_info}]{Colors.RESET} vs {Colors.MAGENTA}P2 [{self.p2_info}]{Colors.RESET}"
        )
        lines.append("")

        if hasattr(state, "p1_hp"):
            hp_bar_p1 = self._hp_bar(state.p1_hp, int(self.config.max_hp))
            eng_bar_p1 = self._eng_bar(state.p1_eng, int(self.config.max_energy))
            shield_str_p1 = f"🛡{state.p1_shield}" if state.p1_shield > 0 else "  "
            ammo_str_p1 = f"💎{state.p1_ammo}"
            dodge_str_p1 = (
                f" {Colors.YELLOW}[DODGE]{Colors.RESET}" if state.p1_dodge else ""
            )

            lines.append(
                f"  {Colors.CYAN}P1{Colors.RESET} {hp_bar_p1} {eng_bar_p1} {shield_str_p1} {ammo_str_p1}{dodge_str_p1}"
            )

            hp_bar_p2 = self._hp_bar(state.p2_hp, int(self.config.max_hp))
            eng_bar_p2 = self._eng_bar(state.p2_eng, int(self.config.max_energy))
            shield_str_p2 = f"🛡{state.p2_shield}" if state.p2_shield > 0 else "  "
            ammo_str_p2 = f"💎{state.p2_ammo}"
            dodge_str_p2 = (
                f" {Colors.YELLOW}[DODGE]{Colors.RESET}" if state.p2_dodge else ""
            )

            lines.append(
                f"  {Colors.MAGENTA}P2{Colors.RESET} {hp_bar_p2} {eng_bar_p2} {shield_str_p2} {ammo_str_p2}{dodge_str_p2}"
            )
            lines.append("")
            lines.append(self._render_map(state))
        else:
            if hasattr(state, "render_colored"):
                lines.append(state.render_colored())
            else:
                lines.append(state.render())
            lines.append("")

        action_names = self.config.action_names
        a1_str = (
            action_names[action_p1]
            if 0 <= action_p1 < len(action_names)
            else str(action_p1)
        )
        a2_str = (
            action_names[action_p2]
            if 0 <= action_p2 < len(action_names)
            else str(action_p2)
        )
        lines.append(
            f"  {Colors.CYAN}P1: {a1_str:8s}{Colors.RESET}  |  {Colors.MAGENTA}P2: {a2_str:8s}{Colors.RESET}"
        )

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

        if done:
            lines.append("")
            lines.append(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

            p1_win = info.get("p1_win", 0) > 0.5 if info else False
            p2_win = info.get("p2_win", 0) > 0.5 if info else False

            if not p1_win and not p2_win and hasattr(state, "p1_hp"):
                if state.p1_hp > state.p2_hp:
                    p1_win = True
                elif state.p2_hp > state.p1_hp:
                    p2_win = True

            if p1_win:
                lines.append(
                    f"  {Colors.CYAN}{Colors.BOLD}>>> P1 WINS! <<<{Colors.RESET}"
                )
            elif p2_win:
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
        filled = "▓" * current
        empty = "░" * (max_val - current)
        return f"E[{Colors.BLUE}{filled}{empty}{Colors.RESET}]"

    def _render_map(self, state: SimpleDuelState) -> str:
        map_size = self.config.map_size
        lines = []
        top_border = "  +" + "--+" * map_size

        lines.append(top_border)

        for y in range(map_size - 1, -1, -1):
            row_str = "  |"
            for x in range(map_size):
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
                elif terrain == 1:
                    content = f"{Colors.WHITE}██{Colors.RESET}"
                elif terrain == 2:
                    content = f"{Colors.BLUE}~~{Colors.RESET}"
                elif terrain == 3:
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
        state: SimpleDuelState,
        action_p1: int,
        action_p2: int,
        reward_p1: float,
        reward_p2: float,
        done: bool,
        info: dict = None,
    ):
        frame = self.render_frame(
            step_idx, state, action_p1, action_p2, reward_p1, reward_p2, done, info
        )
        self.frames.append((frame, done))

    def play(self):
        try:
            for frame, done in self.frames:
                self.clear_screen()
                print(frame)
                if done:
                    break
                time.sleep(self.delay)
            time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n\nReplay stopped by user.")


class AsciiReplayLogger:
    def __init__(
        self,
        output_dir: str = "replays",
        p1_info: str = "Unknown",
        p2_info: str = "Unknown",
        config=None,
    ):
        self.output_dir = output_dir
        self.p1_info = p1_info
        self.p2_info = p2_info
        if config is None:
            from config import SimpleDuelConfig

            config = SimpleDuelConfig()
        self.config = config

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
        state: Any,
        action_p1: int,
        action_p2: int,
        reward_p1: float,
        reward_p2: float,
        done: bool,
        info: dict | None = None,
    ):
        if hasattr(state, "p1_hp"):
            self.file.write(
                f"Step {step_idx} | "
                f"P1: {state.p1_pos} HP={state.p1_hp} E={state.p1_eng} Sh={state.p1_shield} Am={state.p1_ammo} | "
                f"P2: {state.p2_pos} HP={state.p2_hp} E={state.p2_eng} Sh={state.p2_shield} Am={state.p2_ammo}\n"
            )

            if state.p1_dodge:
                self.file.write("  [P1 Dodge Active]\n")
            if state.p2_dodge:
                self.file.write("  [P2 Dodge Active]\n")
        else:
            self.file.write(f"Step {step_idx} | Board Game\n")

        self._draw_map(state)

        a1_str = self._get_action_name(action_p1)
        a2_str = self._get_action_name(action_p2)
        self.file.write(f"Action: P1={a1_str}, P2={a2_str}\n\n")

        if done:
            self.file.write("-" * 60 + "\n")
            self.file.write(
                f"Game Over! Reward: P1={reward_p1:.1f}, P2={reward_p2:.1f}\n"
            )

            p1_win = info.get("p1_win", 0) > 0.5 if info else False
            p2_win = info.get("p2_win", 0) > 0.5 if info else False

            if not p1_win and not p2_win and hasattr(state, "p1_hp"):
                if state.p1_hp > state.p2_hp:
                    p1_win = True
                elif state.p2_hp > state.p1_hp:
                    p2_win = True

            if p1_win:
                self.file.write(f"Winner: P1 ({self.p1_info})\n")
            elif p2_win:
                self.file.write(f"Winner: P2 ({self.p2_info})\n")
            else:
                self.file.write("Result: Draw\n")
            if info:
                self.file.write(f"Info: {info}\n")

    def _draw_map(self, state: Any):
        if hasattr(state, "p1_hp"):
            map_size = self.config.map_size
            top_border = "+" + "--+" * map_size
            self.file.write(top_border + "\n")

            for y in range(map_size - 1, -1, -1):
                row_str = "|"
                for x in range(map_size):
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
                    elif terrain == 1:
                        content = "##"
                    elif terrain == 2:
                        content = "~~"
                    elif terrain == 3:
                        content = "^^"
                    elif item > 0:
                        content = ITEM_CHARS.get(item, "?") + " "
                    else:
                        content = ". "

                    row_str += content + "|"
                self.file.write(row_str + "\n")
                self.file.write(top_border + "\n")

            self.file.write(
                "Legend: 1=P1, 2=P2, ##=Wall, ~~=Water, ^^=HighGround, H=Health, E=Energy, A=Ammo, S=Shield\n"
            )
        else:
            self.file.write(state.render() + "\n")

    def _get_action_name(self, action_idx: int) -> str:
        action_names = self.config.action_names
        if 0 <= action_idx < len(action_names):
            return action_names[action_idx]
        return str(action_idx)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def run_evaluation_episode(
    agent_p1,
    agent_p2,
    env,
    device,
    game_name: str = "simple_duel",
    config=None,
    logger=None,
    live_player=None,
    max_steps: int = 60,
):
    obs, mask = env.reset()

    obs_t = torch.FloatTensor(obs).to(device)
    mask_t = torch.FloatTensor(mask).to(device)

    for step in range(max_steps):
        obs_p1 = obs_t[0:1]
        mask_p1 = mask_t[0:1]

        obs_p2 = obs_t[1:2]
        mask_p2 = mask_t[1:2]

        with torch.no_grad():
            act1_t, _ = agent_p1.get_action(obs_p1, mask_p1, deterministic=True)
            act2_t, _ = agent_p2.get_action(obs_p2, mask_p2, deterministic=True)

        a1 = act1_t.item()
        a2 = act2_t.item()

        obs_new, reward, done, mask_new, info_list = env.step([a1], [a2])

        state = GameStateFactory.create(game_name, obs[0], config=config)
        info = info_list[0] if info_list else {}

        if logger:
            logger.log_step(step, state, a1, a2, reward[0], reward[1], done[0], info)

        if live_player:
            live_player.add_frame(
                step, state, a1, a2, reward[0], reward[1], done[0], info
            )

        if done[0]:
            break

        obs = obs_new
        mask = mask_new
        obs_t = torch.FloatTensor(obs).to(device)
        mask_t = torch.FloatTensor(mask).to(device)


def render_ascii_game_to_file(agent_p1, agent_p2, config, replay_dir="replays"):
    from config import GameConfig

    game_name = config.game if hasattr(config, "game") else "simple_duel"
    game_config = (
        config.game_config
        if hasattr(config, "game_config")
        else GameConfig.from_name(game_name)
    )

    env = high_perf_env.create_env(game_name, 1)

    p1_info = get_agent_info(agent_p1)
    p2_info = get_agent_info(agent_p2)

    logger = AsciiReplayLogger(replay_dir, p1_info, p2_info, config=game_config)
    try:
        run_evaluation_episode(
            agent_p1,
            agent_p2,
            env,
            config.device,
            game_name=game_name,
            config=game_config,
            logger=logger,
        )
    finally:
        logger.close()


def play_live_game(agent_p1, agent_p2, config, delay: float = 0.3):
    from config import GameConfig

    game_name = config.game if hasattr(config, "game") else "simple_duel"
    game_config = (
        config.game_config
        if hasattr(config, "game_config")
        else GameConfig.from_name(game_name)
    )

    env = high_perf_env.create_env(game_name, 1)

    p1_info = get_agent_info(agent_p1)
    p2_info = get_agent_info(agent_p2)

    live_player = LiveReplayPlayer(p1_info, p2_info, delay, config=game_config)

    print(f"\nPreparing battle: {p1_info} vs {p2_info}")
    print("Recording game...")

    run_evaluation_episode(
        agent_p1,
        agent_p2,
        env,
        config.device,
        game_name=game_name,
        config=game_config,
        live_player=live_player,
    )

    print(f"Recorded {len(live_player.frames)} frames. Starting playback...\n")
    time.sleep(1)

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
