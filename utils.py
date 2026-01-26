import torch
import numpy as np
import high_perf_env
import time
import os
import sys
from games.registry import get_game


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


def run_evaluation_episode(
    agent_p1,
    agent_p2,
    env,
    device,
    game_name: str,
    logger=None,
    live_player=None,
    max_steps=100,
):
    """
    Core Game Loop for Evaluation (Generic).
    """
    game = get_game(game_name)
    GameState = game.parser

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

        state = GameState(obs[0])
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


def render_ascii_game_to_file(
    agent_p1, agent_p2, config, game_name: str, replay_dir="replays"
):
    """
    Main entry point for generating replays (file only).
    Uses the decoupled components to run and log a game.
    """
    # Create a temporary environment specifically for this replay
    env = high_perf_env.create_env(game_name, 1)

    # Get agent info
    p1_info = get_agent_info(agent_p1)
    p2_info = get_agent_info(agent_p2)

    game = get_game(game_name)
    AsciiReplayLogger = game.renderer["AsciiReplayLogger"]

    logger = AsciiReplayLogger(replay_dir, p1_info, p2_info)
    try:
        run_evaluation_episode(
            agent_p1, agent_p2, env, config.device, game_name, logger=logger
        )
    finally:
        logger.close()


def play_live_game(agent_p1, agent_p2, device, game_name: str, delay: float = 0.3):
    """
    在终端实时播放一场对战
    """
    env = high_perf_env.create_env(game_name, 1)

    p1_info = get_agent_info(agent_p1)
    p2_info = get_agent_info(agent_p2)

    game = get_game(game_name)
    LiveReplayPlayer = game.renderer["LiveReplayPlayer"]

    live_player = LiveReplayPlayer(p1_info, p2_info, delay)

    print(f"\nPreparing battle ({game_name}): {p1_info} vs {p2_info}")
    print("Recording game...")

    # 先录制整场比赛
    run_evaluation_episode(
        agent_p1, agent_p2, env, device, game_name, live_player=live_player
    )

    print(f"Recorded {len(live_player.frames)} frames. Starting playback...\n")
    time.sleep(1)

    # 播放
    live_player.play()


def watch_replay(replay_file: str, delay: float = 0.3, game_name: str = "simple_duel"):
    """
    从文件加载并播放回放（从游戏插件中动态获取渲染逻辑）
    """
    if not os.path.exists(replay_file):
        print(f"Error: Replay file not found: {replay_file}")
        return

    # TODO: 目前 watch_replay 的实现逻辑还是比较 hardcoded 在 utils.py 里处理字符串。
    # 如果要完全 de-hardcode，回放文件的格式也应该由插件定义。
    # 这里为了兼容性，暂时保留基本逻辑，但尽量减少 hardcoded 常数。

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
            if line.startswith("+--") or line.startswith("+-"):
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
