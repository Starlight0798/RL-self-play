#!/usr/bin/env python3
"""
观看对战回放的脚本

用法:
    # 观看 PPO vs PPO 实时对战
    python watch.py

    # 观看 PPO vs Rule 实时对战
    python watch.py --p2 rule

    # 加载已训练的模型
    python watch.py --model ppo_final.pth

    # 调整播放速度 (默认 0.3 秒/帧)
    python watch.py --delay 0.5

    # 播放已有的回放文件
    python watch.py --replay replays/replay_20240101_120000.txt
"""

import argparse
import torch
import os


def main():
    parser = argparse.ArgumentParser(description="观看对战回放")
    parser.add_argument(
        "--model", type=str, default=None, help="PPO 模型路径 (如 ppo_final.pth)"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="simple_duel",
        help="游戏名称 (如 simple_duel, tictactoe)",
    )
    parser.add_argument(
        "--p1", type=str, default="ppo", choices=["ppo", "rule"], help="P1 类型"
    )

    parser.add_argument(
        "--p2", type=str, default="ppo", choices=["ppo", "rule"], help="P2 类型"
    )
    parser.add_argument("--delay", type=float, default=0.3, help="每帧延迟 (秒)")
    parser.add_argument("--replay", type=str, default=None, help="回放文件路径")
    parser.add_argument("--save", action="store_true", help="同时保存到文件")

    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载游戏插件
    from games.registry import get_game

    game = get_game(args.game)
    ConfigClass = game.config
    assert ConfigClass is not None, f"Could not load config for game {args.game}"
    game_config = ConfigClass()

    # 如果指定了回放文件，直接播放
    if args.replay:
        from utils import watch_replay

        watch_replay(args.replay, args.delay, game_name=args.game)
        return

    # 创建智能体
    from agents.rule_based import RuleBasedAgent
    from algorithms.ppo import PPO
    from config import Config

    config = Config()

    def create_agent(agent_type: str, model_path: str | None = None):
        if agent_type == "rule":
            return RuleBasedAgent(device=str(device))
        else:
            agent = PPO(
                config, obs_dim=game_config.obs_dim, action_dim=game_config.action_dim
            )
            if model_path and os.path.exists(model_path):
                agent.load(model_path)
                agent.checkpoint_path = os.path.basename(model_path)
                print(f"Loaded model: {model_path}")
            return agent

    # 创建 P1 和 P2
    agent_p1 = create_agent(args.p1, args.model if args.p1 == "ppo" else None)
    agent_p2 = create_agent(args.p2, args.model if args.p2 == "ppo" else None)

    # 显示对战信息
    from utils import get_agent_info

    p1_info = get_agent_info(agent_p1)
    p2_info = get_agent_info(agent_p2)
    print(f"\n{'=' * 50}")
    print(f"  Battle ({args.game}): {p1_info} vs {p2_info}")
    print(f"{'=' * 50}\n")

    # 是否同时保存文件
    if args.save:
        from utils import render_ascii_game_to_file

        render_ascii_game_to_file(agent_p1, agent_p2, config, game_name=args.game)
        print("Replay saved to replays/ directory")

    # 播放实时对战
    from utils import play_live_game

    play_live_game(agent_p1, agent_p2, device, delay=args.delay, game_name=args.game)


if __name__ == "__main__":
    main()
