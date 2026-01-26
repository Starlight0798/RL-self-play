import torch
import numpy as np
import time
import os
import argparse
from pathlib import Path
from collections import deque

from config import Config, FastConfig, LongConfig
from envs import create_env, get_game_info
from algorithms.registry import get_algorithm
from agents.rule_based import RuleBasedAgent
from utils import render_ascii_game_to_file


def train(args, config=None):
    if config is None:
        config = Config()

    # Get game info
    game_info = get_game_info(args.game)
    obs_dim = game_info.obs_dim
    action_dim = game_info.action_dim

    # 初始化环境
    env = create_env(args.game, config.num_envs)

    # 初始化算法
    from algorithms.base import BaseAlgorithm
    from games.registry import get_game

    # Get game plugin and shaper
    game_plugin = get_game(args.game)

    # Select model
    if args.model:
        model_name = args.model
    else:
        model_name = getattr(game_plugin.config, "default_model_name", "simple_mlp")

    reward_shaper = None
    if hasattr(game_plugin, "shaper") and game_plugin.shaper:
        reward_shaper = game_plugin.shaper(config.device)

    algo: BaseAlgorithm = get_algorithm(
        args.algorithm,
        config=config,
        obs_dim=obs_dim,
        action_dim=action_dim,
        model_name=model_name,
        reward_shaper=reward_shaper,
    )

    # 初始化规则模型 (仅用于 simple_duel 评估)
    rule_agent = None
    if args.game == "simple_duel":
        rule_agent = RuleBasedAgent(device=str(config.device))

    # 训练统计
    global_step = 0
    start_time = time.time()
    train_metrics = {}

    # 获取初始状态
    obs, mask = env.reset()
    obs = torch.FloatTensor(obs).to(config.device)
    mask = torch.FloatTensor(mask).to(config.device)

    # Episode 统计
    current_returns = np.zeros(2 * config.num_envs)
    episode_returns = deque(maxlen=100)
    current_lengths = np.zeros(2 * config.num_envs)
    episode_lengths = deque(maxlen=100)

    # 详细统计
    stats = {
        "p1_wins": deque(maxlen=100),
        "p2_wins": deque(maxlen=100),
        "draws": deque(maxlen=100),
        "p1_attacks": deque(maxlen=100),
        "p2_attacks": deque(maxlen=100),
        "p1_damage": deque(maxlen=100),
        "p2_damage": deque(maxlen=100),
    }

    print("=" * 60)
    print("  PPO Self-Play Training")
    print("=" * 60)
    print(f"  Game: {args.game} | Algorithm: {args.algorithm}")
    print(f"  Model: {model_name}")
    print(f"  Obs Dim: {obs_dim}, Action Dim: {action_dim}")
    print(
        f"  Batch Size: {config.batch_size:,}, Total Steps: {config.total_timesteps:,}"
    )
    print(f"  Reward Shaping: {config.use_reward_shaping}")
    print(f"  Target KL: {config.target_kl}")
    print(f"  Learning Rate: {config.learning_rate}")
    print("=" * 60)
    print()

    # 创建检查点目录
    checkpoint_dir = Path("checkpoints") / f"{args.game}_{args.algorithm}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    try:
        for update in range(1, config.train_steps + 1):
            # -----------------------------------------------------------------
            # 1. Rollout Phase (Self-Play)
            # -----------------------------------------------------------------
            for step in range(config.num_steps):
                global_step += 2 * config.num_envs

                with torch.no_grad():
                    action, info = algo.get_action(obs, mask)

                # 执行动作
                actions_np = action.cpu().numpy()
                actions_p1 = actions_np[: config.num_envs]
                actions_p2 = actions_np[config.num_envs :]

                obs_new, reward, done, mask_new, info_list = env.step(
                    actions_p1.tolist(), actions_p2.tolist()
                )

                # 转换为 tensor
                obs_new = torch.FloatTensor(obs_new).to(config.device)
                mask_new = torch.FloatTensor(mask_new).to(config.device)
                reward = torch.FloatTensor(reward).to(config.device)
                done = torch.BoolTensor(done).to(config.device)

                # 扩展 done 到 2*N
                done_expanded = torch.cat([done, done])

                # 存储转移 (包含 next_obs 用于奖励塑形)
                algo.store_transition(
                    obs,
                    action,
                    reward,
                    done_expanded,
                    mask,
                    info,
                    next_obs=obs_new if config.use_reward_shaping else None,
                )

                obs = obs_new
                mask = mask_new

                # 记录统计
                current_returns += reward.cpu().numpy()
                current_lengths += 1

                done_np = done.cpu().numpy()

                for i, d in enumerate(done_np):
                    if d:
                        if info_list and len(info_list) > i:
                            info_item = info_list[i]
                            if info_item:
                                if "p1_win" in info_item:
                                    if info_item["p1_win"] > 0.5:
                                        stats["p1_wins"].append(1)
                                        stats["p2_wins"].append(0)
                                        stats["draws"].append(0)
                                    elif info_item["p2_win"] > 0.5:
                                        stats["p1_wins"].append(0)
                                        stats["p2_wins"].append(1)
                                        stats["draws"].append(0)
                                    else:
                                        stats["p1_wins"].append(0)
                                        stats["p2_wins"].append(0)
                                        stats["draws"].append(1)

                                if "p1_attacks" in info_item:
                                    stats["p1_attacks"].append(info_item["p1_attacks"])
                                if "p2_attacks" in info_item:
                                    stats["p2_attacks"].append(info_item["p2_attacks"])
                                if "p1_damage" in info_item:
                                    stats["p1_damage"].append(info_item["p1_damage"])
                                if "p2_damage" in info_item:
                                    stats["p2_damage"].append(info_item["p2_damage"])

                        episode_returns.append(current_returns[i])
                        current_returns[i] = 0
                        episode_returns.append(current_returns[config.num_envs + i])
                        current_returns[config.num_envs + i] = 0

                        episode_lengths.append(current_lengths[i])
                        current_lengths[i] = 0
                        episode_lengths.append(current_lengths[config.num_envs + i])
                        current_lengths[config.num_envs + i] = 0

            # -----------------------------------------------------------------
            # 2. Update Phase
            # -----------------------------------------------------------------
            train_metrics = algo.update(next_obs=obs) or {}

            # -----------------------------------------------------------------
            # 3. Logging
            # -----------------------------------------------------------------
            if update % 10 == 0:
                avg_ret = np.mean(episode_returns) if episode_returns else 0.0
                avg_len = np.mean(episode_lengths) if episode_lengths else 0.0

                p1_win_rate = np.mean(stats["p1_wins"]) if stats["p1_wins"] else 0.0
                p2_win_rate = np.mean(stats["p2_wins"]) if stats["p2_wins"] else 0.0
                draw_rate = np.mean(stats["draws"]) if stats["draws"] else 0.0
                p1_avg_atk = (
                    np.mean(stats["p1_attacks"]) if stats["p1_attacks"] else 0.0
                )
                p2_avg_atk = (
                    np.mean(stats["p2_attacks"]) if stats["p2_attacks"] else 0.0
                )
                p1_avg_dmg = np.mean(stats["p1_damage"]) if stats["p1_damage"] else 0.0
                p2_avg_dmg = np.mean(stats["p2_damage"]) if stats["p2_damage"] else 0.0

                fps = int(global_step / (time.time() - start_time))
                lr = train_metrics.get("lr", config.learning_rate)

                print(
                    f"Update {update}/{config.train_steps} | Steps: {global_step:,} | FPS: {fps:,}"
                )
                print(f"  Return: {avg_ret:.2f} | EpLen: {avg_len:.1f}")
                print(
                    f"  WinRate: P1 {p1_win_rate:.2%} / P2 {p2_win_rate:.2%} / Draw {draw_rate:.2%}"
                )
                if args.game == "simple_duel":
                    print(
                        f"  AvgAtk: P1 {p1_avg_atk:.1f} / P2 {p2_avg_atk:.1f} | AvgDmg: P1 {p1_avg_dmg:.1f} / P2 {p2_avg_dmg:.1f}"
                    )
                print(
                    f"  Loss: {train_metrics.get('loss', 0):.4f} | VLoss: {train_metrics.get('v_loss', 0):.4f} | "
                    f"KL: {train_metrics.get('approx_kl', 0):.4f} | LR: {lr:.2e}"
                )
                print()

            # -----------------------------------------------------------------
            # 4. Evaluation & Replay
            # -----------------------------------------------------------------
            if update % config.eval_interval == 0 or update == 1:
                if args.game == "simple_duel":
                    print(f"  >> Generating replay...")
                    opponent = rule_agent if config.eval_opponent == "rule" else algo
                    render_ascii_game_to_file(
                        algo, opponent, config, args.game, "replays"
                    )

            # -----------------------------------------------------------------
            # 5. Save Checkpoint
            # -----------------------------------------------------------------
            if hasattr(config, "save_interval") and update % config.save_interval == 0:
                checkpoint_path = checkpoint_dir / f"algo_step_{global_step}.pth"
                algo.save(str(checkpoint_path))
                print(f"  >> Checkpoint saved: {checkpoint_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving model...")
        save_name = f"{args.game}_{args.algorithm}_interrupted.pth"
        algo.save(save_name)
        print(f"Model saved to {save_name}")
        return algo
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        return None

    print("Training finished!")
    final_save_name = f"{args.game}_{args.algorithm}_final.pth"
    algo.save(final_save_name)
    print(f"Model saved to {final_save_name}")
    return algo


def main():
    parser = argparse.ArgumentParser(description="RL Self-Play Training")
    parser.add_argument(
        "--game",
        type=str,
        default="simple_duel",
        help="Game to train on (simple_duel, tictactoe)",
    )
    parser.add_argument(
        "--algorithm", type=str, default="ppo", help="Algorithm to use (ppo, a2c)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture (auto-selected if not specified)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "fast", "long"],
        help="Training configuration",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    args = parser.parse_args()

    # 选择配置
    if args.config == "fast":
        config = FastConfig()
    elif args.config == "long":
        config = LongConfig()
    else:
        config = Config()

    # 更新配置中的游戏设置
    config.game = args.game
    config.__post_init__()
    # 使用游戏特定的奖励塑形设置
    if hasattr(config.game_config, "use_reward_shaping"):
        config.use_reward_shaping = config.game_config.use_reward_shaping

    # 训练
    agent = train(args, config)

    # 如果指定了恢复
    if args.resume and agent:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")


if __name__ == "__main__":
    main()
