#!/usr/bin/env python3
"""Integration test script for RL self-play framework.

Tests all 12 combinations of games and algorithms:
- Games: simple_duel, tictactoe, connect4, reversi
- Algorithms: ppo, dqn, sac

For each combination:
1. Create environment with num_envs=1
2. Get game info
3. Select appropriate model (cnn for connect4/reversi, actor_critic for others)
4. Initialize algorithm
5. Run 2 steps of interaction
6. Run 1 update step
"""

import traceback
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from config import Config
from envs import create_env, get_game_info, has_native_backend
from algorithms.registry import build_algorithm


if not has_native_backend():
    pytestmark = pytest.mark.skip(
        reason="Rust extension `high_perf_env` is not built; run `maturin develop --release`."
    )


@dataclass
class TestResult:
    """Result of a single integration test."""

    game: str
    algorithm: str
    success: bool
    error: Optional[str] = None


TestResult.__test__ = False


def select_model(game: str) -> str:
    """Select appropriate model for the game."""
    if game == "simple_duel":
        return "actor_critic"
    return "simple_mlp"


def run_combination(game: str, algo_name: str) -> TestResult:
    """Test a single game-algorithm combination.

    Args:
        game: Game name
        algo_name: Algorithm name

    Returns:
        TestResult with success status and optional error message
    """
    try:
        # 1. Create environment with num_envs=1
        env = create_env(game, num_envs=1)

        # 2. Get game info
        game_info = get_game_info(game)
        obs_dim = game_info.obs_dim
        action_dim = game_info.action_dim

        # 3. Select model
        model_name = select_model(game)

        # 4. Initialize algorithm
        # Create a minimal config for testing
        config = Config()
        config.num_envs = 1
        config.num_steps = 2
        config.batch_size = 2 * 1 * 2  # 2 * num_envs * num_steps
        config.minibatch_size = 4
        config.device = torch.device("cpu")  # Use CPU for testing

        algo = build_algorithm(
            algo_name,
            config=config,
            obs_dim=obs_dim,
            action_dim=action_dim,
            model_name=model_name,
            device="cpu",
        )

        # 5. Run 2 steps of interaction
        obs, mask = env.reset()
        obs = torch.FloatTensor(obs)
        mask = torch.FloatTensor(mask)

        for _ in range(2):
            # Get action
            action, info = algo.get_action(obs, mask, deterministic=False)

            # Execute action (self-play: both players use same action for simplicity)
            actions_np = action.cpu().numpy()
            # For num_envs=1, we have 2 players
            actions_p1 = actions_np[:1].tolist()
            actions_p2 = actions_np[1:].tolist() if len(actions_np) > 1 else actions_p1

            obs_new, reward, done, mask_new, info_list = env.step(
                actions_p1, actions_p2
            )

            # Convert to tensors
            obs_new = torch.FloatTensor(obs_new)
            mask_new = torch.FloatTensor(mask_new)
            reward = torch.FloatTensor(reward)
            done = torch.BoolTensor(done)

            # Expand done for 2 players
            done_expanded = torch.cat([done, done])

            # Store transition (PPO doesn't accept next_mask, DQN/SAC do)
            if algo_name == "ppo":
                algo.store_transition(
                    obs,
                    action,
                    reward,
                    done_expanded,
                    mask,
                    info,
                    next_obs=obs_new,
                )
            else:
                algo.store_transition(
                    obs,
                    action,
                    reward,
                    done_expanded,
                    mask,
                    info,
                    next_obs=obs_new,
                    next_mask=mask_new,
                )

            obs = obs_new
            mask = mask_new

        # 6. Run 1 update step
        algo.update(next_obs=obs)

        return TestResult(game=game, algorithm=algo_name, success=True)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return TestResult(
            game=game, algorithm=algo_name, success=False, error=error_msg
        )


def main():
    """Run all integration tests and print summary."""
    games = ["simple_duel", "tictactoe", "connect4", "reversi"]
    algorithms = ["ppo", "dqn", "sac"]

    print("=" * 70)
    print("  RL Self-Play Framework Integration Tests")
    print("=" * 70)
    print(f"  Games: {', '.join(games)}")
    print(f"  Algorithms: {', '.join(algorithms)}")
    print(f"  Total combinations: {len(games) * len(algorithms)}")
    print("=" * 70)
    print()

    results: list[TestResult] = []

    for game in games:
        for algo in algorithms:
            print(f"Testing {game} + {algo}...", end=" ", flush=True)
            result = run_combination(game, algo)
            results.append(result)

            if result.success:
                print("PASS")
            else:
                print("FAIL")
                # Print error details indented
                if result.error:
                    for line in result.error.split("\n")[:5]:  # First 5 lines
                        print(f"    {line}")

    # Print summary table
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Game':<15} | {'Algorithm':<10} | {'Result':<10}")
    print("-" * 40)

    passed = 0
    failed = 0
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"{r.game:<15} | {r.algorithm:<10} | {status:<10}")
        if r.success:
            passed += 1
        else:
            failed += 1

    print("-" * 40)
    print(f"Total: {passed}/{len(results)} passed")
    print()

    if failed > 0:
        print("FAILED TESTS:")
        for r in results:
            if not r.success:
                print(f"\n  {r.game} + {r.algorithm}:")
                if r.error:
                    for line in r.error.split("\n")[:10]:
                        print(f"    {line}")

    # Exit with appropriate code
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
