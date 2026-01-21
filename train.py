import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import high_perf_env
import time
from collections import deque
from config import Config
from algorithms.ppo import PPO
from agents.rule_based import RuleBasedAgent
from utils import render_ascii_game_to_file

def train():
    config = Config()
    
    # 初始化环境
    # VectorizedEnv 内部自动创建 config.num_envs 个 SimpleDuel
    env = high_perf_env.VectorizedEnv(config.num_envs)
    
    # 初始化算法 (PPO)
    # Obs Dim = 72 (8 stats + 64 grid), Action Dim = 7
    ppo_agent = PPO(config, obs_dim=72, action_dim=7)
    
    # 初始化规则模型 (用于评估)
    rule_agent = RuleBasedAgent(device=config.device)
    
    # 训练统计
    global_step = 0
    start_time = time.time()
    
    # 获取初始状态
    # obs, mask 的 shape 都是 [2 * num_envs, dim]
    obs, mask = env.reset()
    obs = torch.FloatTensor(obs).to(config.device)
    mask = torch.FloatTensor(mask).to(config.device)
    
    # 存储 episode returns
    current_returns = np.zeros(2 * config.num_envs)
    episode_returns = deque(maxlen=100)
    current_lengths = np.zeros(2 * config.num_envs)
    episode_lengths = deque(maxlen=100)
    
    # 统计信息
    stats = {
        "p1_wins": deque(maxlen=100),
        "p2_wins": deque(maxlen=100),
        "draws": deque(maxlen=100),
        "p1_attacks": deque(maxlen=100),
        "p2_attacks": deque(maxlen=100),
    }
    
    print("Starting training...")
    
    # 训练循环
    try:
        for update in range(1, config.train_steps + 1):
            # ---------------------------------------------------------------------
            # 1. Rollout Phase (Self-Play: PPO vs PPO)
            # ---------------------------------------------------------------------
            for step in range(config.num_steps):
                global_step += 2 * config.num_envs
                
                with torch.no_grad():
                    # 双方都使用 PPO 策略
                    # generic get_action returns (action, info)
                    action, info = ppo_agent.get_action(obs, mask)
                    
                # Step
                actions_np = action.cpu().numpy()
                actions_p1 = actions_np[:config.num_envs] # [N]
                actions_p2 = actions_np[config.num_envs:] # [N]
                
                obs_new, reward, done, mask_new, info_list = env.step(actions_p1.tolist(), actions_p2.tolist())
                
                # 转换 tensor
                obs_new = torch.FloatTensor(obs_new).to(config.device)
                mask_new = torch.FloatTensor(mask_new).to(config.device)
                reward = torch.FloatTensor(reward).to(config.device)
                done = torch.BoolTensor(done).to(config.device) # [N]
                
                # 处理 Done (Expand to 2*N for storage)
                done_expanded = torch.cat([done, done]) # [2*N]
                
                # Store Transition
                ppo_agent.store_transition(obs, action, reward, done_expanded, mask, info)
                
                obs = obs_new
                mask = mask_new
                
                # 记录统计信息
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
                        
                        episode_returns.append(current_returns[i])
                        current_returns[i] = 0
                        episode_returns.append(current_returns[config.num_envs + i])
                        current_returns[config.num_envs + i] = 0
                        
                        episode_lengths.append(current_lengths[i])
                        current_lengths[i] = 0
                        episode_lengths.append(current_lengths[config.num_envs + i])
                        current_lengths[config.num_envs + i] = 0

            # ---------------------------------------------------------------------
            # 2. Update Phase (Generic)
            # ---------------------------------------------------------------------
            # Update Algorithm (passing next_obs for GAE if needed)
            train_metrics = ppo_agent.update(next_obs=obs)

            # ---------------------------------------------------------------------
            # 3. Logging & Eval
            # ---------------------------------------------------------------------
            if update % 10 == 0:
                avg_ret = np.mean(episode_returns) if len(episode_returns) > 0 else 0.0
                avg_len = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0.0
                
                p1_win_rate = np.mean(stats["p1_wins"]) if len(stats["p1_wins"]) > 0 else 0.0
                p2_win_rate = np.mean(stats["p2_wins"]) if len(stats["p2_wins"]) > 0 else 0.0
                draw_rate = np.mean(stats["draws"]) if len(stats["draws"]) > 0 else 0.0
                p1_avg_atk = np.mean(stats["p1_attacks"]) if len(stats["p1_attacks"]) > 0 else 0.0
                p2_avg_atk = np.mean(stats["p2_attacks"]) if len(stats["p2_attacks"]) > 0 else 0.0
                
                fps = int(global_step / (time.time() - start_time))
                print(f"Step {update}/{config.train_steps} | Steps: {global_step} | FPS: {fps}")
                print(f"  Return: {avg_ret:.2f} | Len: {avg_len:.1f}")
                print(f"  WinRate: P1 {p1_win_rate:.2f} / P2 {p2_win_rate:.2f} / Draw {draw_rate:.2f}")
                print(f"  AvgAttacks: P1 {p1_avg_atk:.1f} / P2 {p2_avg_atk:.1f}")
                print(f"  Loss: {train_metrics.get('loss', 0):.4f} | ValLoss: {train_metrics.get('v_loss', 0):.4f}")

            if update % config.eval_interval == 0 or update == 1:
                 print(f"Logging replay to 'replays' directory...")
                 # 评估: P1 (PPO) vs P2 (Rule or Self)
                 opponent = rule_agent
                 if config.eval_opponent == "self":
                     opponent = ppo_agent
                 
                 render_ascii_game_to_file(ppo_agent, opponent, config, "replays")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving model...")
        ppo_agent.save("ppo_interrupted.pth")
        print("Model saved to ppo_interrupted.pth")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("Training finished!")
    ppo_agent.save("ppo_final.pth")

if __name__ == "__main__":
    train()
