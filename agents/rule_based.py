import torch
import numpy as np
import random

class RuleBasedAgent:
    def __init__(self, device="cpu"):
        self.device = device

    def get_action(self, obs, mask=None, deterministic=False):
        """
        Rule-based agent for upgraded 12x12 tactical game:
        1. Kill if possible (HP<=1).
        2. Use Heal if low HP and available.
        3. Use Dodge/Shield if about to be attacked.
        4. Flee if Danger (HP<=1).
        5. Reload if low ammo.
        6. Kiting/Positioning: Maintain range and LOS.
        7. Approach (BFS) if no better option.

        Args:
            obs: Tensor [B, 160]
            mask: Tensor [B, 13]
        """
        B = obs.shape[0]
        obs_np = obs.cpu().numpy()
        if mask is not None:
            mask_np = mask.cpu().numpy()
        else:
            mask_np = None

        # Constants - match upgraded environment
        MAP_SIZE = 12
        MAX_HP = 4.0
        MAX_ENERGY = 7.0
        MAX_SHIELD = 2.0
        MAX_AMMO = 6.0

        # De-normalize helpers
        def denorm_pos(val): return int(round(val * (MAP_SIZE - 1)))
        def denorm_hp(val): return val * MAX_HP
        def denorm_eng(val): return val * MAX_ENERGY
        def denorm_shield(val): return int(round(val * MAX_SHIELD))
        def denorm_ammo(val): return int(round(val * MAX_AMMO))

        # Action indices
        ACT_STAY = 0
        ACT_UP = 1
        ACT_DOWN = 2
        ACT_LEFT = 3
        ACT_RIGHT = 4
        ACT_ATTACK = 5
        ACT_SHOOT = 6
        ACT_DODGE = 7
        ACT_SHIELD = 8
        ACT_DASH = 9
        ACT_AOE = 10
        ACT_HEAL = 11
        ACT_RELOAD = 12

        actions = []

        for i in range(B):
            o = obs_np[i]
            mx, my = denorm_pos(o[0]), denorm_pos(o[1])
            ex, ey = denorm_pos(o[2]), denorm_pos(o[3])
            mhp, ehp = denorm_hp(o[4]), denorm_hp(o[5])
            meng, eeng = denorm_eng(o[6]), denorm_eng(o[7])
            mshield = denorm_shield(o[8])
            eshield = denorm_shield(o[9])
            mammo = denorm_ammo(o[10])
            eammo = denorm_ammo(o[11])
            my_dodge = o[12] > 0.5
            enemy_dodge = o[13] > 0.5
            heal_cd = int(round(o[14] * 5))

            # Parse terrain from grid (index 16-159)
            grid_data = o[16:]
            terrain = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)
            for gy in range(MAP_SIZE):
                for gx in range(MAP_SIZE):
                    idx = gy * MAP_SIZE + gx
                    val = grid_data[idx]
                    if val < 0.1:
                        terrain[gy, gx] = 0  # Empty
                    elif val < 0.4:
                        terrain[gy, gx] = 1  # Wall
                    elif val < 0.6:
                        terrain[gy, gx] = 2  # Water
                    else:
                        terrain[gy, gx] = 3  # High ground or other

            # Check if position is wall
            def is_wall(x, y):
                if x < 0 or x >= MAP_SIZE or y < 0 or y >= MAP_SIZE:
                    return True
                return terrain[y, x] == 1

            # Mask
            m = mask_np[i] if mask_np is not None else np.ones(13)

            act = 0  # Default Stay

            dist = abs(mx - ex) + abs(my - ey)  # Manhattan
            cheb_dist = max(abs(mx - ex), abs(my - ey))  # Chebyshev for range

            can_shoot = m[ACT_SHOOT] > 0.5
            can_attack = m[ACT_ATTACK] > 0.5
            can_aoe = m[ACT_AOE] > 0.5
            can_heal = m[ACT_HEAL] > 0.5
            can_dodge = m[ACT_DODGE] > 0.5
            can_shield = m[ACT_SHIELD] > 0.5
            can_dash = m[ACT_DASH] > 0.5
            can_reload = m[ACT_RELOAD] > 0.5

            # Helper: Check LOS (Python version of Bresenham)
            def has_los(p1, p2):
                x0, y0 = p1
                x1, y1 = p2
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                err = dx - dy

                x, y = x0, y0
                while (x, y) != (x1, y1):
                    if (x, y) != p1 and is_wall(x, y):
                        return False

                    e2 = 2 * err
                    nx, ny = x, y
                    if e2 > -dy:
                        err -= dy
                        nx += sx
                    if e2 < dx:
                        err += dx
                        ny += sy

                    # Diagonal check
                    if nx != x and ny != y:
                        if is_wall(nx, y) and is_wall(x, ny):
                            return False

                    x, y = nx, ny
                return True

            # 1. Kill Opportunity (Greedy) - prioritize if enemy low HP
            if ehp <= 1.5:
                if can_attack:
                    act = ACT_ATTACK
                elif can_aoe:
                    act = ACT_AOE
                elif can_shoot:
                    act = ACT_SHOOT

            # 2. Use Heal if low HP and available
            if act == 0 and mhp <= 2 and can_heal:
                act = ACT_HEAL

            # 3. Defensive: Use Shield if taking damage and low shields
            if act == 0 and mhp <= 2 and mshield < MAX_SHIELD and can_shield:
                # If enemy is close, consider shield
                if cheb_dist <= 2:
                    act = ACT_SHIELD

            # 4. Defensive: Use Dodge if enemy might attack
            if act == 0 and mhp <= 2 and can_dodge and cheb_dist <= 1:
                act = ACT_DODGE

            # 5. Survival (Flee) - use Dash if available
            if act == 0 and mhp <= 1.5:
                if can_dash:
                    act = ACT_DASH
                else:
                    # Simple Flee: Maximize distance
                    best_move = 0
                    max_d = dist
                    moves = [(ACT_UP, (0, 1)), (ACT_DOWN, (0, -1)),
                             (ACT_LEFT, (-1, 0)), (ACT_RIGHT, (1, 0))]
                    for ma, (dx, dy) in moves:
                        if m[ma] > 0.5:
                            nx, ny = mx + dx, my + dy
                            d = abs(nx - ex) + abs(ny - ey)
                            if d > max_d:
                                max_d = d
                                best_move = ma
                    if best_move != 0:
                        act = best_move

            # 6. Reload if low ammo
            if act == 0 and mammo <= 2 and can_reload:
                act = ACT_RELOAD

            # 7. Combat / Kiting
            if act == 0:
                # If we can shoot (Energy ok, ammo ok), try to shoot or move to shooting pos
                if can_shoot and cheb_dist <= 4 and has_los((mx, my), (ex, ey)):
                    act = ACT_SHOOT
                elif can_attack and cheb_dist <= 1:
                    act = ACT_ATTACK
                elif can_aoe and cheb_dist <= 1:
                    act = ACT_AOE
                elif meng >= 3.0 and mammo > 0:
                    # BFS to find best shooting position
                    queue = [(mx, my, [])]
                    visited = set([(mx, my)])

                    best_target = None
                    best_score = -1
                    best_path_to_target = None

                    head = 0
                    while head < len(queue):
                        cx, cy, path = queue[head]
                        head += 1

                        if len(path) > 6:
                            continue

                        cdist = max(abs(cx - ex), abs(cy - ey))
                        if cdist <= 4:
                            if has_los((cx, cy), (ex, ey)):
                                score = cdist
                                if score > best_score:
                                    best_score = score
                                    best_target = (cx, cy)
                                    best_path_to_target = path
                                elif score == best_score:
                                    if best_path_to_target and len(path) < len(best_path_to_target):
                                        best_path_to_target = path
                                        best_target = (cx, cy)

                        # Neighbors
                        for ma, (dx, dy) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)], 1):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                                if not is_wall(nx, ny) and (nx, ny) not in visited:
                                    visited.add((nx, ny))
                                    queue.append((nx, ny, path + [ma]))

                    if best_path_to_target:
                        act = best_path_to_target[0]
                        if m[act] < 0.5:
                            act = 0

            # 8. Approach (If nothing else, and has energy)
            if act == 0 and meng >= 1.0:
                queue = [(mx, my, [])]
                visited = set([(mx, my)])
                target = (ex, ey)
                found = False
                head = 0
                while head < len(queue):
                    cx, cy, path = queue[head]
                    head += 1
                    if (cx, cy) == target:
                        if path:
                            act = path[0]
                            if m[act] < 0.5:
                                act = 0
                        found = True
                        break

                    if len(path) > 10:
                        continue

                    for ma, (dx, dy) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)], 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                            if not is_wall(nx, ny) and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny, path + [ma]))

            actions.append(act)

        return torch.LongTensor(actions).to(self.device), {}
