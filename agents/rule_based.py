import torch
import numpy as np
import random

class RuleBasedAgent:
    def __init__(self, device="cpu"):
        self.device = device
    
    def get_action(self, obs, mask=None, deterministic=False):
        """
        Rule-based agent:
        1. Kill if possible (HP<=1).
        2. Flee if Danger (HP<=1).
        3. Kiting/Positioning: Maintain range and LOS.
        4. Approach (BFS) if no better option.
        
        Args:
            obs: Tensor [B, 72]
            mask: Tensor [B, 7]
        """
        B = obs.shape[0]
        obs_np = obs.cpu().numpy()
        if mask is not None:
            mask_np = mask.cpu().numpy()
        else:
            mask_np = None
        
        # Constants
        MAP_SIZE = 8
        MAX_HP = 3.0
        MAX_ENERGY = 5.0
        
        # De-normalize helpers
        def denorm_pos(val): return int(round(val * (MAP_SIZE - 1)))
        def denorm_hp(val): return val * MAX_HP
        def denorm_eng(val): return val * MAX_ENERGY
        
        actions = []
        
        for i in range(B):
            o = obs_np[i]
            mx, my = denorm_pos(o[0]), denorm_pos(o[1])
            ex, ey = denorm_pos(o[2]), denorm_pos(o[3])
            mhp, ehp = denorm_hp(o[4]), denorm_hp(o[5])
            meng, eeng = denorm_eng(o[6]), denorm_eng(o[7])
            
            # Walls
            walls = o[8:].reshape((MAP_SIZE, MAP_SIZE)) > 0.5
            
            # Mask
            m = mask_np[i] if mask_np is not None else np.ones(7)
            
            act = 0 # Default Stay
            
            dist = abs(mx - ex) + abs(my - ey) # Manhattan
            cheb_dist = max(abs(mx - ex), abs(my - ey)) # Chebyshev for shooting range
            
            can_shoot = m[6] > 0.5
            can_attack = m[5] > 0.5
            
            # Helper: Check LOS (Python version of Bresenham with diagonal check)
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
                    if (x, y) != p1 and walls[y, x]: return False
                    
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
                        if walls[y, nx] and walls[ny, x]: return False
                        
                    x, y = nx, ny
                return True

            # 1. Kill Opportunity (Greedy)
            if ehp <= 1.5:
                if can_attack: act = 5
                elif can_shoot: act = 6
            
            # 2. Survival (Flee)
            if act == 0 and mhp <= 1.5:
                # Simple Flee: Maximize distance
                best_move = 0
                max_d = dist
                moves = [(1, (0, 1)), (2, (0, -1)), (3, (-1, 0)), (4, (1, 0))]
                for ma, (dx, dy) in moves:
                    if m[ma] > 0.5:
                        nx, ny = mx + dx, my + dy
                        d = abs(nx - ex) + abs(ny - ey)
                        if d > max_d:
                            max_d = d
                            best_move = ma
                if best_move != 0:
                    act = best_move
            
            # 3. Combat / Kiting
            if act == 0:
                # If we can shoot (Energy ok), try to shoot or move to shooting pos
                if can_shoot and cheb_dist <= 3 and has_los((mx, my), (ex, ey)):
                    act = 6
                elif meng >= 3.0: # Has energy for shoot, but maybe not in pos
                    # BFS to find best shooting position
                    # Best Pos: Range <= 3, Has LOS, Maximize Distance (Safety)
                    queue = [(mx, my, [])]
                    visited = set([(mx, my)])
                    
                    best_target = None
                    best_score = -1
                    best_path_to_target = None
                    
                    # BFS
                    head = 0
                    while head < len(queue):
                        cx, cy, path = queue[head]
                        head += 1
                        
                        if len(path) > 6: continue # Limit search
                        
                        # Check if this is a good spot
                        cdist = max(abs(cx - ex), abs(cy - ey))
                        if cdist <= 3:
                            if has_los((cx, cy), (ex, ey)):
                                # Score: Distance (prefer far)
                                score = cdist
                                if score > best_score:
                                    best_score = score
                                    best_target = (cx, cy)
                                    best_path_to_target = path
                                elif score == best_score:
                                    # Tie breaker: shorter path
                                    if best_path_to_target and len(path) < len(best_path_to_target):
                                        best_path_to_target = path
                                        best_target = (cx, cy)
                        
                        # Neighbors
                        for ma, (dx, dy) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)], 1):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                                if not walls[ny, nx] and (nx, ny) not in visited:
                                    visited.add((nx, ny))
                                    queue.append((nx, ny, path + [ma]))
                    
                    if best_path_to_target:
                         act = best_path_to_target[0]
                         # Check mask
                         if m[act] < 0.5: act = 0 # Fallback
            
            # 4. Approach (If nothing else, and has energy)
            if act == 0 and meng >= 1.0:
                 # Standard BFS to enemy
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
                            if m[act] < 0.5: act = 0
                        found = True
                        break
                    
                    if len(path) > 8: continue
                    
                    for ma, (dx, dy) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)], 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                            if not walls[ny, nx] and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny, path + [ma]))

            actions.append(act)
            
        return torch.LongTensor(actions).to(self.device), {}
