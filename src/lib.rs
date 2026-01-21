use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray2, PyArray1, PyArrayMethods};
use rayon::prelude::*;
use rand::prelude::*;

// ============================================================================
// 1. 核心 Trait 定义
// ============================================================================

/// 定义通用游戏环境接口
/// 所有的具体游戏逻辑（如 GridWorld, 连续控制等）都应实现此 Trait
pub trait GameEnv: Send + Sync + Clone {
    // 这里为了简化 Python 交互，我们假定 Action 是 usize (离散)，Obs 是 Vec<f32>
    // 如果需要连续动作，可以改为 Vec<f32>
    
    // 创建新游戏实例
    fn new() -> Self;
    
    // 重置游戏
    // 返回 (Obs_P1, Obs_P2, Mask_P1, Mask_P2)
    // Mask: 1.0 表示合法，0.0 表示非法
    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    
    // 执行一步
    // 输入: P1 和 P2 的动作
    // 返回: (Obs_P1, Obs_P2, Reward_P1, Reward_P2, Done, Mask_P1, Mask_P2, Info)
    // Info: 统计信息，仅在 Done=true 时有内容，否则为空
    fn step(&mut self, action_p1: usize, action_p2: usize) 
        -> (Vec<f32>, Vec<f32>, f32, f32, bool, Vec<f32>, Vec<f32>, HashMap<String, f32>);
        
    fn obs_dim() -> usize;
    fn action_dim() -> usize;
}

// ============================================================================
// 2. SimpleDuel 实现
// ============================================================================

const MAP_SIZE: i32 = 8;
const MAX_HP: i32 = 3;
const MAX_ENERGY: i32 = 5;
const REGEN_ENERGY: i32 = 1;

const COST_MOVE: i32 = 1;
const COST_ATTACK: i32 = 2;
const COST_SHOOT: i32 = 3;

const ATTACK_RANGE: i32 = 1;
const SHOOT_RANGE: i32 = 3;
const MAX_STEPS: i32 = 50;
const SUDDEN_DEATH_STEP: i32 = 30; // Stop energy regen after this step

// 动作定义
// 0: Stay
// 1: Up (y+)
// 2: Down (y-)
// 3: Left (x-)
// 4: Right (x+)
// 5: Attack (Melee)
// 6: Shoot (Ranged)
const ACT_STAY: usize = 0;
const ACT_UP: usize = 1;
const ACT_DOWN: usize = 2;
const ACT_LEFT: usize = 3;
const ACT_RIGHT: usize = 4;
const ACT_ATTACK: usize = 5;
const ACT_SHOOT: usize = 6;

const ACTION_DIM: usize = 7;
// [mx, my, ex, ey, mhp, ehp, meng, eeng] + [Map Grid 8x8 = 64]
const OBS_DIM: usize = 8 + 64; 

#[derive(Clone)]
struct SimpleDuel {
    p1_pos: (i32, i32),
    p2_pos: (i32, i32),
    p1_hp: i32,
    p2_hp: i32,
    p1_energy: i32,
    p2_energy: i32,
    step_count: i32,
    walls: Vec<bool>, // Flattened MAP_SIZE*MAP_SIZE grid
    rng: StdRng,
    // 统计信息
    p1_attacks: i32,
    p2_attacks: i32,
}

impl SimpleDuel {
    // 辅助函数：判断是否在地图内
    fn is_valid(pos: (i32, i32)) -> bool {
        pos.0 >= 0 && pos.0 < MAP_SIZE && pos.1 >= 0 && pos.1 < MAP_SIZE
    }

    fn is_wall(&self, pos: (i32, i32)) -> bool {
        if !Self::is_valid(pos) { return true; } // Out of bounds is wall-like
        let idx = (pos.1 * MAP_SIZE + pos.0) as usize;
        self.walls[idx]
    }

    // 辅助函数：生成 Observation
    // is_p2_perspective: 是否生成 P2 的视角（需要翻转）
    fn get_obs(&self, is_p2_perspective: bool) -> Vec<f32> {
        let (my_pos, enemy_pos, my_hp, enemy_hp, my_eng, enemy_eng) = if is_p2_perspective {
            (self.p2_pos, self.p1_pos, self.p2_hp, self.p1_hp, self.p2_energy, self.p1_energy)
        } else {
            (self.p1_pos, self.p2_pos, self.p1_hp, self.p2_hp, self.p1_energy, self.p2_energy)
        };

        let (mx, my) = self.transform_pos(my_pos, is_p2_perspective);
        let (ex, ey) = self.transform_pos(enemy_pos, is_p2_perspective);

        // 归一化到 0-1
        let scale = (MAP_SIZE - 1) as f32;
        let hp_scale = MAX_HP as f32;
        let eng_scale = MAX_ENERGY as f32;

        let mut obs = vec![
            mx as f32 / scale,
            my as f32 / scale,
            ex as f32 / scale,
            ey as f32 / scale,
            my_hp as f32 / hp_scale,
            enemy_hp as f32 / hp_scale,
            my_eng as f32 / eng_scale,
            enemy_eng as f32 / eng_scale,
        ];

        // Append Wall Grid
        // Grid should also be transformed for perspective!
        // If P2 perspective: (x, y) -> (W-1-x, H-1-y)
        // Original Grid: index = y * W + x
        // Transformed Grid: for y' in 0..H, for x' in 0..W:
        //    orig_x = W - 1 - x'
        //    orig_y = H - 1 - y'
        //    val = walls[orig_y * W + orig_x]
        
        for y in 0..MAP_SIZE {
            for x in 0..MAP_SIZE {
                let (qx, qy) = self.transform_pos((x, y), is_p2_perspective);
                // qx, qy is the coordinate in the AGENT'S view that corresponds to (x, y) in PHYSICAL view?
                // No. transform_pos converts PHYSICAL (x,y) to AGENT (ax, ay).
                // We want to fill the grid in AGENT'S coordinate system order (0..W, 0..H).
                // So for agent_y in 0..H, agent_x in 0..W:
                //    phys_x, phys_y = transform_pos_inverse(agent_x, agent_y)
                //    val = walls[phys_y * W + phys_x]
                
                // Since transform is symmetric (invert twice = identity), transform_pos is its own inverse.
                let (px, py) = self.transform_pos((x, y), is_p2_perspective);
                
                let idx = (py * MAP_SIZE + px) as usize;
                obs.push(if self.walls[idx] { 1.0 } else { 0.0 });
            }
        }
        
        obs
    }

    // 坐标转换
    // 如果是 P2 视角，我们进行中心对称翻转（或者理解为旋转180度）
    // (x, y) -> (W-1-x, H-1-y)
    // 这样 P2 在 (W-1, H-1) 时，他看到的自己是在 (0, 0)
    // 从而保证策略的通用性
    fn transform_pos(&self, pos: (i32, i32), invert: bool) -> (i32, i32) {
        if invert {
            (MAP_SIZE - 1 - pos.0, MAP_SIZE - 1 - pos.1)
        } else {
            pos
        }
    }

    // 动作转换
    // 如果是 P2 视角，他输出 "Up" (y+)，在翻转的坐标系里意味着 y 值增加。
    // 但在物理坐标系 (y_real = H-1 - y_obs) 里，y_obs 增加意味着 y_real 减少。
    // 所以 P2 的 "Up" 对应物理的 "Down"。
    // 同理 "Left" (x-) 对应物理的 "Right" (x+)。
    fn transform_action(action: usize, invert: bool) -> usize {
        if !invert {
            return action;
        }
        match action {
            ACT_UP => ACT_DOWN,
            ACT_DOWN => ACT_UP,
            ACT_LEFT => ACT_RIGHT,
            ACT_RIGHT => ACT_LEFT,
            _ => action,
        }
    }

    // 获取 Action Mask
    fn get_mask(&self, is_p2_perspective: bool) -> Vec<f32> {
        let mut mask = vec![0.0; ACTION_DIM];
        
        let (my_pos_phys, enemy_pos_phys, my_energy) = if is_p2_perspective { 
            (self.p2_pos, self.p1_pos, self.p2_energy) 
        } else { 
            (self.p1_pos, self.p2_pos, self.p1_energy) 
        };
        
        // 遍历所有逻辑动作
        for act in 0..ACTION_DIM {
            let phys_act = Self::transform_action(act, is_p2_perspective);
            let is_legal = match phys_act {
                ACT_STAY => true,
                ACT_UP => {
                     let target = (my_pos_phys.0, my_pos_phys.1 + 1);
                     my_pos_phys.1 < MAP_SIZE - 1 && my_energy >= COST_MOVE && !self.is_wall(target)
                },
                ACT_DOWN => {
                     let target = (my_pos_phys.0, my_pos_phys.1 - 1);
                     my_pos_phys.1 > 0 && my_energy >= COST_MOVE && !self.is_wall(target)
                },
                ACT_LEFT => {
                     let target = (my_pos_phys.0 - 1, my_pos_phys.1);
                     my_pos_phys.0 > 0 && my_energy >= COST_MOVE && !self.is_wall(target)
                },
                ACT_RIGHT => {
                     let target = (my_pos_phys.0 + 1, my_pos_phys.1);
                     my_pos_phys.0 < MAP_SIZE - 1 && my_energy >= COST_MOVE && !self.is_wall(target)
                },
                ACT_ATTACK => my_energy >= COST_ATTACK && self.check_hit(my_pos_phys, enemy_pos_phys, ATTACK_RANGE),
                ACT_SHOOT => my_energy >= COST_SHOOT && self.check_hit(my_pos_phys, enemy_pos_phys, SHOOT_RANGE),
                _ => false,
            };
            
            if is_legal {
                mask[act] = 1.0;
            }
        }
        mask
    }

    fn apply_move(&mut self, is_p2: bool, phys_action: usize) {
        let pos = if is_p2 { self.p2_pos } else { self.p1_pos }; // copy
        let mut new_pos = pos;
        
        match phys_action {
            ACT_UP => new_pos.1 = (pos.1 + 1).min(MAP_SIZE - 1),
            ACT_DOWN => new_pos.1 = (pos.1 - 1).max(0),
            ACT_LEFT => new_pos.0 = (pos.0 - 1).max(0),
            ACT_RIGHT => new_pos.0 = (pos.0 + 1).min(MAP_SIZE - 1),
            _ => {},
        }
        
        if !self.is_wall(new_pos) {
             if is_p2 { self.p2_pos = new_pos; } else { self.p1_pos = new_pos; }
        }
    }
    
    // Raycast check for walls
    fn has_line_of_sight(&self, p1: (i32, i32), p2: (i32, i32)) -> bool {
        let dx = (p2.0 - p1.0).abs();
        let dy = (p2.1 - p1.1).abs();
        let sx = if p1.0 < p2.0 { 1 } else { -1 };
        let sy = if p1.1 < p2.1 { 1 } else { -1 };
        let mut err = dx - dy;
        let mut x = p1.0;
        let mut y = p1.1;
        
        while (x, y) != p2 {
            if self.is_wall((x, y)) && (x, y) != p1 { // Don't block self
                 return false;
            }
            
            let e2 = 2 * err;
            let mut next_x = x;
            let mut next_y = y;
            
            if e2 > -dy {
                err -= dy;
                next_x += sx;
            }
            if e2 < dx {
                err += dx;
                next_y += sy;
            }
            
            // Diagonal check: Block if squeezing through two walls (closed corner)
            if next_x != x && next_y != y {
                 if self.is_wall((next_x, y)) && self.is_wall((x, next_y)) {
                     return false;
                 }
            }
            
            x = next_x;
            y = next_y;
        }
        true
    }

    fn check_hit(&self, attacker_pos: (i32, i32), target_pos: (i32, i32), range: i32) -> bool {
        let dx = (attacker_pos.0 - target_pos.0).abs();
        let dy = (attacker_pos.1 - target_pos.1).abs();
        if dx <= range && dy <= range {
            // Check walls
            return self.has_line_of_sight(attacker_pos, target_pos);
        }
        false
    }
}

impl GameEnv for SimpleDuel {
    fn new() -> Self {
        let rng = StdRng::from_entropy();
        SimpleDuel {
            p1_pos: (0, 0),
            p2_pos: (MAP_SIZE - 1, MAP_SIZE - 1),
            p1_hp: MAX_HP,
            p2_hp: MAX_HP,
            p1_energy: MAX_ENERGY,
            p2_energy: MAX_ENERGY,
            step_count: 0,
            walls: vec![false; (MAP_SIZE * MAP_SIZE) as usize],
            rng,
            p1_attacks: 0,
            p2_attacks: 0,
        }
    }

    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.step_count = 0;
        self.p1_attacks = 0;
        self.p2_attacks = 0;
        
        // 1. Generate Walls
        // Randomly place 4-8 walls, ensuring symmetry
        let num_walls = self.rng.gen_range(4..9); 
        self.walls.fill(false);
        
        let mut count = 0;
        while count < num_walls {
            let wx = self.rng.gen_range(0..MAP_SIZE);
            let wy = self.rng.gen_range(0..MAP_SIZE);
            // Symmetry: (x, y) and (W-1-x, H-1-y)
            let sym_wx = MAP_SIZE - 1 - wx;
            let sym_wy = MAP_SIZE - 1 - wy;
            
            // Don't block corners (spawn areas approx)
            if (wx + wy) < 3 || (sym_wx + sym_wy) < 3 { continue; } // Top-left
            if (wx + wy) > (2 * MAP_SIZE - 4) { continue; } // Bottom-right
            
            let idx1 = (wy * MAP_SIZE + wx) as usize;
            let idx2 = (sym_wy * MAP_SIZE + sym_wx) as usize;
            
            if !self.walls[idx1] {
                self.walls[idx1] = true;
                self.walls[idx2] = true; // Symmetric
                count += 1; // Count pairs or single walls? Just approx.
            }
        }

        // 2. Spawn Players
        self.p1_pos = (self.rng.gen_range(0..MAP_SIZE), self.rng.gen_range(0..MAP_SIZE));
        self.p2_pos = (self.rng.gen_range(0..MAP_SIZE), self.rng.gen_range(0..MAP_SIZE));
        
        // Retry if invalid
        while self.p1_pos == self.p2_pos || self.is_wall(self.p1_pos) || self.is_wall(self.p2_pos) {
             self.p1_pos = (self.rng.gen_range(0..MAP_SIZE), self.rng.gen_range(0..MAP_SIZE));
             self.p2_pos = (self.rng.gen_range(0..MAP_SIZE), self.rng.gen_range(0..MAP_SIZE));
        }
        
        self.p1_hp = MAX_HP;
        self.p2_hp = MAX_HP;
        self.p1_energy = MAX_ENERGY;
        self.p2_energy = MAX_ENERGY;

        (
            self.get_obs(false),
            self.get_obs(true),
            self.get_mask(false),
            self.get_mask(true),
        )
    }

    fn step(&mut self, action_p1: usize, action_p2: usize) 
        -> (Vec<f32>, Vec<f32>, f32, f32, bool, Vec<f32>, Vec<f32>, HashMap<String, f32>) 
    {
        self.step_count += 1;

        // 1. 转换动作为物理动作
        let phys_act_p1 = Self::transform_action(action_p1, false);
        let phys_act_p2 = Self::transform_action(action_p2, true);

        // 2. 消耗能量 & 移动
        // P1
        let mut cost_p1 = 0;
        match phys_act_p1 {
            ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => cost_p1 = COST_MOVE,
            ACT_ATTACK => cost_p1 = COST_ATTACK,
            ACT_SHOOT => cost_p1 = COST_SHOOT,
            _ => {},
        }
        if self.p1_energy >= cost_p1 {
            self.p1_energy -= cost_p1;
            self.apply_move(false, phys_act_p1);
        }

        // P2
        let mut cost_p2 = 0;
        match phys_act_p2 {
            ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => cost_p2 = COST_MOVE,
            ACT_ATTACK => cost_p2 = COST_ATTACK,
            ACT_SHOOT => cost_p2 = COST_SHOOT,
            _ => {},
        }
        if self.p2_energy >= cost_p2 {
            self.p2_energy -= cost_p2;
            self.apply_move(true, phys_act_p2);
        }

        // 3. 攻击判定
        let mut r1 = 0.0;
        let mut r2 = 0.0;

        // P1 Action
        if phys_act_p1 == ACT_ATTACK && cost_p1 == COST_ATTACK { 
             // Melee: Check adjacency + Line of Sight (Walls block melee?)
             // Let's say Walls block Melee too.
             if self.check_hit(self.p1_pos, self.p2_pos, ATTACK_RANGE) {
                 self.p2_hp -= 1;
                 r1 += 1.0;
                 r2 -= 1.0;
                 self.p1_attacks += 1;
             }
        } else if phys_act_p1 == ACT_SHOOT && cost_p1 == COST_SHOOT {
             if self.check_hit(self.p1_pos, self.p2_pos, SHOOT_RANGE) {
                 self.p2_hp -= 1; 
                 r1 += 1.0;
                 r2 -= 1.0;
                 self.p1_attacks += 1;
             }
        }

        // P2 Action
        if phys_act_p2 == ACT_ATTACK && cost_p2 == COST_ATTACK {
             if self.check_hit(self.p2_pos, self.p1_pos, ATTACK_RANGE) {
                 self.p1_hp -= 1;
                 r2 += 1.0;
                 r1 -= 1.0;
                 self.p2_attacks += 1;
             }
        } else if phys_act_p2 == ACT_SHOOT && cost_p2 == COST_SHOOT {
             if self.check_hit(self.p2_pos, self.p1_pos, SHOOT_RANGE) {
                 self.p1_hp -= 1;
                 r2 += 1.0;
                 r1 -= 1.0;
                 self.p2_attacks += 1;
             }
        }
        
        // 4. 能量恢复 (Sudden Death: Stop regen)
        let regen = if self.step_count > SUDDEN_DEATH_STEP { 0 } else { REGEN_ENERGY };
        
        self.p1_energy = (self.p1_energy + regen).min(MAX_ENERGY);
        self.p2_energy = (self.p2_energy + regen).min(MAX_ENERGY);

        // 5. 结束判定
        let done = self.p1_hp <= 0 || self.p2_hp <= 0 || self.step_count >= MAX_STEPS;
        
        let mut info = HashMap::new();
        
        if done {
            if self.p1_hp > self.p2_hp {
                r1 += 5.0; r2 -= 5.0;
                info.insert("p1_win".to_string(), 1.0);
                info.insert("p2_win".to_string(), 0.0);
                info.insert("draw".to_string(), 0.0);
            } else if self.p2_hp > self.p1_hp {
                r2 += 5.0; r1 -= 5.0;
                info.insert("p1_win".to_string(), 0.0);
                info.insert("p2_win".to_string(), 1.0);
                info.insert("draw".to_string(), 0.0);
            } else {
                info.insert("p1_win".to_string(), 0.0);
                info.insert("p2_win".to_string(), 0.0);
                info.insert("draw".to_string(), 1.0);
            }
            
            info.insert("p1_attacks".to_string(), self.p1_attacks as f32);
            info.insert("p2_attacks".to_string(), self.p2_attacks as f32);
            info.insert("steps".to_string(), self.step_count as f32);
        }

        (
            self.get_obs(false),
            self.get_obs(true),
            r1,
            r2,
            done,
            self.get_mask(false),
            self.get_mask(true),
            info,
        )
    }

    fn obs_dim() -> usize { OBS_DIM }
    fn action_dim() -> usize { ACTION_DIM }
}

// ============================================================================
// 3. VectorizedEnv PyClass
// ============================================================================

#[pyclass]
struct VectorizedEnv {
    envs: Vec<SimpleDuel>,
}

#[pymethods]
impl VectorizedEnv {
    #[new]
    fn new(num_envs: usize) -> Self {
        let mut envs = Vec::with_capacity(num_envs);
        for _ in 0..num_envs {
            envs.push(SimpleDuel::new());
        }
        VectorizedEnv { envs }
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
        let n = self.envs.len();
        let obs_dim = SimpleDuel::obs_dim();
        let act_dim = SimpleDuel::action_dim();

        // 预分配内存
        // Obs Batch: [2 * N, Obs_Dim] (P1 前 N 个，P2 后 N 个)
        // Mask Batch: [2 * N, Act_Dim]
        
        // 我们需要收集所有环境的结果
        // 使用 Rayon 并行处理
        let results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = self.envs.par_iter_mut()
            .map(|env| env.reset())
            .collect();

        // 拼装数据
        let mut obs_batch = vec![0.0; 2 * n * obs_dim];
        let mut mask_batch = vec![0.0; 2 * n * act_dim];

        for (i, (o1, o2, m1, m2)) in results.into_iter().enumerate() {
            // P1 数据放在 i
            // P2 数据放在 n + i
            
            // Obs
            let p1_start = i * obs_dim;
            let p2_start = (n + i) * obs_dim;
            obs_batch[p1_start..p1_start+obs_dim].copy_from_slice(&o1);
            obs_batch[p2_start..p2_start+obs_dim].copy_from_slice(&o2);
            
            // Mask
            let m1_start = i * act_dim;
            let m2_start = (n + i) * act_dim;
            mask_batch[m1_start..m1_start+act_dim].copy_from_slice(&m1);
            mask_batch[m2_start..m2_start+act_dim].copy_from_slice(&m2);
        }

        let py_obs = PyArray1::from_vec(py, obs_batch).reshape((2 * n, obs_dim)).unwrap();
        let py_mask = PyArray1::from_vec(py, mask_batch).reshape((2 * n, act_dim)).unwrap();
        
        (py_obs, py_mask)
    }

    fn step<'py>(&mut self, py: Python<'py>, actions_p1: Vec<usize>, actions_p2: Vec<usize>) 
        -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyList>) 
    {
        let n = self.envs.len();
        let obs_dim = SimpleDuel::obs_dim();
        let act_dim = SimpleDuel::action_dim();
        
        assert_eq!(actions_p1.len(), n);
        assert_eq!(actions_p2.len(), n);

        // Rayon 并行 Step
        let results: Vec<(Vec<f32>, Vec<f32>, f32, f32, bool, Vec<f32>, Vec<f32>, HashMap<String, f32>)> = self.envs.par_iter_mut()
            .zip(actions_p1.par_iter().zip(actions_p2.par_iter()))
            .map(|(env, (&a1, &a2))| {
                let (o1, o2, r1, r2, d, m1, m2, info) = env.step(a1, a2);
                if d {
                    // 自动 reset
                    let (new_o1, new_o2, new_m1, new_m2) = env.reset();
                    // 返回 reset 后的 obs 和 mask，但 reward 和 done 保持
                    // 注意：info 也是这一步产生的，所以要保留
                    (new_o1, new_o2, r1, r2, true, new_m1, new_m2, info)
                } else {
                    (o1, o2, r1, r2, false, m1, m2, info)
                }
            })
            .collect();

        // 拼装
        let mut obs_batch = vec![0.0; 2 * n * obs_dim];
        let mut reward_batch = vec![0.0; 2 * n];
        let mut done_batch = vec![false; n]; // 只需要 N 个，因为环境同时结束
        let mut mask_batch = vec![0.0; 2 * n * act_dim];
        
        // Info 需要转换为 Python 对象，这必须在 GIL 持有下串行进行
        // 我们创建一个 list，包含 n 个 dict
        let py_info_list = PyList::empty(py);

        for (i, (o1, o2, r1, r2, d, m1, m2, info)) in results.into_iter().enumerate() {
            // Obs
            let p1_obs_idx = i * obs_dim;
            let p2_obs_idx = (n + i) * obs_dim;
            obs_batch[p1_obs_idx..p1_obs_idx+obs_dim].copy_from_slice(&o1);
            obs_batch[p2_obs_idx..p2_obs_idx+obs_dim].copy_from_slice(&o2);
            
            // Reward
            reward_batch[i] = r1;
            reward_batch[n + i] = r2;
            
            // Done
            done_batch[i] = d;
            
            // Mask
            let p1_mask_idx = i * act_dim;
            let p2_mask_idx = (n + i) * act_dim;
            mask_batch[p1_mask_idx..p1_mask_idx+act_dim].copy_from_slice(&m1);
            mask_batch[p2_mask_idx..p2_mask_idx+act_dim].copy_from_slice(&m2);
            
            // Info
            let py_dict = PyDict::new(py);
            for (k, v) in info {
                py_dict.set_item(k, v).unwrap();
            }
            py_info_list.append(py_dict).unwrap();
        }

        let py_obs = PyArray1::from_vec(py, obs_batch).reshape((2 * n, obs_dim)).unwrap();
        let py_reward = PyArray1::from_vec(py, reward_batch); // [2*N]
        let py_done = PyArray1::from_vec(py, done_batch); // [N]
        let py_mask = PyArray1::from_vec(py, mask_batch).reshape((2 * n, act_dim)).unwrap();

        (py_obs, py_reward, py_done, py_mask, py_info_list)
    }

    fn obs_dim(&self) -> usize { SimpleDuel::obs_dim() }
    fn action_dim(&self) -> usize { SimpleDuel::action_dim() }
}

#[pymodule]
fn high_perf_env(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VectorizedEnv>()?;
    Ok(())
}
