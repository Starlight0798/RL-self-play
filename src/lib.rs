use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::LazyLock;

// ============================================================================
// Game Registry - Type-erased game factory system
// ============================================================================

/// Factory function type for creating game instances
type GameFactory = fn() -> GameEnvDispatch;

/// Game registry: maps game name to (factory, obs_dim, action_dim)
static GAME_REGISTRY: LazyLock<HashMap<&'static str, (GameFactory, usize, usize)>> =
    LazyLock::new(|| {
        let mut registry = HashMap::new();
        registry.insert(
            "simple_duel",
            (
                (|| GameEnvDispatch::SimpleDuel(<SimpleDuel as GameEnv>::new())) as GameFactory,
                OBS_DIM,
                ACTION_DIM,
            ),
        );
        registry.insert(
            "tictactoe",
            (
                (|| GameEnvDispatch::TicTacToe(<TicTacToe as GameEnv>::new())) as GameFactory,
                TICTACTOE_OBS_DIM,
                TICTACTOE_ACTION_DIM,
            ),
        );
        registry
    });

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
    fn step(
        &mut self,
        action_p1: usize,
        action_p2: usize,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        f32,
        f32,
        bool,
        Vec<f32>,
        Vec<f32>,
        HashMap<String, f32>,
    );

    fn obs_dim() -> usize;
    fn action_dim() -> usize;
}

/// Zero-copy game environment interface for high-performance vectorized execution.
/// Implementations write directly into pre-allocated buffers.
pub trait GameEnvZeroCopy: Send + Sync + Clone {
    fn new() -> Self;

    /// Reset and write observations/masks into provided buffers
    fn reset_into(
        &mut self,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    );

    /// Step and write results into provided buffers
    /// Returns (reward_p1, reward_p2, done, info)
    fn step_into(
        &mut self,
        action_p1: usize,
        action_p2: usize,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) -> (f32, f32, bool, GameInfo);

    fn obs_dim() -> usize;
    fn action_dim() -> usize;
}

// ============================================================================
// 2. SimpleDuel 实现 - 升级版 12x12 战术游戏
// ============================================================================

const MAP_SIZE: i32 = 12;
const MAX_HP: i32 = 4;
const MAX_ENERGY: i32 = 7;
const MAX_SHIELD: i32 = 2;
const MAX_AMMO: i32 = 6;
const REGEN_ENERGY: i32 = 1;

// 动作能量消耗
const COST_MOVE: i32 = 1;
const COST_ATTACK: i32 = 2;
const COST_SHOOT: i32 = 3;
const COST_DODGE: i32 = 2;
const COST_SHIELD: i32 = 3;
const COST_DASH: i32 = 3;
const COST_AOE: i32 = 4;
const COST_HEAL: i32 = 4;

// 攻击范围
const ATTACK_RANGE: i32 = 1;
const SHOOT_RANGE: i32 = 4; // 增加远程射击范围
const AOE_RANGE: i32 = 1;

// 游戏时间
const MAX_STEPS: i32 = 60; // 增加最大步数
const SUDDEN_DEATH_STEP: i32 = 40; // 突然死亡阶段开始时间

// 冷却时间
const HEAL_COOLDOWN: i32 = 5;

// 动作定义
// 0: Stay
// 1: Up (y+)
// 2: Down (y-)
// 3: Left (x-)
// 4: Right (x+)
// 5: Attack (Melee)
// 6: Shoot (Ranged)
// 7: Dodge - 2能量，下回合免疫1次攻击
// 8: Shield - 3能量，获得1层护盾
// 9: Dash - 3能量，向远离敌人方向移动2格
// 10: AOE - 4能量，对周围1格内敌人造成伤害
// 11: Heal - 4能量，恢复1HP（5步冷却）
// 12: Reload - 0能量，恢复3发弹药
const ACT_STAY: usize = 0;
const ACT_UP: usize = 1;
const ACT_DOWN: usize = 2;
const ACT_LEFT: usize = 3;
const ACT_RIGHT: usize = 4;
const ACT_ATTACK: usize = 5;
const ACT_SHOOT: usize = 6;
const ACT_DODGE: usize = 7;
const ACT_SHIELD: usize = 8;
const ACT_DASH: usize = 9;
const ACT_AOE: usize = 10;
const ACT_HEAL: usize = 11;
const ACT_RELOAD: usize = 12;

const ACTION_DIM: usize = 13;
// [0-7]: mx, my, ex, ey, mhp, ehp, meng, eeng (归一化)
// [8-11]: my_shield, enemy_shield, my_ammo, enemy_ammo
// [12-15]: my_dodge, enemy_dodge, heal_cd, step_progress
// [16-159]: 12x12 地形网格
const OBS_DIM: usize = 16 + 144; // = 160

// 地形类型
const TERRAIN_EMPTY: u8 = 0;
const TERRAIN_WALL: u8 = 1;
const TERRAIN_WATER: u8 = 2;
const TERRAIN_HIGH_GROUND: u8 = 3;

// 道具类型
const ITEM_NONE: u8 = 0;
const ITEM_HEALTH: u8 = 1;
const ITEM_ENERGY: u8 = 2;
const ITEM_AMMO: u8 = 3;
const ITEM_SHIELD: u8 = 4;

// 道具刷新时间
const ITEM_RESPAWN_TIME: i32 = 10;

/// Typed game info struct - replaces HashMap<String, f32> for zero allocation
#[derive(Clone, Default, Copy)]
pub struct GameInfo {
    pub p1_win: f32,
    pub p2_win: f32,
    pub draw: f32,
    pub p1_attacks: f32,
    pub p2_attacks: f32,
    pub p1_damage: f32,
    pub p2_damage: f32,
    pub steps: f32,
    pub is_terminal: bool,
}

impl GameInfo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn terminal(
        p1_win: bool,
        p2_win: bool,
        draw: bool,
        p1_attacks: i32,
        p2_attacks: i32,
        p1_damage: i32,
        p2_damage: i32,
        steps: i32,
    ) -> Self {
        Self {
            p1_win: if p1_win { 1.0 } else { 0.0 },
            p2_win: if p2_win { 1.0 } else { 0.0 },
            draw: if draw { 1.0 } else { 0.0 },
            p1_attacks: p1_attacks as f32,
            p2_attacks: p2_attacks as f32,
            p1_damage: p1_damage as f32,
            p2_damage: p2_damage as f32,
            steps: steps as f32,
            is_terminal: true,
        }
    }
}

#[derive(Clone)]
struct SimpleDuel {
    p1_pos: (i32, i32),
    p2_pos: (i32, i32),
    p1_hp: i32,
    p2_hp: i32,
    p1_energy: i32,
    p2_energy: i32,
    p1_shield: i32,
    p2_shield: i32,
    p1_ammo: i32,
    p2_ammo: i32,
    p1_dodge_active: bool,
    p2_dodge_active: bool,
    p1_heal_cooldown: i32,
    p2_heal_cooldown: i32,
    step_count: i32,
    terrain: Vec<u8>,       // 144格地形
    items: Vec<u8>,         // 144格道具
    item_respawn: Vec<i32>, // 道具刷新倒计时
    rng: StdRng,
    // 统计信息
    p1_attacks: i32,
    p2_attacks: i32,
    p1_damage_dealt: i32,
    p2_damage_dealt: i32,
}

impl SimpleDuel {
    // 辅助函数：判断是否在地图内
    fn is_valid(pos: (i32, i32)) -> bool {
        pos.0 >= 0 && pos.0 < MAP_SIZE && pos.1 >= 0 && pos.1 < MAP_SIZE
    }

    fn pos_to_idx(pos: (i32, i32)) -> usize {
        (pos.1 * MAP_SIZE + pos.0) as usize
    }

    fn idx_to_pos(idx: usize) -> (i32, i32) {
        let x = (idx as i32) % MAP_SIZE;
        let y = (idx as i32) / MAP_SIZE;
        (x, y)
    }

    fn get_terrain(&self, pos: (i32, i32)) -> u8 {
        if !Self::is_valid(pos) {
            return TERRAIN_WALL;
        }
        self.terrain[Self::pos_to_idx(pos)]
    }

    fn is_wall(&self, pos: (i32, i32)) -> bool {
        self.get_terrain(pos) == TERRAIN_WALL
    }

    fn is_water(&self, pos: (i32, i32)) -> bool {
        self.get_terrain(pos) == TERRAIN_WATER
    }

    fn is_high_ground(&self, pos: (i32, i32)) -> bool {
        self.get_terrain(pos) == TERRAIN_HIGH_GROUND
    }

    // 生成对称地形
    fn generate_terrain(&mut self) {
        self.terrain.fill(TERRAIN_EMPTY);
        self.items.fill(ITEM_NONE);
        self.item_respawn.fill(0);

        // 生成墙体 (6-10个)
        let num_walls = self.rng.gen_range(6..11);
        let mut count = 0;
        while count < num_walls {
            let wx = self.rng.gen_range(0..MAP_SIZE);
            let wy = self.rng.gen_range(0..MAP_SIZE);
            let sym_wx = MAP_SIZE - 1 - wx;
            let sym_wy = MAP_SIZE - 1 - wy;

            // 不阻挡角落出生区域
            if (wx + wy) < 4 || (sym_wx + sym_wy) < 4 {
                continue;
            }
            if (wx + wy) > (2 * MAP_SIZE - 5) {
                continue;
            }

            let idx1 = Self::pos_to_idx((wx, wy));
            let idx2 = Self::pos_to_idx((sym_wx, sym_wy));

            if self.terrain[idx1] == TERRAIN_EMPTY {
                self.terrain[idx1] = TERRAIN_WALL;
                self.terrain[idx2] = TERRAIN_WALL;
                count += 1;
            }
        }

        // 生成水域 (2-4个)
        let num_water = self.rng.gen_range(2..5);
        count = 0;
        while count < num_water {
            let wx = self.rng.gen_range(1..MAP_SIZE - 1);
            let wy = self.rng.gen_range(1..MAP_SIZE - 1);
            let sym_wx = MAP_SIZE - 1 - wx;
            let sym_wy = MAP_SIZE - 1 - wy;

            // 不阻挡角落
            if (wx + wy) < 4 || (sym_wx + sym_wy) < 4 {
                continue;
            }

            let idx1 = Self::pos_to_idx((wx, wy));
            let idx2 = Self::pos_to_idx((sym_wx, sym_wy));

            if self.terrain[idx1] == TERRAIN_EMPTY {
                self.terrain[idx1] = TERRAIN_WATER;
                self.terrain[idx2] = TERRAIN_WATER;
                count += 1;
            }
        }

        // 生成高地 (2-3个)
        let num_high = self.rng.gen_range(2..4);
        count = 0;
        while count < num_high {
            let hx = self.rng.gen_range(2..MAP_SIZE - 2);
            let hy = self.rng.gen_range(2..MAP_SIZE - 2);
            let sym_hx = MAP_SIZE - 1 - hx;
            let sym_hy = MAP_SIZE - 1 - hy;

            let idx1 = Self::pos_to_idx((hx, hy));
            let idx2 = Self::pos_to_idx((sym_hx, sym_hy));

            if self.terrain[idx1] == TERRAIN_EMPTY {
                self.terrain[idx1] = TERRAIN_HIGH_GROUND;
                self.terrain[idx2] = TERRAIN_HIGH_GROUND;
                count += 1;
            }
        }

        // 生成道具 (2-3个)
        let num_items = self.rng.gen_range(2..4);
        count = 0;
        while count < num_items {
            let ix = self.rng.gen_range(2..MAP_SIZE - 2);
            let iy = self.rng.gen_range(2..MAP_SIZE - 2);
            let sym_ix = MAP_SIZE - 1 - ix;
            let sym_iy = MAP_SIZE - 1 - iy;

            let idx1 = Self::pos_to_idx((ix, iy));
            let idx2 = Self::pos_to_idx((sym_ix, sym_iy));

            // 不放在墙上或水中
            if self.terrain[idx1] != TERRAIN_EMPTY && self.terrain[idx1] != TERRAIN_HIGH_GROUND {
                continue;
            }
            if self.items[idx1] != ITEM_NONE {
                continue;
            }

            // 随机选择道具类型
            let item_type = match self.rng.gen_range(0..4) {
                0 => ITEM_HEALTH,
                1 => ITEM_ENERGY,
                2 => ITEM_AMMO,
                _ => ITEM_SHIELD,
            };

            self.items[idx1] = item_type;
            self.items[idx2] = item_type;
            count += 1;
        }
    }

    // 辅助函数：生成 Observation
    fn get_obs(&self, is_p2_perspective: bool) -> Vec<f32> {
        let (
            my_pos,
            enemy_pos,
            my_hp,
            enemy_hp,
            my_eng,
            enemy_eng,
            my_shield,
            enemy_shield,
            my_ammo,
            enemy_ammo,
            my_dodge,
            enemy_dodge,
            my_heal_cd,
        ) = if is_p2_perspective {
            (
                self.p2_pos,
                self.p1_pos,
                self.p2_hp,
                self.p1_hp,
                self.p2_energy,
                self.p1_energy,
                self.p2_shield,
                self.p1_shield,
                self.p2_ammo,
                self.p1_ammo,
                self.p2_dodge_active,
                self.p1_dodge_active,
                self.p2_heal_cooldown,
            )
        } else {
            (
                self.p1_pos,
                self.p2_pos,
                self.p1_hp,
                self.p2_hp,
                self.p1_energy,
                self.p2_energy,
                self.p1_shield,
                self.p2_shield,
                self.p1_ammo,
                self.p2_ammo,
                self.p1_dodge_active,
                self.p2_dodge_active,
                self.p1_heal_cooldown,
            )
        };

        let (mx, my) = self.transform_pos(my_pos, is_p2_perspective);
        let (ex, ey) = self.transform_pos(enemy_pos, is_p2_perspective);

        // 归一化到 0-1
        let scale = (MAP_SIZE - 1) as f32;
        let hp_scale = MAX_HP as f32;
        let eng_scale = MAX_ENERGY as f32;
        let shield_scale = MAX_SHIELD as f32;
        let ammo_scale = MAX_AMMO as f32;

        let mut obs = vec![
            // [0-7]: 位置和基础属性
            mx as f32 / scale,
            my as f32 / scale,
            ex as f32 / scale,
            ey as f32 / scale,
            my_hp as f32 / hp_scale,
            enemy_hp as f32 / hp_scale,
            my_eng as f32 / eng_scale,
            enemy_eng as f32 / eng_scale,
            // [8-11]: 护盾和弹药
            my_shield as f32 / shield_scale,
            enemy_shield as f32 / shield_scale,
            my_ammo as f32 / ammo_scale,
            enemy_ammo as f32 / ammo_scale,
            // [12-15]: 闪避、冷却、进度
            if my_dodge { 1.0 } else { 0.0 },
            if enemy_dodge { 1.0 } else { 0.0 },
            my_heal_cd as f32 / HEAL_COOLDOWN as f32,
            self.step_count as f32 / MAX_STEPS as f32,
        ];

        // [16-159]: 12x12 地形网格
        // 编码: 0=空地, 0.25=墙, 0.5=水域, 0.75=高地, 1.0=道具
        // 为了更丰富的信息，我们使用多个通道的概念但压缩到单个值
        // 地形值 + 道具偏移
        for y in 0..MAP_SIZE {
            for x in 0..MAP_SIZE {
                let (px, py) = self.transform_pos((x, y), is_p2_perspective);
                let idx = Self::pos_to_idx((px, py));

                let terrain_val = match self.terrain[idx] {
                    TERRAIN_EMPTY => 0.0,
                    TERRAIN_WALL => 0.25,
                    TERRAIN_WATER => 0.5,
                    TERRAIN_HIGH_GROUND => 0.75,
                    _ => 0.0,
                };

                // 如果有道具，添加一个小偏移表示
                let item_val = if self.items[idx] != ITEM_NONE {
                    0.1 * (self.items[idx] as f32)
                } else {
                    0.0
                };

                obs.push(terrain_val + item_val);
            }
        }

        obs
    }

    // 坐标转换 - 180度旋转
    fn transform_pos(&self, pos: (i32, i32), invert: bool) -> (i32, i32) {
        if invert {
            (MAP_SIZE - 1 - pos.0, MAP_SIZE - 1 - pos.1)
        } else {
            pos
        }
    }

    // 动作转换
    fn transform_action(action: usize, invert: bool) -> usize {
        if !invert {
            return action;
        }
        match action {
            ACT_UP => ACT_DOWN,
            ACT_DOWN => ACT_UP,
            ACT_LEFT => ACT_RIGHT,
            ACT_RIGHT => ACT_LEFT,
            // 非方向性动作不变
            _ => action,
        }
    }

    // 获取 Action Mask
    fn get_mask(&self, is_p2_perspective: bool) -> Vec<f32> {
        let mut mask = vec![0.0; ACTION_DIM];

        let (my_pos_phys, enemy_pos_phys, my_energy, my_hp, my_shield, my_ammo, my_heal_cd) =
            if is_p2_perspective {
                (
                    self.p2_pos,
                    self.p1_pos,
                    self.p2_energy,
                    self.p2_hp,
                    self.p2_shield,
                    self.p2_ammo,
                    self.p2_heal_cooldown,
                )
            } else {
                (
                    self.p1_pos,
                    self.p2_pos,
                    self.p1_energy,
                    self.p1_hp,
                    self.p1_shield,
                    self.p1_ammo,
                    self.p1_heal_cooldown,
                )
            };

        // 计算移动到水域的额外消耗
        let water_cost = |target: (i32, i32)| -> i32 {
            if self.is_water(target) {
                1
            } else {
                0
            }
        };

        // 遍历所有逻辑动作
        for act in 0..ACTION_DIM {
            let phys_act = Self::transform_action(act, is_p2_perspective);
            let is_legal = match phys_act {
                ACT_STAY => true,
                ACT_UP => {
                    let target = (my_pos_phys.0, my_pos_phys.1 + 1);
                    let cost = COST_MOVE + water_cost(target);
                    my_pos_phys.1 < MAP_SIZE - 1 && my_energy >= cost && !self.is_wall(target)
                }
                ACT_DOWN => {
                    let target = (my_pos_phys.0, my_pos_phys.1 - 1);
                    let cost = COST_MOVE + water_cost(target);
                    my_pos_phys.1 > 0 && my_energy >= cost && !self.is_wall(target)
                }
                ACT_LEFT => {
                    let target = (my_pos_phys.0 - 1, my_pos_phys.1);
                    let cost = COST_MOVE + water_cost(target);
                    my_pos_phys.0 > 0 && my_energy >= cost && !self.is_wall(target)
                }
                ACT_RIGHT => {
                    let target = (my_pos_phys.0 + 1, my_pos_phys.1);
                    let cost = COST_MOVE + water_cost(target);
                    my_pos_phys.0 < MAP_SIZE - 1 && my_energy >= cost && !self.is_wall(target)
                }
                ACT_ATTACK => {
                    my_energy >= COST_ATTACK
                        && self.check_hit(my_pos_phys, enemy_pos_phys, ATTACK_RANGE)
                }
                ACT_SHOOT => {
                    // 射击需要弹药且不被水域阻挡
                    my_energy >= COST_SHOOT
                        && my_ammo > 0
                        && self.check_ranged_hit(my_pos_phys, enemy_pos_phys, SHOOT_RANGE)
                }
                ACT_DODGE => my_energy >= COST_DODGE,
                ACT_SHIELD => my_energy >= COST_SHIELD && my_shield < MAX_SHIELD,
                ACT_DASH => {
                    // Dash需要有空间可以逃跑
                    my_energy >= COST_DASH && self.can_dash(my_pos_phys, enemy_pos_phys)
                }
                ACT_AOE => {
                    my_energy >= COST_AOE && self.check_hit(my_pos_phys, enemy_pos_phys, AOE_RANGE)
                }
                ACT_HEAL => my_energy >= COST_HEAL && my_heal_cd == 0 && my_hp < MAX_HP,
                ACT_RELOAD => my_ammo < MAX_AMMO,
                _ => false,
            };

            if is_legal {
                mask[act] = 1.0;
            }
        }
        mask
    }

    fn can_dash(&self, my_pos: (i32, i32), enemy_pos: (i32, i32)) -> bool {
        // 计算远离敌人的方向
        let dx = my_pos.0 - enemy_pos.0;
        let dy = my_pos.1 - enemy_pos.1;

        // 选择主要逃跑方向
        let (move_dx, move_dy) = if dx.abs() >= dy.abs() {
            (if dx >= 0 { 1 } else { -1 }, 0)
        } else {
            (0, if dy >= 0 { 1 } else { -1 })
        };

        // 检查能否移动2格
        let mid = (my_pos.0 + move_dx, my_pos.1 + move_dy);
        let target = (my_pos.0 + 2 * move_dx, my_pos.1 + 2 * move_dy);

        Self::is_valid(mid) && Self::is_valid(target) && !self.is_wall(mid) && !self.is_wall(target)
    }

    fn apply_move(&mut self, is_p2: bool, phys_action: usize) -> i32 {
        let pos = if is_p2 { self.p2_pos } else { self.p1_pos };
        let mut new_pos = pos;

        match phys_action {
            ACT_UP => new_pos.1 = (pos.1 + 1).min(MAP_SIZE - 1),
            ACT_DOWN => new_pos.1 = (pos.1 - 1).max(0),
            ACT_LEFT => new_pos.0 = (pos.0 - 1).max(0),
            ACT_RIGHT => new_pos.0 = (pos.0 + 1).min(MAP_SIZE - 1),
            _ => {}
        }

        let mut extra_cost = 0;
        if !self.is_wall(new_pos) {
            // 水域额外消耗
            if self.is_water(new_pos) {
                extra_cost = 1;
            }
            if is_p2 {
                self.p2_pos = new_pos;
            } else {
                self.p1_pos = new_pos;
            }
        }
        extra_cost
    }

    fn apply_dash(&mut self, is_p2: bool) {
        let (my_pos, enemy_pos) = if is_p2 {
            (self.p2_pos, self.p1_pos)
        } else {
            (self.p1_pos, self.p2_pos)
        };

        // 计算远离敌人的方向
        let dx = my_pos.0 - enemy_pos.0;
        let dy = my_pos.1 - enemy_pos.1;

        let (move_dx, move_dy) = if dx.abs() >= dy.abs() {
            (if dx >= 0 { 1 } else { -1 }, 0)
        } else {
            (0, if dy >= 0 { 1 } else { -1 })
        };

        // 移动2格
        let mid = (my_pos.0 + move_dx, my_pos.1 + move_dy);
        let target = (my_pos.0 + 2 * move_dx, my_pos.1 + 2 * move_dy);

        let final_pos = if Self::is_valid(target) && !self.is_wall(target) {
            target
        } else if Self::is_valid(mid) && !self.is_wall(mid) {
            mid
        } else {
            my_pos
        };

        if is_p2 {
            self.p2_pos = final_pos;
        } else {
            self.p1_pos = final_pos;
        }
    }

    // 射线检测（用于近战和AOE）
    fn has_line_of_sight(&self, p1: (i32, i32), p2: (i32, i32)) -> bool {
        let dx = (p2.0 - p1.0).abs();
        let dy = (p2.1 - p1.1).abs();
        let sx = if p1.0 < p2.0 { 1 } else { -1 };
        let sy = if p1.1 < p2.1 { 1 } else { -1 };
        let mut err = dx - dy;
        let mut x = p1.0;
        let mut y = p1.1;

        while (x, y) != p2 {
            if self.is_wall((x, y)) && (x, y) != p1 {
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

            // Diagonal check
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

    // 远程攻击视线检查（墙和水域都阻挡）
    fn has_ranged_line_of_sight(&self, p1: (i32, i32), p2: (i32, i32)) -> bool {
        let dx = (p2.0 - p1.0).abs();
        let dy = (p2.1 - p1.1).abs();
        let sx = if p1.0 < p2.0 { 1 } else { -1 };
        let sy = if p1.1 < p2.1 { 1 } else { -1 };
        let mut err = dx - dy;
        let mut x = p1.0;
        let mut y = p1.1;

        while (x, y) != p2 {
            // 墙和水域都阻挡远程攻击
            if (x, y) != p1 && (self.is_wall((x, y)) || self.is_water((x, y))) {
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

            if next_x != x && next_y != y {
                let blocked1 = self.is_wall((next_x, y)) || self.is_water((next_x, y));
                let blocked2 = self.is_wall((x, next_y)) || self.is_water((x, next_y));
                if blocked1 && blocked2 {
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
            return self.has_line_of_sight(attacker_pos, target_pos);
        }
        false
    }

    fn check_ranged_hit(
        &self,
        attacker_pos: (i32, i32),
        target_pos: (i32, i32),
        range: i32,
    ) -> bool {
        let dx = (attacker_pos.0 - target_pos.0).abs();
        let dy = (attacker_pos.1 - target_pos.1).abs();

        // 高地额外射程
        let bonus_range = if self.is_high_ground(attacker_pos) {
            1
        } else {
            0
        };

        if dx <= range + bonus_range && dy <= range + bonus_range {
            return self.has_ranged_line_of_sight(attacker_pos, target_pos);
        }
        false
    }

    // 应用伤害（考虑闪避和护盾）
    fn apply_damage(&mut self, is_p2_target: bool, damage: i32, is_ranged: bool) -> bool {
        let (dodge_active, shield, high_ground) = if is_p2_target {
            (
                self.p2_dodge_active,
                self.p2_shield,
                self.is_high_ground(self.p2_pos),
            )
        } else {
            (
                self.p1_dodge_active,
                self.p1_shield,
                self.is_high_ground(self.p1_pos),
            )
        };

        // 闪避检查
        if dodge_active {
            // 闪避成功，清除闪避状态
            if is_p2_target {
                self.p2_dodge_active = false;
            } else {
                self.p1_dodge_active = false;
            }
            return false;
        }

        // 高地远程攻击减伤（50%几率闪避）
        if is_ranged && high_ground {
            if self.rng.gen_bool(0.5) {
                return false;
            }
        }

        let mut remaining_damage = damage;

        // 护盾吸收
        if shield > 0 {
            let absorbed = remaining_damage.min(shield);
            remaining_damage -= absorbed;
            if is_p2_target {
                self.p2_shield -= absorbed;
            } else {
                self.p1_shield -= absorbed;
            }
        }

        // 实际伤害
        if remaining_damage > 0 {
            if is_p2_target {
                self.p2_hp -= remaining_damage;
            } else {
                self.p1_hp -= remaining_damage;
            }
            return true;
        }

        false
    }

    // 道具拾取
    fn pickup_item(&mut self, is_p2: bool) {
        let pos = if is_p2 { self.p2_pos } else { self.p1_pos };
        let idx = Self::pos_to_idx(pos);

        match self.items[idx] {
            ITEM_HEALTH => {
                if is_p2 {
                    self.p2_hp = (self.p2_hp + 1).min(MAX_HP);
                } else {
                    self.p1_hp = (self.p1_hp + 1).min(MAX_HP);
                }
                self.items[idx] = ITEM_NONE;
                self.item_respawn[idx] = ITEM_RESPAWN_TIME;
            }
            ITEM_ENERGY => {
                if is_p2 {
                    self.p2_energy = (self.p2_energy + 3).min(MAX_ENERGY);
                } else {
                    self.p1_energy = (self.p1_energy + 3).min(MAX_ENERGY);
                }
                self.items[idx] = ITEM_NONE;
                self.item_respawn[idx] = ITEM_RESPAWN_TIME;
            }
            ITEM_AMMO => {
                if is_p2 {
                    self.p2_ammo = (self.p2_ammo + 2).min(MAX_AMMO);
                } else {
                    self.p1_ammo = (self.p1_ammo + 2).min(MAX_AMMO);
                }
                self.items[idx] = ITEM_NONE;
                self.item_respawn[idx] = ITEM_RESPAWN_TIME;
            }
            ITEM_SHIELD => {
                if is_p2 {
                    self.p2_shield = (self.p2_shield + 1).min(MAX_SHIELD);
                } else {
                    self.p1_shield = (self.p1_shield + 1).min(MAX_SHIELD);
                }
                self.items[idx] = ITEM_NONE;
                self.item_respawn[idx] = ITEM_RESPAWN_TIME;
            }
            _ => {}
        }
    }

    // 更新冷却和道具刷新
    fn update_cooldowns(&mut self) {
        if self.p1_heal_cooldown > 0 {
            self.p1_heal_cooldown -= 1;
        }
        if self.p2_heal_cooldown > 0 {
            self.p2_heal_cooldown -= 1;
        }

        // 道具刷新 - 需要记住原始道具类型
        for idx in 0..self.item_respawn.len() {
            if self.item_respawn[idx] > 0 {
                self.item_respawn[idx] -= 1;
                if self.item_respawn[idx] == 0 {
                    // 重新生成随机道具
                    let item_type = match self.rng.gen_range(0..4) {
                        0 => ITEM_HEALTH,
                        1 => ITEM_ENERGY,
                        2 => ITEM_AMMO,
                        _ => ITEM_SHIELD,
                    };
                    self.items[idx] = item_type;

                    // 对称位置也刷新
                    let pos = Self::idx_to_pos(idx);
                    let sym_pos = (MAP_SIZE - 1 - pos.0, MAP_SIZE - 1 - pos.1);
                    let sym_idx = Self::pos_to_idx(sym_pos);
                    if sym_idx != idx {
                        self.items[sym_idx] = item_type;
                        self.item_respawn[sym_idx] = 0;
                    }
                }
            }
        }
    }

    fn get_obs_into(&self, is_p2_perspective: bool, obs: &mut [f32]) {
        // Same logic as get_obs but writes to slice instead of Vec
        let (
            my_pos,
            enemy_pos,
            my_hp,
            enemy_hp,
            my_eng,
            enemy_eng,
            my_shield,
            enemy_shield,
            my_ammo,
            enemy_ammo,
            my_dodge,
            enemy_dodge,
            my_heal_cd,
        ) = if is_p2_perspective {
            (
                self.p2_pos,
                self.p1_pos,
                self.p2_hp,
                self.p1_hp,
                self.p2_energy,
                self.p1_energy,
                self.p2_shield,
                self.p1_shield,
                self.p2_ammo,
                self.p1_ammo,
                self.p2_dodge_active,
                self.p1_dodge_active,
                self.p2_heal_cooldown,
            )
        } else {
            (
                self.p1_pos,
                self.p2_pos,
                self.p1_hp,
                self.p2_hp,
                self.p1_energy,
                self.p2_energy,
                self.p1_shield,
                self.p2_shield,
                self.p1_ammo,
                self.p2_ammo,
                self.p1_dodge_active,
                self.p2_dodge_active,
                self.p1_heal_cooldown,
            )
        };

        let (mx, my) = self.transform_pos(my_pos, is_p2_perspective);
        let (ex, ey) = self.transform_pos(enemy_pos, is_p2_perspective);

        let scale = (MAP_SIZE - 1) as f32;
        let hp_scale = MAX_HP as f32;
        let eng_scale = MAX_ENERGY as f32;
        let shield_scale = MAX_SHIELD as f32;
        let ammo_scale = MAX_AMMO as f32;

        // Write scalar features [0-15]
        obs[0] = mx as f32 / scale;
        obs[1] = my as f32 / scale;
        obs[2] = ex as f32 / scale;
        obs[3] = ey as f32 / scale;
        obs[4] = my_hp as f32 / hp_scale;
        obs[5] = enemy_hp as f32 / hp_scale;
        obs[6] = my_eng as f32 / eng_scale;
        obs[7] = enemy_eng as f32 / eng_scale;
        obs[8] = my_shield as f32 / shield_scale;
        obs[9] = enemy_shield as f32 / shield_scale;
        obs[10] = my_ammo as f32 / ammo_scale;
        obs[11] = enemy_ammo as f32 / ammo_scale;
        obs[12] = if my_dodge { 1.0 } else { 0.0 };
        obs[13] = if enemy_dodge { 1.0 } else { 0.0 };
        obs[14] = my_heal_cd as f32 / HEAL_COOLDOWN as f32;
        obs[15] = self.step_count as f32 / MAX_STEPS as f32;

        // Write terrain grid [16-159]
        let mut idx = 16;
        for y in 0..MAP_SIZE {
            for x in 0..MAP_SIZE {
                let (px, py) = self.transform_pos((x, y), is_p2_perspective);
                let tidx = Self::pos_to_idx((px, py));

                let terrain_val = match self.terrain[tidx] {
                    TERRAIN_EMPTY => 0.0,
                    TERRAIN_WALL => 0.25,
                    TERRAIN_WATER => 0.5,
                    TERRAIN_HIGH_GROUND => 0.75,
                    _ => 0.0,
                };

                let item_val = if self.items[tidx] != ITEM_NONE {
                    0.1 * (self.items[tidx] as f32)
                } else {
                    0.0
                };

                obs[idx] = terrain_val + item_val;
                idx += 1;
            }
        }
    }

    fn get_mask_into(&self, is_p2_perspective: bool, mask: &mut [f32]) {
        // Same logic as get_mask but writes to slice
        let (my_pos_phys, enemy_pos_phys, my_energy, my_hp, my_shield, my_ammo, my_heal_cd) =
            if is_p2_perspective {
                (
                    self.p2_pos,
                    self.p1_pos,
                    self.p2_energy,
                    self.p2_hp,
                    self.p2_shield,
                    self.p2_ammo,
                    self.p2_heal_cooldown,
                )
            } else {
                (
                    self.p1_pos,
                    self.p2_pos,
                    self.p1_energy,
                    self.p1_hp,
                    self.p1_shield,
                    self.p1_ammo,
                    self.p1_heal_cooldown,
                )
            };

        let water_cost = |target: (i32, i32)| -> i32 {
            if self.is_water(target) {
                1
            } else {
                0
            }
        };

        for act in 0..ACTION_DIM {
            let phys_act = Self::transform_action(act, is_p2_perspective);
            let is_legal = match phys_act {
                ACT_STAY => true,
                ACT_UP => {
                    let target = (my_pos_phys.0, my_pos_phys.1 + 1);
                    let cost = COST_MOVE + water_cost(target);
                    my_pos_phys.1 < MAP_SIZE - 1 && my_energy >= cost && !self.is_wall(target)
                }
                ACT_DOWN => {
                    let target = (my_pos_phys.0, my_pos_phys.1 - 1);
                    let cost = COST_MOVE + water_cost(target);
                    my_pos_phys.1 > 0 && my_energy >= cost && !self.is_wall(target)
                }
                ACT_LEFT => {
                    let target = (my_pos_phys.0 - 1, my_pos_phys.1);
                    let cost = COST_MOVE + water_cost(target);
                    my_pos_phys.0 > 0 && my_energy >= cost && !self.is_wall(target)
                }
                ACT_RIGHT => {
                    let target = (my_pos_phys.0 + 1, my_pos_phys.1);
                    let cost = COST_MOVE + water_cost(target);
                    my_pos_phys.0 < MAP_SIZE - 1 && my_energy >= cost && !self.is_wall(target)
                }
                ACT_ATTACK => {
                    my_energy >= COST_ATTACK
                        && self.check_hit(my_pos_phys, enemy_pos_phys, ATTACK_RANGE)
                }
                ACT_SHOOT => {
                    my_energy >= COST_SHOOT
                        && my_ammo > 0
                        && self.check_ranged_hit(my_pos_phys, enemy_pos_phys, SHOOT_RANGE)
                }
                ACT_DODGE => my_energy >= COST_DODGE,
                ACT_SHIELD => my_energy >= COST_SHIELD && my_shield < MAX_SHIELD,
                ACT_DASH => my_energy >= COST_DASH && self.can_dash(my_pos_phys, enemy_pos_phys),
                ACT_AOE => {
                    my_energy >= COST_AOE && self.check_hit(my_pos_phys, enemy_pos_phys, AOE_RANGE)
                }
                ACT_HEAL => my_energy >= COST_HEAL && my_heal_cd == 0 && my_hp < MAX_HP,
                ACT_RELOAD => my_ammo < MAX_AMMO,
                _ => false,
            };

            mask[act] = if is_legal { 1.0 } else { 0.0 };
        }
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
            p1_shield: 0,
            p2_shield: 0,
            p1_ammo: MAX_AMMO,
            p2_ammo: MAX_AMMO,
            p1_dodge_active: false,
            p2_dodge_active: false,
            p1_heal_cooldown: 0,
            p2_heal_cooldown: 0,
            step_count: 0,
            terrain: vec![TERRAIN_EMPTY; (MAP_SIZE * MAP_SIZE) as usize],
            items: vec![ITEM_NONE; (MAP_SIZE * MAP_SIZE) as usize],
            item_respawn: vec![0; (MAP_SIZE * MAP_SIZE) as usize],
            rng,
            p1_attacks: 0,
            p2_attacks: 0,
            p1_damage_dealt: 0,
            p2_damage_dealt: 0,
        }
    }

    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.step_count = 0;
        self.p1_attacks = 0;
        self.p2_attacks = 0;
        self.p1_damage_dealt = 0;
        self.p2_damage_dealt = 0;
        self.p1_shield = 0;
        self.p2_shield = 0;
        self.p1_ammo = MAX_AMMO;
        self.p2_ammo = MAX_AMMO;
        self.p1_dodge_active = false;
        self.p2_dodge_active = false;
        self.p1_heal_cooldown = 0;
        self.p2_heal_cooldown = 0;

        // 生成地形
        self.generate_terrain();

        // 生成玩家位置
        self.p1_pos = (
            self.rng.gen_range(0..MAP_SIZE),
            self.rng.gen_range(0..MAP_SIZE),
        );
        self.p2_pos = (
            self.rng.gen_range(0..MAP_SIZE),
            self.rng.gen_range(0..MAP_SIZE),
        );

        // 确保位置合法
        while self.p1_pos == self.p2_pos
            || self.is_wall(self.p1_pos)
            || self.is_wall(self.p2_pos)
            || self.is_water(self.p1_pos)
            || self.is_water(self.p2_pos)
        {
            self.p1_pos = (
                self.rng.gen_range(0..MAP_SIZE),
                self.rng.gen_range(0..MAP_SIZE),
            );
            self.p2_pos = (
                self.rng.gen_range(0..MAP_SIZE),
                self.rng.gen_range(0..MAP_SIZE),
            );
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

    fn step(
        &mut self,
        action_p1: usize,
        action_p2: usize,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        f32,
        f32,
        bool,
        Vec<f32>,
        Vec<f32>,
        HashMap<String, f32>,
    ) {
        self.step_count += 1;

        // 清除上回合的闪避状态
        self.p1_dodge_active = false;
        self.p2_dodge_active = false;

        // 1. 转换动作为物理动作
        let phys_act_p1 = Self::transform_action(action_p1, false);
        let phys_act_p2 = Self::transform_action(action_p2, true);

        // 2. 处理动作（按类型分组处理）

        // P1 动作处理
        let mut cost_p1 = 0;
        match phys_act_p1 {
            ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => cost_p1 = COST_MOVE,
            ACT_ATTACK => cost_p1 = COST_ATTACK,
            ACT_SHOOT => cost_p1 = COST_SHOOT,
            ACT_DODGE => cost_p1 = COST_DODGE,
            ACT_SHIELD => cost_p1 = COST_SHIELD,
            ACT_DASH => cost_p1 = COST_DASH,
            ACT_AOE => cost_p1 = COST_AOE,
            ACT_HEAL => cost_p1 = COST_HEAL,
            ACT_RELOAD => cost_p1 = 0,
            _ => {}
        }

        // P2 动作处理
        let mut cost_p2 = 0;
        match phys_act_p2 {
            ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => cost_p2 = COST_MOVE,
            ACT_ATTACK => cost_p2 = COST_ATTACK,
            ACT_SHOOT => cost_p2 = COST_SHOOT,
            ACT_DODGE => cost_p2 = COST_DODGE,
            ACT_SHIELD => cost_p2 = COST_SHIELD,
            ACT_DASH => cost_p2 = COST_DASH,
            ACT_AOE => cost_p2 = COST_AOE,
            ACT_HEAL => cost_p2 = COST_HEAL,
            ACT_RELOAD => cost_p2 = 0,
            _ => {}
        }

        // 执行非攻击动作
        if self.p1_energy >= cost_p1 {
            self.p1_energy -= cost_p1;
            match phys_act_p1 {
                ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => {
                    let extra = self.apply_move(false, phys_act_p1);
                    self.p1_energy -= extra;
                }
                ACT_DODGE => self.p1_dodge_active = true,
                ACT_SHIELD => self.p1_shield = (self.p1_shield + 1).min(MAX_SHIELD),
                ACT_DASH => self.apply_dash(false),
                ACT_HEAL => {
                    if self.p1_heal_cooldown == 0 {
                        self.p1_hp = (self.p1_hp + 1).min(MAX_HP);
                        self.p1_heal_cooldown = HEAL_COOLDOWN;
                    }
                }
                ACT_RELOAD => self.p1_ammo = (self.p1_ammo + 3).min(MAX_AMMO),
                _ => {}
            }
        }

        if self.p2_energy >= cost_p2 {
            self.p2_energy -= cost_p2;
            match phys_act_p2 {
                ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => {
                    let extra = self.apply_move(true, phys_act_p2);
                    self.p2_energy -= extra;
                }
                ACT_DODGE => self.p2_dodge_active = true,
                ACT_SHIELD => self.p2_shield = (self.p2_shield + 1).min(MAX_SHIELD),
                ACT_DASH => self.apply_dash(true),
                ACT_HEAL => {
                    if self.p2_heal_cooldown == 0 {
                        self.p2_hp = (self.p2_hp + 1).min(MAX_HP);
                        self.p2_heal_cooldown = HEAL_COOLDOWN;
                    }
                }
                ACT_RELOAD => self.p2_ammo = (self.p2_ammo + 3).min(MAX_AMMO),
                _ => {}
            }
        }

        // 3. 道具拾取
        self.pickup_item(false);
        self.pickup_item(true);

        // 4. 攻击判定
        let mut r1 = 0.0;
        let mut r2 = 0.0;

        // P1 攻击
        if phys_act_p1 == ACT_ATTACK && cost_p1 == COST_ATTACK {
            if self.check_hit(self.p1_pos, self.p2_pos, ATTACK_RANGE) {
                if self.apply_damage(true, 1, false) {
                    r1 += 1.0;
                    r2 -= 1.0;
                    self.p1_damage_dealt += 1;
                }
                self.p1_attacks += 1;
            }
        } else if phys_act_p1 == ACT_SHOOT && cost_p1 == COST_SHOOT && self.p1_ammo > 0 {
            self.p1_ammo -= 1;
            if self.check_ranged_hit(self.p1_pos, self.p2_pos, SHOOT_RANGE) {
                if self.apply_damage(true, 1, true) {
                    r1 += 1.0;
                    r2 -= 1.0;
                    self.p1_damage_dealt += 1;
                }
                self.p1_attacks += 1;
            }
        } else if phys_act_p1 == ACT_AOE && cost_p1 == COST_AOE {
            if self.check_hit(self.p1_pos, self.p2_pos, AOE_RANGE) {
                if self.apply_damage(true, 1, false) {
                    r1 += 1.0;
                    r2 -= 1.0;
                    self.p1_damage_dealt += 1;
                }
                self.p1_attacks += 1;
            }
        }

        // P2 攻击
        if phys_act_p2 == ACT_ATTACK && cost_p2 == COST_ATTACK {
            if self.check_hit(self.p2_pos, self.p1_pos, ATTACK_RANGE) {
                if self.apply_damage(false, 1, false) {
                    r2 += 1.0;
                    r1 -= 1.0;
                    self.p2_damage_dealt += 1;
                }
                self.p2_attacks += 1;
            }
        } else if phys_act_p2 == ACT_SHOOT && cost_p2 == COST_SHOOT && self.p2_ammo > 0 {
            self.p2_ammo -= 1;
            if self.check_ranged_hit(self.p2_pos, self.p1_pos, SHOOT_RANGE) {
                if self.apply_damage(false, 1, true) {
                    r2 += 1.0;
                    r1 -= 1.0;
                    self.p2_damage_dealt += 1;
                }
                self.p2_attacks += 1;
            }
        } else if phys_act_p2 == ACT_AOE && cost_p2 == COST_AOE {
            if self.check_hit(self.p2_pos, self.p1_pos, AOE_RANGE) {
                if self.apply_damage(false, 1, false) {
                    r2 += 1.0;
                    r1 -= 1.0;
                    self.p2_damage_dealt += 1;
                }
                self.p2_attacks += 1;
            }
        }

        // 5. 更新冷却
        self.update_cooldowns();

        // 6. 能量恢复 (突然死亡阶段停止恢复)
        let regen = if self.step_count > SUDDEN_DEATH_STEP {
            0
        } else {
            REGEN_ENERGY
        };

        self.p1_energy = (self.p1_energy + regen).min(MAX_ENERGY);
        self.p2_energy = (self.p2_energy + regen).min(MAX_ENERGY);

        // 7. 结束判定
        let done = self.p1_hp <= 0 || self.p2_hp <= 0 || self.step_count >= MAX_STEPS;

        let mut info = HashMap::new();

        if done {
            if self.p1_hp > self.p2_hp {
                r1 += 5.0;
                r2 -= 5.0;
                info.insert("p1_win".to_string(), 1.0);
                info.insert("p2_win".to_string(), 0.0);
                info.insert("draw".to_string(), 0.0);
            } else if self.p2_hp > self.p1_hp {
                r2 += 5.0;
                r1 -= 5.0;
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
            info.insert("p1_damage".to_string(), self.p1_damage_dealt as f32);
            info.insert("p2_damage".to_string(), self.p2_damage_dealt as f32);
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

    fn obs_dim() -> usize {
        OBS_DIM
    }
    fn action_dim() -> usize {
        ACTION_DIM
    }
}

impl GameEnvZeroCopy for SimpleDuel {
    fn new() -> Self {
        <Self as GameEnv>::new()
    }

    fn reset_into(
        &mut self,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) {
        // Reset state (same as GameEnv::reset but without allocations)
        self.step_count = 0;
        self.p1_attacks = 0;
        self.p2_attacks = 0;
        self.p1_damage_dealt = 0;
        self.p2_damage_dealt = 0;
        self.p1_shield = 0;
        self.p2_shield = 0;
        self.p1_ammo = MAX_AMMO;
        self.p2_ammo = MAX_AMMO;
        self.p1_dodge_active = false;
        self.p2_dodge_active = false;
        self.p1_heal_cooldown = 0;
        self.p2_heal_cooldown = 0;

        self.generate_terrain();

        self.p1_pos = (
            self.rng.gen_range(0..MAP_SIZE),
            self.rng.gen_range(0..MAP_SIZE),
        );
        self.p2_pos = (
            self.rng.gen_range(0..MAP_SIZE),
            self.rng.gen_range(0..MAP_SIZE),
        );

        while self.p1_pos == self.p2_pos
            || self.is_wall(self.p1_pos)
            || self.is_wall(self.p2_pos)
            || self.is_water(self.p1_pos)
            || self.is_water(self.p2_pos)
        {
            self.p1_pos = (
                self.rng.gen_range(0..MAP_SIZE),
                self.rng.gen_range(0..MAP_SIZE),
            );
            self.p2_pos = (
                self.rng.gen_range(0..MAP_SIZE),
                self.rng.gen_range(0..MAP_SIZE),
            );
        }

        self.p1_hp = MAX_HP;
        self.p2_hp = MAX_HP;
        self.p1_energy = MAX_ENERGY;
        self.p2_energy = MAX_ENERGY;

        // Write directly to buffers
        self.get_obs_into(false, obs_p1);
        self.get_obs_into(true, obs_p2);
        self.get_mask_into(false, mask_p1);
        self.get_mask_into(true, mask_p2);
    }

    fn step_into(
        &mut self,
        action_p1: usize,
        action_p2: usize,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) -> (f32, f32, bool, GameInfo) {
        self.step_count += 1;

        // 清除上回合的闪避状态
        self.p1_dodge_active = false;
        self.p2_dodge_active = false;

        // 1. 转换动作为物理动作
        let phys_act_p1 = Self::transform_action(action_p1, false);
        let phys_act_p2 = Self::transform_action(action_p2, true);

        // 2. 处理动作（按类型分组处理）

        // P1 动作处理
        let mut cost_p1 = 0;
        match phys_act_p1 {
            ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => cost_p1 = COST_MOVE,
            ACT_ATTACK => cost_p1 = COST_ATTACK,
            ACT_SHOOT => cost_p1 = COST_SHOOT,
            ACT_DODGE => cost_p1 = COST_DODGE,
            ACT_SHIELD => cost_p1 = COST_SHIELD,
            ACT_DASH => cost_p1 = COST_DASH,
            ACT_AOE => cost_p1 = COST_AOE,
            ACT_HEAL => cost_p1 = COST_HEAL,
            ACT_RELOAD => cost_p1 = 0,
            _ => {}
        }

        // P2 动作处理
        let mut cost_p2 = 0;
        match phys_act_p2 {
            ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => cost_p2 = COST_MOVE,
            ACT_ATTACK => cost_p2 = COST_ATTACK,
            ACT_SHOOT => cost_p2 = COST_SHOOT,
            ACT_DODGE => cost_p2 = COST_DODGE,
            ACT_SHIELD => cost_p2 = COST_SHIELD,
            ACT_DASH => cost_p2 = COST_DASH,
            ACT_AOE => cost_p2 = COST_AOE,
            ACT_HEAL => cost_p2 = COST_HEAL,
            ACT_RELOAD => cost_p2 = 0,
            _ => {}
        }

        // 执行非攻击动作
        if self.p1_energy >= cost_p1 {
            self.p1_energy -= cost_p1;
            match phys_act_p1 {
                ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => {
                    let extra = self.apply_move(false, phys_act_p1);
                    self.p1_energy -= extra;
                }
                ACT_DODGE => self.p1_dodge_active = true,
                ACT_SHIELD => self.p1_shield = (self.p1_shield + 1).min(MAX_SHIELD),
                ACT_DASH => self.apply_dash(false),
                ACT_HEAL => {
                    if self.p1_heal_cooldown == 0 {
                        self.p1_hp = (self.p1_hp + 1).min(MAX_HP);
                        self.p1_heal_cooldown = HEAL_COOLDOWN;
                    }
                }
                ACT_RELOAD => self.p1_ammo = (self.p1_ammo + 3).min(MAX_AMMO),
                _ => {}
            }
        }

        if self.p2_energy >= cost_p2 {
            self.p2_energy -= cost_p2;
            match phys_act_p2 {
                ACT_UP | ACT_DOWN | ACT_LEFT | ACT_RIGHT => {
                    let extra = self.apply_move(true, phys_act_p2);
                    self.p2_energy -= extra;
                }
                ACT_DODGE => self.p2_dodge_active = true,
                ACT_SHIELD => self.p2_shield = (self.p2_shield + 1).min(MAX_SHIELD),
                ACT_DASH => self.apply_dash(true),
                ACT_HEAL => {
                    if self.p2_heal_cooldown == 0 {
                        self.p2_hp = (self.p2_hp + 1).min(MAX_HP);
                        self.p2_heal_cooldown = HEAL_COOLDOWN;
                    }
                }
                ACT_RELOAD => self.p2_ammo = (self.p2_ammo + 3).min(MAX_AMMO),
                _ => {}
            }
        }

        // 3. 道具拾取
        self.pickup_item(false);
        self.pickup_item(true);

        // 4. 攻击判定
        let mut r1 = 0.0;
        let mut r2 = 0.0;

        // P1 攻击
        if phys_act_p1 == ACT_ATTACK && cost_p1 == COST_ATTACK {
            if self.check_hit(self.p1_pos, self.p2_pos, ATTACK_RANGE) {
                if self.apply_damage(true, 1, false) {
                    r1 += 1.0;
                    r2 -= 1.0;
                    self.p1_damage_dealt += 1;
                }
                self.p1_attacks += 1;
            }
        } else if phys_act_p1 == ACT_SHOOT && cost_p1 == COST_SHOOT && self.p1_ammo > 0 {
            self.p1_ammo -= 1;
            if self.check_ranged_hit(self.p1_pos, self.p2_pos, SHOOT_RANGE) {
                if self.apply_damage(true, 1, true) {
                    r1 += 1.0;
                    r2 -= 1.0;
                    self.p1_damage_dealt += 1;
                }
                self.p1_attacks += 1;
            }
        } else if phys_act_p1 == ACT_AOE && cost_p1 == COST_AOE {
            if self.check_hit(self.p1_pos, self.p2_pos, AOE_RANGE) {
                if self.apply_damage(true, 1, false) {
                    r1 += 1.0;
                    r2 -= 1.0;
                    self.p1_damage_dealt += 1;
                }
                self.p1_attacks += 1;
            }
        }

        // P2 攻击
        if phys_act_p2 == ACT_ATTACK && cost_p2 == COST_ATTACK {
            if self.check_hit(self.p2_pos, self.p1_pos, ATTACK_RANGE) {
                if self.apply_damage(false, 1, false) {
                    r2 += 1.0;
                    r1 -= 1.0;
                    self.p2_damage_dealt += 1;
                }
                self.p2_attacks += 1;
            }
        } else if phys_act_p2 == ACT_SHOOT && cost_p2 == COST_SHOOT && self.p2_ammo > 0 {
            self.p2_ammo -= 1;
            if self.check_ranged_hit(self.p2_pos, self.p1_pos, SHOOT_RANGE) {
                if self.apply_damage(false, 1, true) {
                    r2 += 1.0;
                    r1 -= 1.0;
                    self.p2_damage_dealt += 1;
                }
                self.p2_attacks += 1;
            }
        } else if phys_act_p2 == ACT_AOE && cost_p2 == COST_AOE {
            if self.check_hit(self.p2_pos, self.p1_pos, AOE_RANGE) {
                if self.apply_damage(false, 1, false) {
                    r2 += 1.0;
                    r1 -= 1.0;
                    self.p2_damage_dealt += 1;
                }
                self.p2_attacks += 1;
            }
        }

        // 5. 更新冷却
        self.update_cooldowns();

        // 6. 能量恢复 (突然死亡阶段停止恢复)
        let regen = if self.step_count > SUDDEN_DEATH_STEP {
            0
        } else {
            REGEN_ENERGY
        };

        self.p1_energy = (self.p1_energy + regen).min(MAX_ENERGY);
        self.p2_energy = (self.p2_energy + regen).min(MAX_ENERGY);

        // 7. 结束判定
        let done = self.p1_hp <= 0 || self.p2_hp <= 0 || self.step_count >= MAX_STEPS;

        // Write directly to buffers
        self.get_obs_into(false, obs_p1);
        self.get_obs_into(true, obs_p2);
        self.get_mask_into(false, mask_p1);
        self.get_mask_into(true, mask_p2);

        let info = if done {
            if self.p1_hp > self.p2_hp {
                r1 += 5.0;
                r2 -= 5.0;
            } else if self.p2_hp > self.p1_hp {
                r2 += 5.0;
                r1 -= 5.0;
            }

            GameInfo::terminal(
                self.p1_hp > self.p2_hp,
                self.p2_hp > self.p1_hp,
                self.p1_hp == self.p2_hp,
                self.p1_attacks,
                self.p2_attacks,
                self.p1_damage_dealt,
                self.p2_damage_dealt,
                self.step_count,
            )
        } else {
            GameInfo::new()
        };

        (r1, r2, done, info)
    }

    fn obs_dim() -> usize {
        OBS_DIM
    }
    fn action_dim() -> usize {
        ACTION_DIM
    }
}

// ============================================================================
// 3. TicTacToe Implementation - Simple 3x3 game for testing
// ============================================================================

const TICTACTOE_OBS_DIM: usize = 27;
const TICTACTOE_ACTION_DIM: usize = 9;

#[derive(Clone)]
struct TicTacToe {
    board: [i8; 9],
    current_player: i8,
    step_count: i32,
}

impl TicTacToe {
    fn check_winner(&self) -> Option<i8> {
        const WIN_PATTERNS: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for pattern in WIN_PATTERNS {
            let a = self.board[pattern[0]];
            let b = self.board[pattern[1]];
            let c = self.board[pattern[2]];
            if a != 0 && a == b && b == c {
                return Some(a);
            }
        }
        None
    }

    fn is_board_full(&self) -> bool {
        self.board.iter().all(|&cell| cell != 0)
    }

    fn get_obs_for_player(&self, player: i8) -> Vec<f32> {
        let mut obs = vec![0.0; TICTACTOE_OBS_DIM];
        for (i, &cell) in self.board.iter().enumerate() {
            let transformed_cell = if player == 1 { cell } else { -cell };
            let base = i * 3;
            match transformed_cell {
                0 => obs[base] = 1.0,
                1 => obs[base + 1] = 1.0,
                -1 => obs[base + 2] = 1.0,
                _ => {}
            }
        }
        obs
    }

    fn get_obs_into_for_player(&self, player: i8, obs: &mut [f32]) {
        obs.fill(0.0);
        for (i, &cell) in self.board.iter().enumerate() {
            let transformed_cell = if player == 1 { cell } else { -cell };
            let base = i * 3;
            match transformed_cell {
                0 => obs[base] = 1.0,
                1 => obs[base + 1] = 1.0,
                -1 => obs[base + 2] = 1.0,
                _ => {}
            }
        }
    }

    fn get_mask(&self) -> Vec<f32> {
        let mut mask = vec![0.0; TICTACTOE_ACTION_DIM];
        for (i, &cell) in self.board.iter().enumerate() {
            if cell == 0 {
                mask[i] = 1.0;
            }
        }
        mask
    }

    fn get_mask_into(&self, mask: &mut [f32]) {
        for (i, &cell) in self.board.iter().enumerate() {
            mask[i] = if cell == 0 { 1.0 } else { 0.0 };
        }
    }
}

impl GameEnv for TicTacToe {
    fn new() -> Self {
        TicTacToe {
            board: [0; 9],
            current_player: 1,
            step_count: 0,
        }
    }

    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.board = [0; 9];
        self.current_player = 1;
        self.step_count = 0;

        let obs_p1 = self.get_obs_for_player(1);
        let obs_p2 = self.get_obs_for_player(-1);
        let mask = self.get_mask();

        (obs_p1, obs_p2, mask.clone(), mask)
    }

    fn step(
        &mut self,
        action_p1: usize,
        action_p2: usize,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        f32,
        f32,
        bool,
        Vec<f32>,
        Vec<f32>,
        HashMap<String, f32>,
    ) {
        self.step_count += 1;

        let action = if self.current_player == 1 {
            action_p1
        } else {
            action_p2
        };

        let mut r1 = 0.0;
        let mut r2 = 0.0;
        let mut done = false;
        let mut info = HashMap::new();

        if action < 9 && self.board[action] == 0 {
            self.board[action] = self.current_player;

            if let Some(winner) = self.check_winner() {
                done = true;
                if winner == 1 {
                    r1 = 1.0;
                    r2 = -1.0;
                    info.insert("p1_win".to_string(), 1.0);
                    info.insert("p2_win".to_string(), 0.0);
                } else {
                    r1 = -1.0;
                    r2 = 1.0;
                    info.insert("p1_win".to_string(), 0.0);
                    info.insert("p2_win".to_string(), 1.0);
                }
                info.insert("draw".to_string(), 0.0);
                info.insert("steps".to_string(), self.step_count as f32);
            } else if self.is_board_full() {
                done = true;
                info.insert("p1_win".to_string(), 0.0);
                info.insert("p2_win".to_string(), 0.0);
                info.insert("draw".to_string(), 1.0);
                info.insert("steps".to_string(), self.step_count as f32);
            }

            self.current_player = -self.current_player;
        }

        let obs_p1 = self.get_obs_for_player(1);
        let obs_p2 = self.get_obs_for_player(-1);
        let mask = self.get_mask();

        (obs_p1, obs_p2, r1, r2, done, mask.clone(), mask, info)
    }

    fn obs_dim() -> usize {
        TICTACTOE_OBS_DIM
    }

    fn action_dim() -> usize {
        TICTACTOE_ACTION_DIM
    }
}

impl GameEnvZeroCopy for TicTacToe {
    fn new() -> Self {
        <Self as GameEnv>::new()
    }

    fn reset_into(
        &mut self,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) {
        self.board = [0; 9];
        self.current_player = 1;
        self.step_count = 0;

        self.get_obs_into_for_player(1, obs_p1);
        self.get_obs_into_for_player(-1, obs_p2);
        self.get_mask_into(mask_p1);
        self.get_mask_into(mask_p2);
    }

    fn step_into(
        &mut self,
        action_p1: usize,
        action_p2: usize,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) -> (f32, f32, bool, GameInfo) {
        self.step_count += 1;

        let action = if self.current_player == 1 {
            action_p1
        } else {
            action_p2
        };

        let mut r1 = 0.0;
        let mut r2 = 0.0;
        let mut done = false;
        let mut info = GameInfo::new();

        if action < 9 && self.board[action] == 0 {
            self.board[action] = self.current_player;

            if let Some(winner) = self.check_winner() {
                done = true;
                if winner == 1 {
                    r1 = 1.0;
                    r2 = -1.0;
                    info = GameInfo::terminal(true, false, false, 0, 0, 0, 0, self.step_count);
                } else {
                    r1 = -1.0;
                    r2 = 1.0;
                    info = GameInfo::terminal(false, true, false, 0, 0, 0, 0, self.step_count);
                }
            } else if self.is_board_full() {
                done = true;
                info = GameInfo::terminal(false, false, true, 0, 0, 0, 0, self.step_count);
            }

            self.current_player = -self.current_player;
        }

        self.get_obs_into_for_player(1, obs_p1);
        self.get_obs_into_for_player(-1, obs_p2);
        self.get_mask_into(mask_p1);
        self.get_mask_into(mask_p2);

        (r1, r2, done, info)
    }

    fn obs_dim() -> usize {
        TICTACTOE_OBS_DIM
    }

    fn action_dim() -> usize {
        TICTACTOE_ACTION_DIM
    }
}

// ============================================================================
// 4. GameEnvDispatch - Type erasure enum for multiple games
// ============================================================================

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum GameEnvDispatch {
    SimpleDuel(SimpleDuel),
    TicTacToe(TicTacToe),
}

#[allow(dead_code)]
impl GameEnvDispatch {
    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        match self {
            GameEnvDispatch::SimpleDuel(env) => env.reset(),
            GameEnvDispatch::TicTacToe(env) => env.reset(),
        }
    }

    fn step(
        &mut self,
        action_p1: usize,
        action_p2: usize,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        f32,
        f32,
        bool,
        Vec<f32>,
        Vec<f32>,
        HashMap<String, f32>,
    ) {
        match self {
            GameEnvDispatch::SimpleDuel(env) => env.step(action_p1, action_p2),
            GameEnvDispatch::TicTacToe(env) => env.step(action_p1, action_p2),
        }
    }

    fn reset_into(
        &mut self,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) {
        match self {
            GameEnvDispatch::SimpleDuel(env) => env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2),
            GameEnvDispatch::TicTacToe(env) => env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2),
        }
    }

    fn step_into(
        &mut self,
        action_p1: usize,
        action_p2: usize,
        obs_p1: &mut [f32],
        obs_p2: &mut [f32],
        mask_p1: &mut [f32],
        mask_p2: &mut [f32],
    ) -> (f32, f32, bool, GameInfo) {
        match self {
            GameEnvDispatch::SimpleDuel(env) => {
                env.step_into(action_p1, action_p2, obs_p1, obs_p2, mask_p1, mask_p2)
            }
            GameEnvDispatch::TicTacToe(env) => {
                env.step_into(action_p1, action_p2, obs_p1, obs_p2, mask_p1, mask_p2)
            }
        }
    }

    fn obs_dim(&self) -> usize {
        match self {
            GameEnvDispatch::SimpleDuel(_) => <SimpleDuel as GameEnv>::obs_dim(),
            GameEnvDispatch::TicTacToe(_) => <TicTacToe as GameEnv>::obs_dim(),
        }
    }

    fn action_dim(&self) -> usize {
        match self {
            GameEnvDispatch::SimpleDuel(_) => <SimpleDuel as GameEnv>::action_dim(),
            GameEnvDispatch::TicTacToe(_) => <TicTacToe as GameEnv>::action_dim(),
        }
    }
}

// ============================================================================
// 5. VectorizedEnv PyClass (backward compatible)
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
            envs.push(<SimpleDuel as GameEnv>::new());
        }
        VectorizedEnv { envs }
    }

    fn reset<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
        let n = self.envs.len();
        let obs_dim = <SimpleDuel as GameEnv>::obs_dim();
        let act_dim = <SimpleDuel as GameEnv>::action_dim();

        let results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
            self.envs.par_iter_mut().map(|env| env.reset()).collect();

        let mut obs_batch = vec![0.0; 2 * n * obs_dim];
        let mut mask_batch = vec![0.0; 2 * n * act_dim];

        for (i, (o1, o2, m1, m2)) in results.into_iter().enumerate() {
            let p1_start = i * obs_dim;
            let p2_start = (n + i) * obs_dim;
            obs_batch[p1_start..p1_start + obs_dim].copy_from_slice(&o1);
            obs_batch[p2_start..p2_start + obs_dim].copy_from_slice(&o2);

            let m1_start = i * act_dim;
            let m2_start = (n + i) * act_dim;
            mask_batch[m1_start..m1_start + act_dim].copy_from_slice(&m1);
            mask_batch[m2_start..m2_start + act_dim].copy_from_slice(&m2);
        }

        let py_obs = PyArray1::from_vec(py, obs_batch)
            .reshape((2 * n, obs_dim))
            .unwrap();
        let py_mask = PyArray1::from_vec(py, mask_batch)
            .reshape((2 * n, act_dim))
            .unwrap();

        (py_obs, py_mask)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions_p1: Vec<usize>,
        actions_p2: Vec<usize>,
    ) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyList>,
    ) {
        let n = self.envs.len();
        let obs_dim = <SimpleDuel as GameEnv>::obs_dim();
        let act_dim = <SimpleDuel as GameEnv>::action_dim();

        assert_eq!(actions_p1.len(), n);

        assert_eq!(actions_p2.len(), n);

        let results: Vec<(
            Vec<f32>,
            Vec<f32>,
            f32,
            f32,
            bool,
            Vec<f32>,
            Vec<f32>,
            HashMap<String, f32>,
        )> = self
            .envs
            .par_iter_mut()
            .zip(actions_p1.par_iter().zip(actions_p2.par_iter()))
            .map(|(env, (&a1, &a2))| {
                let (o1, o2, r1, r2, d, m1, m2, info) = env.step(a1, a2);
                if d {
                    let (new_o1, new_o2, new_m1, new_m2) = env.reset();
                    (new_o1, new_o2, r1, r2, true, new_m1, new_m2, info)
                } else {
                    (o1, o2, r1, r2, false, m1, m2, info)
                }
            })
            .collect();

        let mut obs_batch = vec![0.0; 2 * n * obs_dim];
        let mut reward_batch = vec![0.0; 2 * n];
        let mut done_batch = vec![false; n];
        let mut mask_batch = vec![0.0; 2 * n * act_dim];

        let py_info_list = PyList::empty(py);

        for (i, (o1, o2, r1, r2, d, m1, m2, info)) in results.into_iter().enumerate() {
            let p1_obs_idx = i * obs_dim;
            let p2_obs_idx = (n + i) * obs_dim;
            obs_batch[p1_obs_idx..p1_obs_idx + obs_dim].copy_from_slice(&o1);
            obs_batch[p2_obs_idx..p2_obs_idx + obs_dim].copy_from_slice(&o2);

            reward_batch[i] = r1;
            reward_batch[n + i] = r2;

            done_batch[i] = d;

            let p1_mask_idx = i * act_dim;
            let p2_mask_idx = (n + i) * act_dim;
            mask_batch[p1_mask_idx..p1_mask_idx + act_dim].copy_from_slice(&m1);
            mask_batch[p2_mask_idx..p2_mask_idx + act_dim].copy_from_slice(&m2);

            let py_dict = PyDict::new(py);
            for (k, v) in info {
                py_dict.set_item(k, v).unwrap();
            }
            py_info_list.append(py_dict).unwrap();
        }

        let py_obs = PyArray1::from_vec(py, obs_batch)
            .reshape((2 * n, obs_dim))
            .unwrap();
        let py_reward = PyArray1::from_vec(py, reward_batch);
        let py_done = PyArray1::from_vec(py, done_batch);
        let py_mask = PyArray1::from_vec(py, mask_batch)
            .reshape((2 * n, act_dim))
            .unwrap();

        (py_obs, py_reward, py_done, py_mask, py_info_list)
    }

    fn obs_dim(&self) -> usize {
        <SimpleDuel as GameEnv>::obs_dim()
    }

    fn action_dim(&self) -> usize {
        <SimpleDuel as GameEnv>::action_dim()
    }
}

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn new(ptr: *mut T) -> Self {
        SendPtr(ptr)
    }
    fn as_ptr(self) -> *mut T {
        self.0
    }
}

#[pyclass]
pub struct VectorizedEnvZeroCopy {
    envs: Vec<SimpleDuel>,
    // Pre-allocated buffers
    obs_buffer: Vec<f32>,
    mask_buffer: Vec<f32>,
    reward_buffer: Vec<f32>,
    done_buffer: Vec<bool>,
    info_buffer: Vec<GameInfo>,
}

#[pymethods]
impl VectorizedEnvZeroCopy {
    #[new]
    fn new(num_envs: usize) -> Self {
        let obs_dim = <SimpleDuel as GameEnvZeroCopy>::obs_dim();
        let act_dim = <SimpleDuel as GameEnvZeroCopy>::action_dim();

        let mut envs = Vec::with_capacity(num_envs);
        for _ in 0..num_envs {
            envs.push(<SimpleDuel as GameEnvZeroCopy>::new());
        }

        VectorizedEnvZeroCopy {
            envs,
            obs_buffer: vec![0.0; 2 * num_envs * obs_dim],
            mask_buffer: vec![0.0; 2 * num_envs * act_dim],
            reward_buffer: vec![0.0; 2 * num_envs],
            done_buffer: vec![false; num_envs],
            info_buffer: vec![GameInfo::new(); num_envs],
        }
    }

    fn reset<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
        let n = self.envs.len();
        let obs_dim = <SimpleDuel as GameEnvZeroCopy>::obs_dim();
        let act_dim = <SimpleDuel as GameEnvZeroCopy>::action_dim();

        let obs_ptr = SendPtr::new(self.obs_buffer.as_mut_ptr());
        let mask_ptr = SendPtr::new(self.mask_buffer.as_mut_ptr());

        self.envs.par_iter_mut().enumerate().for_each(|(i, env)| {
            let p1_obs_start = i * obs_dim;
            let p2_obs_start = (n + i) * obs_dim;
            let p1_mask_start = i * act_dim;
            let p2_mask_start = (n + i) * act_dim;

            unsafe {
                let obs_p1 =
                    std::slice::from_raw_parts_mut(obs_ptr.as_ptr().add(p1_obs_start), obs_dim);
                let obs_p2 =
                    std::slice::from_raw_parts_mut(obs_ptr.as_ptr().add(p2_obs_start), obs_dim);
                let mask_p1 =
                    std::slice::from_raw_parts_mut(mask_ptr.as_ptr().add(p1_mask_start), act_dim);
                let mask_p2 =
                    std::slice::from_raw_parts_mut(mask_ptr.as_ptr().add(p2_mask_start), act_dim);

                env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2);
            }
        });

        let py_obs = PyArray1::from_slice(py, &self.obs_buffer)
            .reshape((2 * n, obs_dim))
            .unwrap();
        let py_mask = PyArray1::from_slice(py, &self.mask_buffer)
            .reshape((2 * n, act_dim))
            .unwrap();

        (py_obs, py_mask)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions_p1: Vec<usize>,
        actions_p2: Vec<usize>,
    ) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyList>,
    ) {
        let n = self.envs.len();
        let obs_dim = <SimpleDuel as GameEnvZeroCopy>::obs_dim();
        let act_dim = <SimpleDuel as GameEnvZeroCopy>::action_dim();

        assert_eq!(actions_p1.len(), n);
        assert_eq!(actions_p2.len(), n);

        let obs_ptr = SendPtr::new(self.obs_buffer.as_mut_ptr());
        let mask_ptr = SendPtr::new(self.mask_buffer.as_mut_ptr());
        let reward_ptr = SendPtr::new(self.reward_buffer.as_mut_ptr());
        let done_ptr = SendPtr::new(self.done_buffer.as_mut_ptr());
        let info_ptr = SendPtr::new(self.info_buffer.as_mut_ptr());

        self.envs
            .par_iter_mut()
            .enumerate()
            .zip(actions_p1.par_iter().zip(actions_p2.par_iter()))
            .for_each(|((i, env), (&a1, &a2))| {
                let p1_obs_start = i * obs_dim;
                let p2_obs_start = (n + i) * obs_dim;
                let p1_mask_start = i * act_dim;
                let p2_mask_start = (n + i) * act_dim;

                unsafe {
                    let obs_p1 =
                        std::slice::from_raw_parts_mut(obs_ptr.as_ptr().add(p1_obs_start), obs_dim);
                    let obs_p2 =
                        std::slice::from_raw_parts_mut(obs_ptr.as_ptr().add(p2_obs_start), obs_dim);
                    let mask_p1 = std::slice::from_raw_parts_mut(
                        mask_ptr.as_ptr().add(p1_mask_start),
                        act_dim,
                    );
                    let mask_p2 = std::slice::from_raw_parts_mut(
                        mask_ptr.as_ptr().add(p2_mask_start),
                        act_dim,
                    );

                    let (r1, r2, done, info) =
                        env.step_into(a1, a2, obs_p1, obs_p2, mask_p1, mask_p2);

                    *reward_ptr.as_ptr().add(i) = r1;
                    *reward_ptr.as_ptr().add(n + i) = r2;
                    *done_ptr.as_ptr().add(i) = done;
                    *info_ptr.as_ptr().add(i) = info;

                    if done {
                        env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2);
                    }
                }
            });

        // Build Python objects
        let py_obs = PyArray1::from_slice(py, &self.obs_buffer)
            .reshape((2 * n, obs_dim))
            .unwrap();
        let py_reward = PyArray1::from_slice(py, &self.reward_buffer);
        let py_done = PyArray1::from_slice(py, &self.done_buffer);
        let py_mask = PyArray1::from_slice(py, &self.mask_buffer)
            .reshape((2 * n, act_dim))
            .unwrap();

        // Build info list (only for terminal states)
        let py_info_list = PyList::empty(py);
        for info in &self.info_buffer {
            let py_dict = PyDict::new(py);
            if info.is_terminal {
                py_dict.set_item("p1_win", info.p1_win).unwrap();
                py_dict.set_item("p2_win", info.p2_win).unwrap();
                py_dict.set_item("draw", info.draw).unwrap();
                py_dict.set_item("p1_attacks", info.p1_attacks).unwrap();
                py_dict.set_item("p2_attacks", info.p2_attacks).unwrap();
                py_dict.set_item("p1_damage", info.p1_damage).unwrap();
                py_dict.set_item("p2_damage", info.p2_damage).unwrap();
                py_dict.set_item("steps", info.steps).unwrap();
            }
            py_info_list.append(py_dict).unwrap();
        }

        (py_obs, py_reward, py_done, py_mask, py_info_list)
    }

    fn obs_dim(&self) -> usize {
        <SimpleDuel as GameEnv>::obs_dim()
    }
    fn action_dim(&self) -> usize {
        <SimpleDuel as GameEnv>::action_dim()
    }
}

// ============================================================================
// 7. VectorizedEnvGeneric - Generic vectorized environment for any game
// ============================================================================

#[pyclass]
pub struct VectorizedEnvGeneric {
    envs: Vec<GameEnvDispatch>,
    game_name: String,
    obs_dim: usize,
    action_dim: usize,
    obs_buffer: Vec<f32>,
    mask_buffer: Vec<f32>,
    reward_buffer: Vec<f32>,
    done_buffer: Vec<bool>,
    info_buffer: Vec<GameInfo>,
}

#[pymethods]
impl VectorizedEnvGeneric {
    #[new]
    fn new(game_name: &str, num_envs: usize) -> PyResult<Self> {
        let (factory, obs_dim, action_dim) = GAME_REGISTRY.get(game_name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown game: '{}'. Available games: {:?}",
                game_name,
                GAME_REGISTRY.keys().collect::<Vec<_>>()
            ))
        })?;

        let mut envs = Vec::with_capacity(num_envs);
        for _ in 0..num_envs {
            envs.push(factory());
        }

        Ok(VectorizedEnvGeneric {
            envs,
            game_name: game_name.to_string(),
            obs_dim: *obs_dim,
            action_dim: *action_dim,
            obs_buffer: vec![0.0; 2 * num_envs * obs_dim],
            mask_buffer: vec![0.0; 2 * num_envs * action_dim],
            reward_buffer: vec![0.0; 2 * num_envs],
            done_buffer: vec![false; num_envs],
            info_buffer: vec![GameInfo::new(); num_envs],
        })
    }

    fn reset<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
        let n = self.envs.len();
        let obs_dim = self.obs_dim;
        let act_dim = self.action_dim;

        let obs_ptr = SendPtr::new(self.obs_buffer.as_mut_ptr());
        let mask_ptr = SendPtr::new(self.mask_buffer.as_mut_ptr());

        self.envs.par_iter_mut().enumerate().for_each(|(i, env)| {
            let p1_obs_start = i * obs_dim;
            let p2_obs_start = (n + i) * obs_dim;
            let p1_mask_start = i * act_dim;
            let p2_mask_start = (n + i) * act_dim;

            unsafe {
                let obs_p1 =
                    std::slice::from_raw_parts_mut(obs_ptr.as_ptr().add(p1_obs_start), obs_dim);
                let obs_p2 =
                    std::slice::from_raw_parts_mut(obs_ptr.as_ptr().add(p2_obs_start), obs_dim);
                let mask_p1 =
                    std::slice::from_raw_parts_mut(mask_ptr.as_ptr().add(p1_mask_start), act_dim);
                let mask_p2 =
                    std::slice::from_raw_parts_mut(mask_ptr.as_ptr().add(p2_mask_start), act_dim);

                env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2);
            }
        });

        let py_obs = PyArray1::from_slice(py, &self.obs_buffer)
            .reshape((2 * n, obs_dim))
            .unwrap();
        let py_mask = PyArray1::from_slice(py, &self.mask_buffer)
            .reshape((2 * n, act_dim))
            .unwrap();

        (py_obs, py_mask)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions_p1: Vec<usize>,
        actions_p2: Vec<usize>,
    ) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyList>,
    ) {
        let n = self.envs.len();
        let obs_dim = self.obs_dim;
        let act_dim = self.action_dim;

        assert_eq!(actions_p1.len(), n);
        assert_eq!(actions_p2.len(), n);

        let obs_ptr = SendPtr::new(self.obs_buffer.as_mut_ptr());
        let mask_ptr = SendPtr::new(self.mask_buffer.as_mut_ptr());
        let reward_ptr = SendPtr::new(self.reward_buffer.as_mut_ptr());
        let done_ptr = SendPtr::new(self.done_buffer.as_mut_ptr());
        let info_ptr = SendPtr::new(self.info_buffer.as_mut_ptr());

        self.envs
            .par_iter_mut()
            .enumerate()
            .zip(actions_p1.par_iter().zip(actions_p2.par_iter()))
            .for_each(|((i, env), (&a1, &a2))| {
                let p1_obs_start = i * obs_dim;
                let p2_obs_start = (n + i) * obs_dim;
                let p1_mask_start = i * act_dim;
                let p2_mask_start = (n + i) * act_dim;

                unsafe {
                    let obs_p1 =
                        std::slice::from_raw_parts_mut(obs_ptr.as_ptr().add(p1_obs_start), obs_dim);
                    let obs_p2 =
                        std::slice::from_raw_parts_mut(obs_ptr.as_ptr().add(p2_obs_start), obs_dim);
                    let mask_p1 = std::slice::from_raw_parts_mut(
                        mask_ptr.as_ptr().add(p1_mask_start),
                        act_dim,
                    );
                    let mask_p2 = std::slice::from_raw_parts_mut(
                        mask_ptr.as_ptr().add(p2_mask_start),
                        act_dim,
                    );

                    let (r1, r2, done, info) =
                        env.step_into(a1, a2, obs_p1, obs_p2, mask_p1, mask_p2);

                    *reward_ptr.as_ptr().add(i) = r1;
                    *reward_ptr.as_ptr().add(n + i) = r2;
                    *done_ptr.as_ptr().add(i) = done;
                    *info_ptr.as_ptr().add(i) = info;

                    if done {
                        env.reset_into(obs_p1, obs_p2, mask_p1, mask_p2);
                    }
                }
            });

        let py_obs = PyArray1::from_slice(py, &self.obs_buffer)
            .reshape((2 * n, obs_dim))
            .unwrap();
        let py_reward = PyArray1::from_slice(py, &self.reward_buffer);
        let py_done = PyArray1::from_slice(py, &self.done_buffer);
        let py_mask = PyArray1::from_slice(py, &self.mask_buffer)
            .reshape((2 * n, act_dim))
            .unwrap();

        let py_info_list = PyList::empty(py);
        for info in &self.info_buffer {
            let py_dict = PyDict::new(py);
            if info.is_terminal {
                py_dict.set_item("p1_win", info.p1_win).unwrap();
                py_dict.set_item("p2_win", info.p2_win).unwrap();
                py_dict.set_item("draw", info.draw).unwrap();
                py_dict.set_item("p1_attacks", info.p1_attacks).unwrap();
                py_dict.set_item("p2_attacks", info.p2_attacks).unwrap();
                py_dict.set_item("p1_damage", info.p1_damage).unwrap();
                py_dict.set_item("p2_damage", info.p2_damage).unwrap();
                py_dict.set_item("steps", info.steps).unwrap();
            }
            py_info_list.append(py_dict).unwrap();
        }

        (py_obs, py_reward, py_done, py_mask, py_info_list)
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn game_name(&self) -> &str {
        &self.game_name
    }
}

// ============================================================================
// 8. PyO3 Factory Functions
// ============================================================================

#[pyfunction]
fn create_env(game_name: &str, num_envs: usize) -> PyResult<VectorizedEnvGeneric> {
    VectorizedEnvGeneric::new(game_name, num_envs)
}

#[pyfunction]
fn list_games() -> Vec<&'static str> {
    GAME_REGISTRY.keys().copied().collect()
}

#[pyfunction]
fn get_game_info(game_name: &str) -> PyResult<(usize, usize)> {
    let (_, obs_dim, action_dim) = GAME_REGISTRY.get(game_name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown game: '{}'. Available games: {:?}",
            game_name,
            GAME_REGISTRY.keys().collect::<Vec<_>>()
        ))
    })?;
    Ok((*obs_dim, *action_dim))
}

#[pymodule]
fn high_perf_env(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VectorizedEnv>()?;
    m.add_class::<VectorizedEnvZeroCopy>()?;
    m.add_class::<VectorizedEnvGeneric>()?;
    m.add_function(wrap_pyfunction!(create_env, m)?)?;
    m.add_function(wrap_pyfunction!(list_games, m)?)?;
    m.add_function(wrap_pyfunction!(get_game_info, m)?)?;
    Ok(())
}
