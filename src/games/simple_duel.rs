use crate::traits::*;
use rand::prelude::*;
use std::collections::HashMap;

// ============================================================================
// SimpleDuel 实现 - 升级版 12x12 战术游戏
// ============================================================================

pub const MAP_SIZE: i32 = 12;
pub const MAX_HP: i32 = 4;
pub const MAX_ENERGY: i32 = 7;
pub const MAX_SHIELD: i32 = 2;
pub const MAX_AMMO: i32 = 6;
pub const REGEN_ENERGY: i32 = 1;

// 动作能量消耗
pub const COST_MOVE: i32 = 1;
pub const COST_ATTACK: i32 = 2;
pub const COST_SHOOT: i32 = 3;
pub const COST_DODGE: i32 = 2;
pub const COST_SHIELD: i32 = 3;
pub const COST_DASH: i32 = 3;
pub const COST_AOE: i32 = 4;
pub const COST_HEAL: i32 = 4;

// 攻击范围
pub const ATTACK_RANGE: i32 = 1;
pub const SHOOT_RANGE: i32 = 4; // 增加远程射击范围
pub const AOE_RANGE: i32 = 1;

// 游戏时间
pub const MAX_STEPS: i32 = 60; // 增加最大步数
pub const SUDDEN_DEATH_STEP: i32 = 40; // 突然死亡阶段开始时间

// 冷却时间
pub const HEAL_COOLDOWN: i32 = 5;

// 动作定义
pub const ACT_STAY: usize = 0;
pub const ACT_UP: usize = 1;
pub const ACT_DOWN: usize = 2;
pub const ACT_LEFT: usize = 3;
pub const ACT_RIGHT: usize = 4;
pub const ACT_ATTACK: usize = 5;
pub const ACT_SHOOT: usize = 6;
pub const ACT_DODGE: usize = 7;
pub const ACT_SHIELD: usize = 8;
pub const ACT_DASH: usize = 9;
pub const ACT_AOE: usize = 10;
pub const ACT_HEAL: usize = 11;
pub const ACT_RELOAD: usize = 12;

pub const ACTION_DIM: usize = 13;
pub const OBS_DIM: usize = 16 + 144; // = 160

// 地形类型
pub const TERRAIN_EMPTY: u8 = 0;
pub const TERRAIN_WALL: u8 = 1;
pub const TERRAIN_WATER: u8 = 2;
pub const TERRAIN_HIGH_GROUND: u8 = 3;

// 道具类型
pub const ITEM_NONE: u8 = 0;
pub const ITEM_HEALTH: u8 = 1;
pub const ITEM_ENERGY: u8 = 2;
pub const ITEM_AMMO: u8 = 3;
pub const ITEM_SHIELD: u8 = 4;

// 道具刷新时间
pub const ITEM_RESPAWN_TIME: i32 = 10;

#[derive(Clone)]
pub struct SimpleDuel {
    pub p1_pos: (i32, i32),
    pub p2_pos: (i32, i32),
    pub p1_hp: i32,
    pub p2_hp: i32,
    pub p1_energy: i32,
    pub p2_energy: i32,
    pub p1_shield: i32,
    pub p2_shield: i32,
    pub p1_ammo: i32,
    pub p2_ammo: i32,
    pub p1_dodge_active: bool,
    pub p2_dodge_active: bool,
    pub p1_heal_cooldown: i32,
    pub p2_heal_cooldown: i32,
    pub step_count: i32,
    pub terrain: Vec<u8>,       // 144格地形
    pub items: Vec<u8>,         // 144格道具
    pub item_respawn: Vec<i32>, // 道具刷新倒计时
    pub rng: StdRng,
    // 统计信息
    pub p1_attacks: i32,
    pub p2_attacks: i32,
    pub p1_damage_dealt: i32,
    pub p2_damage_dealt: i32,
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

    // 检查伤害是否会被闪避或高地减免（不修改状态）
    fn check_damage_blocked(&mut self, is_p2_target: bool, is_ranged: bool) -> bool {
        let (dodge_active, high_ground) = if is_p2_target {
            (self.p2_dodge_active, self.is_high_ground(self.p2_pos))
        } else {
            (self.p1_dodge_active, self.is_high_ground(self.p1_pos))
        };

        // 闪避检查
        if dodge_active {
            return true;
        }

        // 高地远程攻击减伤（50%几率闪避）
        if is_ranged && high_ground {
            if self.rng.gen_bool(0.5) {
                return true;
            }
        }

        false
    }

    // 应用伤害到目标（不检查闪避，闪避已在check_damage_blocked中处理）
    fn apply_damage_direct(&mut self, is_p2_target: bool, damage: i32) -> bool {
        let shield = if is_p2_target {
            self.p2_shield
        } else {
            self.p1_shield
        };

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
                    self.p1_energy = (self.p1_energy - extra).max(0);
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
                    self.p2_energy = (self.p2_energy - extra).max(0);
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

        // 4. 攻击判定 - 同时计算，同时应用
        let mut r1 = 0.0;
        let mut r2 = 0.0;

        // 阶段1: 计算所有攻击是否命中（不应用伤害）
        let mut p1_attack_info: Option<(i32, bool)> = None; // (damage, is_ranged)
        let mut p2_attack_info: Option<(i32, bool)> = None;

        // P1 攻击计算
        if phys_act_p1 == ACT_ATTACK && cost_p1 == COST_ATTACK {
            if self.check_hit(self.p1_pos, self.p2_pos, ATTACK_RANGE) {
                p1_attack_info = Some((1, false));
                self.p1_attacks += 1;
            }
        } else if phys_act_p1 == ACT_SHOOT && cost_p1 == COST_SHOOT && self.p1_ammo > 0 {
            self.p1_ammo -= 1;
            if self.check_ranged_hit(self.p1_pos, self.p2_pos, SHOOT_RANGE) {
                p1_attack_info = Some((1, true));
                self.p1_attacks += 1;
            }
        } else if phys_act_p1 == ACT_AOE && cost_p1 == COST_AOE {
            if self.check_hit(self.p1_pos, self.p2_pos, AOE_RANGE) {
                p1_attack_info = Some((1, false));
                self.p1_attacks += 1;
            }
        }

        // P2 攻击计算
        if phys_act_p2 == ACT_ATTACK && cost_p2 == COST_ATTACK {
            if self.check_hit(self.p2_pos, self.p1_pos, ATTACK_RANGE) {
                p2_attack_info = Some((1, false));
                self.p2_attacks += 1;
            }
        } else if phys_act_p2 == ACT_SHOOT && cost_p2 == COST_SHOOT && self.p2_ammo > 0 {
            self.p2_ammo -= 1;
            if self.check_ranged_hit(self.p2_pos, self.p1_pos, SHOOT_RANGE) {
                p2_attack_info = Some((1, true));
                self.p2_attacks += 1;
            }
        } else if phys_act_p2 == ACT_AOE && cost_p2 == COST_AOE {
            if self.check_hit(self.p2_pos, self.p1_pos, AOE_RANGE) {
                p2_attack_info = Some((1, false));
                self.p2_attacks += 1;
            }
        }

        // 阶段2: 检查闪避（同时检查，闪避可以挡住所有攻击）
        let p1_blocked = if let Some((_, is_ranged)) = p1_attack_info {
            self.check_damage_blocked(true, is_ranged)
        } else {
            false
        };
        let p2_blocked = if let Some((_, is_ranged)) = p2_attack_info {
            self.check_damage_blocked(false, is_ranged)
        } else {
            false
        };

        // 阶段3: 同时应用伤害
        if let Some((damage, _)) = p1_attack_info {
            if !p1_blocked {
                if self.apply_damage_direct(true, damage) {
                    r1 += 1.0;
                    r2 -= 1.0;
                    self.p1_damage_dealt += 1;
                }
            }
        }
        if let Some((damage, _)) = p2_attack_info {
            if !p2_blocked {
                if self.apply_damage_direct(false, damage) {
                    r2 += 1.0;
                    r1 -= 1.0;
                    self.p2_damage_dealt += 1;
                }
            }
        }

        // ============ 奖励塑形 (零和设计) ============

        // 1. 攻击意图奖励 - 鼓励主动进攻 (零和)
        let p1_attacked = p1_attack_info.is_some();
        let p2_attacked = p2_attack_info.is_some();
        if p1_attacked && !p2_attacked {
            r1 += 0.05;
            r2 -= 0.05;
        } else if p2_attacked && !p1_attacked {
            r2 += 0.05;
            r1 -= 0.05;
        }

        // 2. 高地控制奖励 (零和)
        let p1_high = self.is_high_ground(self.p1_pos);
        let p2_high = self.is_high_ground(self.p2_pos);
        if p1_high && !p2_high {
            r1 += 0.02;
            r2 -= 0.02;
        } else if p2_high && !p1_high {
            r2 += 0.02;
            r1 -= 0.02;
        }

        // 3. 距离接近奖励 (鼓励交战)
        let dist = (self.p1_pos.0 - self.p2_pos.0).abs() + (self.p1_pos.1 - self.p2_pos.1).abs();
        let engagement_bonus = if dist <= 3 { 0.01 } else { 0.0 };
        r1 += engagement_bonus;
        r2 += engagement_bonus;

        // 4. 资源效率惩罚 (自身相关)
        if self.p1_energy >= MAX_ENERGY && phys_act_p1 == ACT_STAY {
            r1 -= 0.01;
        }
        if self.p2_energy >= MAX_ENERGY && phys_act_p2 == ACT_STAY {
            r2 -= 0.01;
        }

        // 阶段4: 清除闪避状态（在所有攻击处理完后）
        if p1_blocked {
            self.p2_dodge_active = false;
        }
        if p2_blocked {
            self.p1_dodge_active = false;
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
                    self.p1_energy = (self.p1_energy - extra).max(0);
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
                    self.p2_energy = (self.p2_energy - extra).max(0);
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

        // 4. 攻击判定 - 同时计算，同时应用
        let mut r1 = 0.0;
        let mut r2 = 0.0;

        // 阶段1: 计算所有攻击是否命中（不应用伤害）
        let mut p1_attack_info: Option<(i32, bool)> = None; // (damage, is_ranged)
        let mut p2_attack_info: Option<(i32, bool)> = None;

        // P1 攻击计算
        if phys_act_p1 == ACT_ATTACK && cost_p1 == COST_ATTACK {
            if self.check_hit(self.p1_pos, self.p2_pos, ATTACK_RANGE) {
                p1_attack_info = Some((1, false));
                self.p1_attacks += 1;
            }
        } else if phys_act_p1 == ACT_SHOOT && cost_p1 == COST_SHOOT && self.p1_ammo > 0 {
            self.p1_ammo -= 1;
            if self.check_ranged_hit(self.p1_pos, self.p2_pos, SHOOT_RANGE) {
                p1_attack_info = Some((1, true));
                self.p1_attacks += 1;
            }
        } else if phys_act_p1 == ACT_AOE && cost_p1 == COST_AOE {
            if self.check_hit(self.p1_pos, self.p2_pos, AOE_RANGE) {
                p1_attack_info = Some((1, false));
                self.p1_attacks += 1;
            }
        }

        // P2 攻击计算
        if phys_act_p2 == ACT_ATTACK && cost_p2 == COST_ATTACK {
            if self.check_hit(self.p2_pos, self.p1_pos, ATTACK_RANGE) {
                p2_attack_info = Some((1, false));
                self.p2_attacks += 1;
            }
        } else if phys_act_p2 == ACT_SHOOT && cost_p2 == COST_SHOOT && self.p2_ammo > 0 {
            self.p2_ammo -= 1;
            if self.check_ranged_hit(self.p2_pos, self.p1_pos, SHOOT_RANGE) {
                p2_attack_info = Some((1, true));
                self.p2_attacks += 1;
            }
        } else if phys_act_p2 == ACT_AOE && cost_p2 == COST_AOE {
            if self.check_hit(self.p2_pos, self.p1_pos, AOE_RANGE) {
                p2_attack_info = Some((1, false));
                self.p2_attacks += 1;
            }
        }

        // 阶段2: 检查闪避（同时检查，闪避可以挡住所有攻击）
        let p1_blocked = if let Some((_, is_ranged)) = p1_attack_info {
            self.check_damage_blocked(true, is_ranged)
        } else {
            false
        };
        let p2_blocked = if let Some((_, is_ranged)) = p2_attack_info {
            self.check_damage_blocked(false, is_ranged)
        } else {
            false
        };

        // 阶段3: 同时应用伤害
        if let Some((damage, _)) = p1_attack_info {
            if !p1_blocked {
                if self.apply_damage_direct(true, damage) {
                    r1 += 1.0;
                    r2 -= 1.0;
                    self.p1_damage_dealt += 1;
                }
            }
        }
        if let Some((damage, _)) = p2_attack_info {
            if !p2_blocked {
                if self.apply_damage_direct(false, damage) {
                    r2 += 1.0;
                    r1 -= 1.0;
                    self.p2_damage_dealt += 1;
                }
            }
        }

        // ============ 奖励塑形 (零和设计) ============

        // 1. 攻击意图奖励 - 鼓励主动进攻 (零和)
        // 尝试攻击即获得小奖励，无论是否命中
        let p1_attacked = p1_attack_info.is_some();
        let p2_attacked = p2_attack_info.is_some();
        if p1_attacked && !p2_attacked {
            r1 += 0.05;
            r2 -= 0.05;
        } else if p2_attacked && !p1_attacked {
            r2 += 0.05;
            r1 -= 0.05;
        }
        // 双方都攻击时不给额外奖励，保持零和

        // 2. 高地控制奖励 (零和)
        // 占据高地获得战术优势
        let p1_high = self.is_high_ground(self.p1_pos);
        let p2_high = self.is_high_ground(self.p2_pos);
        if p1_high && !p2_high {
            r1 += 0.02;
            r2 -= 0.02;
        } else if p2_high && !p1_high {
            r2 += 0.02;
            r1 -= 0.02;
        }

        // 3. 距离接近奖励 (零和，鼓励交战)
        // 计算曼哈顿距离
        let dist = (self.p1_pos.0 - self.p2_pos.0).abs() + (self.p1_pos.1 - self.p2_pos.1).abs();
        // 距离越近，双方都获得小奖励（鼓励交战而非逃跑）
        // 但这不是零和的，所以我们改为：主动接近的一方获得奖励
        // 通过比较移动方向来判断谁在接近
        let engagement_bonus = if dist <= 3 {
            0.01 // 近距离交战奖励
        } else {
            0.0
        };
        // 近距离时双方都获得小奖励（鼓励保持交战状态）
        r1 += engagement_bonus;
        r2 += engagement_bonus;

        // 4. 资源效率惩罚 (自身相关，非零和)
        // 能量满时不行动是浪费
        if self.p1_energy >= MAX_ENERGY && phys_act_p1 == ACT_STAY {
            r1 -= 0.01;
        }
        if self.p2_energy >= MAX_ENERGY && phys_act_p2 == ACT_STAY {
            r2 -= 0.01;
        }

        // 阶段4: 清除闪避状态（在所有攻击处理完后）
        if p1_blocked {
            self.p2_dodge_active = false;
        }
        if p2_blocked {
            self.p1_dodge_active = false;
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
