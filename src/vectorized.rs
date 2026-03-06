use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::SimpleDuel;
use crate::registry::*;
use crate::traits::*;

type ResetBatch = Vec<GameReset>;
type StepBatch = Vec<GameStep>;
type PyStepResult<'py> = PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyList>,
)>;

fn validate_action_batch_lengths(
    actions_p1: &[usize],
    actions_p2: &[usize],
    n: usize,
) -> PyResult<()> {
    if actions_p1.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "actions_p1 length mismatch: expected {}, got {}",
            n,
            actions_p1.len()
        )));
    }

    if actions_p2.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "actions_p2 length mismatch: expected {}, got {}",
            n,
            actions_p2.len()
        )));
    }

    Ok(())
}

#[pyclass]
pub struct VectorizedEnv {
    envs: Vec<SimpleDuel>,
}

#[pymethods]
impl VectorizedEnv {
    #[new]
    pub fn new(num_envs: usize) -> Self {
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

        let results: ResetBatch = self.envs.par_iter_mut().map(|env| env.reset()).collect();

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
    ) -> PyStepResult<'py> {
        let n = self.envs.len();
        let obs_dim = <SimpleDuel as GameEnv>::obs_dim();
        let act_dim = <SimpleDuel as GameEnv>::action_dim();

        validate_action_batch_lengths(&actions_p1, &actions_p2, n)?;

        let results: StepBatch = self
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

        Ok((py_obs, py_reward, py_done, py_mask, py_info_list))
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
    fn ptr(self) -> *mut T {
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
    pub fn new(num_envs: usize) -> Self {
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
                    std::slice::from_raw_parts_mut(obs_ptr.ptr().add(p1_obs_start), obs_dim);
                let obs_p2 =
                    std::slice::from_raw_parts_mut(obs_ptr.ptr().add(p2_obs_start), obs_dim);
                let mask_p1 =
                    std::slice::from_raw_parts_mut(mask_ptr.ptr().add(p1_mask_start), act_dim);
                let mask_p2 =
                    std::slice::from_raw_parts_mut(mask_ptr.ptr().add(p2_mask_start), act_dim);

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
    ) -> PyStepResult<'py> {
        let n = self.envs.len();
        let obs_dim = <SimpleDuel as GameEnvZeroCopy>::obs_dim();
        let act_dim = <SimpleDuel as GameEnvZeroCopy>::action_dim();

        validate_action_batch_lengths(&actions_p1, &actions_p2, n)?;

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
                        std::slice::from_raw_parts_mut(obs_ptr.ptr().add(p1_obs_start), obs_dim);
                    let obs_p2 =
                        std::slice::from_raw_parts_mut(obs_ptr.ptr().add(p2_obs_start), obs_dim);
                    let mask_p1 =
                        std::slice::from_raw_parts_mut(mask_ptr.ptr().add(p1_mask_start), act_dim);
                    let mask_p2 =
                        std::slice::from_raw_parts_mut(mask_ptr.ptr().add(p2_mask_start), act_dim);

                    let (r1, r2, done, info) =
                        env.step_into(a1, a2, obs_p1, obs_p2, mask_p1, mask_p2);

                    *reward_ptr.ptr().add(i) = r1;
                    *reward_ptr.ptr().add(n + i) = r2;
                    *done_ptr.ptr().add(i) = done;
                    *info_ptr.ptr().add(i) = info;

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

        Ok((py_obs, py_reward, py_done, py_mask, py_info_list))
    }

    fn obs_dim(&self) -> usize {
        <SimpleDuel as GameEnv>::obs_dim()
    }
    fn action_dim(&self) -> usize {
        <SimpleDuel as GameEnv>::action_dim()
    }
}

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
    pub fn new(game_name: &str, num_envs: usize) -> PyResult<Self> {
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
                    std::slice::from_raw_parts_mut(obs_ptr.ptr().add(p1_obs_start), obs_dim);
                let obs_p2 =
                    std::slice::from_raw_parts_mut(obs_ptr.ptr().add(p2_obs_start), obs_dim);
                let mask_p1 =
                    std::slice::from_raw_parts_mut(mask_ptr.ptr().add(p1_mask_start), act_dim);
                let mask_p2 =
                    std::slice::from_raw_parts_mut(mask_ptr.ptr().add(p2_mask_start), act_dim);

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
    ) -> PyStepResult<'py> {
        let n = self.envs.len();
        let obs_dim = self.obs_dim;
        let act_dim = self.action_dim;

        validate_action_batch_lengths(&actions_p1, &actions_p2, n)?;

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
                        std::slice::from_raw_parts_mut(obs_ptr.ptr().add(p1_obs_start), obs_dim);
                    let obs_p2 =
                        std::slice::from_raw_parts_mut(obs_ptr.ptr().add(p2_obs_start), obs_dim);
                    let mask_p1 =
                        std::slice::from_raw_parts_mut(mask_ptr.ptr().add(p1_mask_start), act_dim);
                    let mask_p2 =
                        std::slice::from_raw_parts_mut(mask_ptr.ptr().add(p2_mask_start), act_dim);

                    let (r1, r2, done, info) =
                        env.step_into(a1, a2, obs_p1, obs_p2, mask_p1, mask_p2);

                    *reward_ptr.ptr().add(i) = r1;
                    *reward_ptr.ptr().add(n + i) = r2;
                    *done_ptr.ptr().add(i) = done;
                    *info_ptr.ptr().add(i) = info;

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

        Ok((py_obs, py_reward, py_done, py_mask, py_info_list))
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
