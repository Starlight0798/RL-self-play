pub mod games;
pub mod registry;
pub mod traits;
pub mod vectorized;

pub use games::connect4::*;
pub use games::reversi::*;
pub use games::simple_duel::*;
pub use games::tictactoe::*;
pub use registry::*;
pub use traits::*;
pub use vectorized::*;

use pyo3::prelude::*;

#[pyfunction]
fn create_env(game_name: &str, num_envs: usize) -> PyResult<VectorizedEnvGeneric> {
    VectorizedEnvGeneric::new(game_name, num_envs)
}

#[pyfunction]
fn list_games() -> Vec<&'static str> {
    GAME_REGISTRY.keys().cloned().collect()
}

#[pyfunction]
fn get_game_info(game_name: &str) -> PyResult<(usize, usize)> {
    let (_, obs_dim, act_dim) = GAME_REGISTRY.get(game_name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Unknown game: {}", game_name))
    })?;
    Ok((*obs_dim, *act_dim))
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
