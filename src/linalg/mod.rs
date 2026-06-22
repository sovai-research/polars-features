use faer::linalg::solvers::{DenseSolveCore, Llt};
use faer::{MatRef, Side};
use faer_ext::IntoNdarray;
use ndarray::Array2;

#[inline]
pub fn lstsq_solver1(x: MatRef<f64>, y: MatRef<f64>) -> Array2<f64> {
    // Solver1. Use closed form solution to solve the least squares problem.
    // This is faster because xtx has a small dimension, so we use the closed
    // form (normal equations) approach.
    let xt = x.transpose();
    let xtx = xt * x;
    // xtx is positive semidefinite, so the Cholesky decomposition succeeds.
    let cholesky = Llt::new(xtx.as_ref(), Side::Lower).unwrap();
    let xtx_inv = cholesky.inverse();
    // Solution: beta = (XtX)^-1 Xt y
    let beta = xtx_inv * xt * y;
    let out = beta.as_ref().into_ndarray();
    out.to_owned()
}
