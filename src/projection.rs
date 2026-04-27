//! 2D projection of high-dimensional centroids via PCA.
//!
//! For visualization only — distances in the 2D plot will be lossy
//! (especially when intrinsic dimensionality is high), but neighborhood
//! structure is roughly preserved.

use anyhow::{Result, anyhow};
use nalgebra::DMatrix;

#[derive(Clone, Copy, Debug)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

/// Project `n × d` centroid vectors to 2D using the top two principal
/// components. Returns one `Point2` per centroid, in the same order.
pub fn pca_2d(centroids: &[Vec<f32>]) -> Result<Vec<Point2>> {
    let n = centroids.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let dim = centroids[0].len();
    if dim == 0 {
        return Err(anyhow!("centroids have zero dimension"));
    }
    if dim < 2 {
        return Err(anyhow!("need at least 2D centroids, got dim={dim}"));
    }

    let mut data = DMatrix::<f64>::zeros(n, dim);
    for (i, v) in centroids.iter().enumerate() {
        for (j, &x) in v.iter().enumerate() {
            data[(i, j)] = x as f64;
        }
    }

    // Center: subtract column means.
    let mean = data.row_mean();
    for j in 0..dim {
        let m = mean[j];
        for i in 0..n {
            data[(i, j)] -= m;
        }
    }

    // SVD: data = U * Σ * V^T. The 2D PCA scores are the first two columns
    // of (U * Σ), which equal data * V[:, :2].
    let svd = data.svd(true, false);
    let u = svd
        .u
        .ok_or_else(|| anyhow!("SVD did not produce U matrix"))?;
    let s = svd.singular_values;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = u[(i, 0)] * s[0];
        let y = u[(i, 1)] * s[1];
        out.push(Point2 {
            x: x as f32,
            y: y as f32,
        });
    }
    Ok(out)
}
