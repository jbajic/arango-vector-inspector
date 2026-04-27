//! 2D projections of high-dimensional centroids for visualization.
//!
//! Two methods are provided:
//!  * `pca_2d` — fast linear projection (top-2 principal components). Tends
//!    to render data on a hypersphere as a 2D disk, which can look "circly".
//!  * `tsne_2d` — non-linear t-SNE projection. Slower but breaks the
//!    spherical-disk artifact and produces a more spread-out, "classical"
//!    looking layout suitable for Voronoi diagrams.
//!
//! Distances in either output are lossy; relative neighborhood structure is
//! roughly preserved (more so for t-SNE).

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

/// Project to 2D using Barnes-Hut t-SNE. Slow (seconds) but produces a
/// non-linear layout that doesn't collapse spherical data into a disk.
///
/// Perplexity is auto-clamped to fit small inputs (must be < n/3).
pub fn tsne_2d(centroids: &[Vec<f32>]) -> Result<Vec<Point2>> {
    let n = centroids.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n < 4 {
        // t-SNE needs more points than perplexity; fall back to PCA.
        return pca_2d(centroids);
    }

    let perplexity = (n as f32 / 3.0 - 1.0).min(30.0).max(5.0);
    let samples: Vec<&[f32]> = centroids.iter().map(|v| v.as_slice()).collect();

    let mut tsne: bhtsne::tSNE<f32, &[f32]> = bhtsne::tSNE::new(&samples);
    tsne.embedding_dim(2)
        .perplexity(perplexity)
        .epochs(1000)
        .barnes_hut(0.5, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        });

    let flat = tsne.embedding();
    let mut out = Vec::with_capacity(n);
    for chunk in flat.chunks_exact(2) {
        out.push(Point2 {
            x: chunk[0],
            y: chunk[1],
        });
    }
    if out.len() != n {
        return Err(anyhow!(
            "t-SNE returned {} points, expected {}",
            out.len(),
            n
        ));
    }
    Ok(out)
}
