//! Histogram and percentile helpers over a list of centroid populations.

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Distribution {
    pub min: u64,
    pub max: u64,
    pub mean: f64,
    pub median: u64,
    pub p95: u64,
    pub p99: u64,
    pub stddev: f64,
    pub histogram: Vec<HistogramBucket>,
}

#[derive(Debug, Serialize)]
pub struct HistogramBucket {
    pub label: &'static str,
    pub lo: u64,
    pub hi: Option<u64>, // None = open-ended
    pub count: u64,
}

/// Expects `counts` to contain one entry per list/centroid (zeros allowed for
/// empty lists if the caller wants them represented).
pub fn distribution(counts: &[u64]) -> Option<Distribution> {
    if counts.is_empty() {
        return None;
    }
    let mut sorted: Vec<u64> = counts.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let min = sorted[0];
    let max = sorted[n - 1];
    let sum: u128 = sorted.iter().map(|&x| x as u128).sum();
    let mean = sum as f64 / n as f64;
    let median = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);
    let p99 = percentile(&sorted, 0.99);
    let var: f64 = sorted
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let stddev = var.sqrt();
    let histogram = build_histogram(&sorted);
    Some(Distribution {
        min,
        max,
        mean,
        median,
        p95,
        p99,
        stddev,
        histogram,
    })
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    // nearest-rank; simple and good enough for diagnostics
    let rank = (p * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

fn build_histogram(sorted: &[u64]) -> Vec<HistogramBucket> {
    let edges: [(u64, Option<u64>, &'static str); 6] = [
        (0, Some(0), "0"),
        (1, Some(10), "1-10"),
        (11, Some(100), "11-100"),
        (101, Some(1_000), "101-1k"),
        (1_001, Some(10_000), "1k-10k"),
        (10_001, None, "10k+"),
    ];
    edges
        .iter()
        .map(|&(lo, hi, label)| {
            let count = sorted
                .iter()
                .filter(|&&v| v >= lo && hi.map_or(true, |h| v <= h))
                .count() as u64;
            HistogramBucket {
                label,
                lo,
                hi,
                count,
            }
        })
        .collect()
}
