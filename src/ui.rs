//! Minimal egui UI: render a Voronoi diagram of the centroids projected
//! into 2D, color cells by vector count, show tooltips.

use anyhow::Result;
use eframe::egui;
use voronator::VoronoiDiagram;
use voronator::delaunator::Point;

use crate::projection::Point2;

pub struct CentroidView {
    /// Centroid 2D coordinate
    pub point: Point2,
    /// Number of vectors assigned to this centroid (its inverted-list size)
    pub count: u64,
}

pub struct ViewerData {
    pub title: String,
    /// Key-value rows shown on the Overview tab.
    pub overview: Vec<(String, String)>,
    pub centroids: Vec<CentroidView>,
}

/// Open a window and block until the user closes it.
pub fn run(data: ViewerData) -> Result<()> {
    let title = data.title.clone();
    let app = ViewerApp::new(data);
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(&title, native_options, Box::new(|_cc| Ok(Box::new(app))))
        .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;
    Ok(())
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Tab {
    Overview,
    Voronoi,
    Histogram,
}

struct ViewerApp {
    data: ViewerData,
    /// Cached Voronoi cells in data-space coordinates.
    cells: Vec<Vec<egui::Pos2>>,
    /// Bounding box of all centroid points (data-space).
    bbox: egui::Rect,
    /// Max count, used for histogram scaling.
    max_count: u64,
    /// (count, original_centroid_index) sorted by count descending.
    sorted_counts: Vec<(u64, usize)>,
    tab: Tab,
}

impl ViewerApp {
    fn new(data: ViewerData) -> Self {
        let (cells, bbox) = build_cells(&data.centroids);
        let max_count = data.centroids.iter().map(|c| c.count).max().unwrap_or(1);
        let mut sorted_counts: Vec<(u64, usize)> = data
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (c.count, i))
            .collect();
        // Sort by count descending; break ties by original index ascending so
        // ordering is stable and reproducible.
        sorted_counts.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        Self {
            data,
            cells,
            bbox,
            max_count,
            sorted_counts,
            tab: Tab::Overview,
        }
    }
}

/// Build Voronoi cells; returns one polygon per input centroid (in matching
/// order, possibly empty if the cell is degenerate). Also returns the data
/// bounding box used to clip the diagram.
fn build_cells(centroids: &[CentroidView]) -> (Vec<Vec<egui::Pos2>>, egui::Rect) {
    if centroids.is_empty() {
        return (Vec::new(), egui::Rect::ZERO);
    }
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for c in centroids {
        min_x = min_x.min(c.point.x);
        min_y = min_y.min(c.point.y);
        max_x = max_x.max(c.point.x);
        max_y = max_y.max(c.point.y);
    }
    // Pad the clipping box a little so edge cells aren't clipped tight.
    let pad_x = (max_x - min_x).max(1.0) * 0.1;
    let pad_y = (max_y - min_y).max(1.0) * 0.1;
    let (lo, hi) = (
        Point {
            x: (min_x - pad_x) as f64,
            y: (min_y - pad_y) as f64,
        },
        Point {
            x: (max_x + pad_x) as f64,
            y: (max_y + pad_y) as f64,
        },
    );

    let pts: Vec<Point> = centroids
        .iter()
        .map(|c| Point {
            x: c.point.x as f64,
            y: c.point.y as f64,
        })
        .collect();

    let diagram = VoronoiDiagram::<Point>::new(&lo, &hi, &pts).expect("Voronoi computation failed");

    // diagram.cells() preserves input order.
    let cells: Vec<Vec<egui::Pos2>> = diagram
        .cells()
        .iter()
        .map(|cell| {
            cell.points()
                .iter()
                .map(|p| egui::pos2(p.x as f32, p.y as f32))
                .collect()
        })
        .collect();

    let bbox = egui::Rect::from_min_max(
        egui::pos2(lo.x as f32, lo.y as f32),
        egui::pos2(hi.x as f32, hi.y as f32),
    );
    (cells, bbox)
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::light());

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.heading(&self.data.title);
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Overview, "Overview");
                ui.selectable_value(&mut self.tab, Tab::Voronoi, "Voronoi");
                ui.selectable_value(&mut self.tab, Tab::Histogram, "Histogram");
            });
        });

        match self.tab {
            Tab::Overview => self.render_overview(ctx),
            Tab::Voronoi => self.render_voronoi(ctx),
            Tab::Histogram => self.render_histogram(ctx),
        }
    }
}

impl ViewerApp {
    fn render_overview(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(6.0);
            egui::Grid::new("overview_kvs")
                .striped(true)
                .num_columns(2)
                .spacing([20.0, 4.0])
                .show(ui, |ui| {
                    for (k, v) in &self.data.overview {
                        ui.label(egui::RichText::new(k).strong());
                        ui.label(v);
                        ui.end_row();
                    }
                });
        });
    }

    fn render_voronoi(&self, ctx: &egui::Context) {
        let canvas_frame = egui::Frame::none().fill(egui::Color32::WHITE);
        egui::CentralPanel::default()
            .frame(canvas_frame)
            .show(ctx, |ui| {
                let avail = ui.available_rect_before_wrap();
                let painter = ui.painter_at(avail);

                let transform = fit_transform(self.bbox, avail);
                let to_screen = |p: egui::Pos2| transform * p;

                let stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(40));
                for (i, poly) in self.cells.iter().enumerate() {
                    if poly.len() < 3 {
                        continue;
                    }
                    let fill = cell_color(i);
                    let screen_poly: Vec<egui::Pos2> =
                        poly.iter().copied().map(to_screen).collect();
                    painter.add(egui::Shape::convex_polygon(screen_poly, fill, stroke));
                }

                let response = ui.allocate_rect(avail, egui::Sense::hover());
                let hover_pos = response.hover_pos();
                let mut hovered: Option<usize> = None;
                for (i, c) in self.data.centroids.iter().enumerate() {
                    let p = to_screen(egui::pos2(c.point.x, c.point.y));
                    painter.circle_filled(p, 2.5, egui::Color32::BLACK);
                    if let Some(hp) = hover_pos {
                        if (hp - p).length() < 6.0 {
                            hovered = Some(i);
                        }
                    }
                }
                if let Some(i) = hovered {
                    let c = &self.data.centroids[i];
                    let p = to_screen(egui::pos2(c.point.x, c.point.y));
                    painter.circle_stroke(
                        p,
                        6.0,
                        egui::Stroke::new(1.8, egui::Color32::from_rgb(220, 60, 60)),
                    );
                    egui::show_tooltip_at_pointer(
                        ctx,
                        ui.layer_id(),
                        egui::Id::new(("centroid_tip", i)),
                        |ui| {
                            ui.label(format!("centroid #{i}"));
                            ui.label(format!("vectors: {}", c.count));
                            ui.label(format!("(x, y) = ({:.3}, {:.3})", c.point.x, c.point.y));
                        },
                    );
                }
            });
    }

    fn render_histogram(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(format!(
                "Vector count per centroid (sorted descending). N = {}, max = {}.",
                self.sorted_counts.len(),
                self.max_count
            ));
            ui.add_space(4.0);

            let avail = ui.available_rect_before_wrap();
            let painter = ui.painter_at(avail);
            painter.rect_filled(avail, 0.0, egui::Color32::from_gray(252));

            let n = self.sorted_counts.len();
            if n == 0 {
                return;
            }
            let bar_w = (avail.width() / n as f32).max(1.0);
            let h = avail.height() - 16.0;
            let max = self.max_count.max(1) as f32;

            let response = ui.allocate_rect(avail, egui::Sense::hover());
            let hover_pos = response.hover_pos();
            let hovered_bar: Option<usize> = hover_pos.and_then(|hp| {
                if !avail.contains(hp) {
                    return None;
                }
                let idx = ((hp.x - avail.left()) / bar_w).floor() as i64;
                if idx < 0 || idx as usize >= n {
                    None
                } else {
                    Some(idx as usize)
                }
            });

            for (i, &(c, _)) in self.sorted_counts.iter().enumerate() {
                let x = avail.left() + (i as f32) * bar_w;
                let bh = (c as f32 / max) * h;
                let rect = egui::Rect::from_min_size(
                    egui::pos2(x, avail.bottom() - bh - 8.0),
                    egui::vec2(bar_w.max(1.0), bh),
                );
                let color = if Some(i) == hovered_bar {
                    egui::Color32::from_rgb(220, 60, 60)
                } else {
                    egui::Color32::from_rgb(80, 130, 200)
                };
                painter.rect_filled(rect, 0.0, color);
            }
            painter.line_segment(
                [
                    egui::pos2(avail.left(), avail.bottom() - 8.0),
                    egui::pos2(avail.right(), avail.bottom() - 8.0),
                ],
                egui::Stroke::new(1.0, egui::Color32::from_gray(120)),
            );

            if let Some(i) = hovered_bar {
                let (count, orig_idx) = self.sorted_counts[i];
                egui::show_tooltip_at_pointer(
                    ctx,
                    ui.layer_id(),
                    egui::Id::new(("hist_tip", i)),
                    |ui| {
                        ui.label(format!("rank: {} / {}", i + 1, n));
                        ui.label(format!("centroid #{orig_idx}"));
                        ui.label(format!("vectors: {count}"));
                    },
                );
            }
        });
    }
}

/// Build a TSTransform that maps data-space `bbox` to fit inside `target`,
/// preserving aspect ratio and centering.
fn fit_transform(bbox: egui::Rect, target: egui::Rect) -> egui::emath::TSTransform {
    let bw = bbox.width().max(f32::MIN_POSITIVE);
    let bh = bbox.height().max(f32::MIN_POSITIVE);
    let pad = 16.0;
    let scale = ((target.width() - 2.0 * pad) / bw).min((target.height() - 2.0 * pad) / bh);
    // Translate bbox center to target center.
    let translation = target.center().to_vec2() - (bbox.center().to_vec2() * scale);
    egui::emath::TSTransform::new(translation, scale)
}

/// A pleasant pastel color per cell index. Uses the golden-ratio trick to
/// space hues evenly so cells with sequential indices (often spatially
/// adjacent) get visibly different colors.
fn cell_color(i: usize) -> egui::Color32 {
    const GOLDEN_RATIO_CONJ: f32 = 0.618_034;
    let h = ((i as f32) * GOLDEN_RATIO_CONJ).fract();
    hsl_to_rgb(h, 0.55, 0.80)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> egui::Color32 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    egui::Color32::from_rgb(
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}
