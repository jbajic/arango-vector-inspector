//! egui-based interactive inspector. Multi-tab UI per index, with a top-level
//! index selector. The Voronoi tab supports pan/zoom, several color modes,
//! and click-to-select with a detail panel.

use anyhow::Result;
use eframe::egui;
use voronator::VoronoiDiagram;
use voronator::delaunator::Point;

use crate::projection::Point2;

pub struct CentroidView {
    pub point: Point2,
    pub count: u64,
}

/// Everything the UI needs to render one vector index.
pub struct IndexView {
    pub title: String,
    pub overview: Vec<(String, String)>,
    pub centroids: Vec<CentroidView>,
    /// Original high-dim centroids, used for nearest-neighbor lookups on
    /// the detail panel. Same order as `centroids`.
    pub high_dim: Vec<Vec<f32>>,
    pub cells: Vec<Vec<egui::Pos2>>,
    pub cell_areas: Vec<f64>,
    pub bbox: egui::Rect,
    pub max_count: u64,
    /// (count, original_centroid_index) sorted by count descending.
    pub sorted_counts: Vec<(u64, usize)>,
}

pub struct ViewerData {
    pub indexes: Vec<IndexView>,
}

pub fn run(data: ViewerData) -> Result<()> {
    if data.indexes.is_empty() {
        return Err(anyhow::anyhow!("no indexes to display"));
    }
    let app = ViewerApp::new(data);
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 850.0]),
        ..Default::default()
    };
    eframe::run_native(
        "arango-vector-inspector",
        native_options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;
    Ok(())
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Tab {
    Overview,
    Voronoi,
    Histogram,
    AreaCount,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum ColorMode {
    /// Pastel hue per cell index (no semantic meaning).
    Classical,
    /// Viridis-like ramp on vector count.
    ByCount,
    /// Highlight only dead cells (count == 0); others muted gray.
    DeadOnly,
}

struct ViewerApp {
    data: ViewerData,
    selected_idx: usize,
    tab: Tab,
    color_mode: ColorMode,
    /// Pan/zoom transform applied to the Voronoi canvas (data → screen).
    voronoi_xform: egui::emath::TSTransform,
    /// True after the first frame, so we can lazily auto-fit when the user
    /// switches indexes.
    voronoi_xform_ready: bool,
    /// Centroid currently picked by clicking on a cell.
    selected_cell: Option<usize>,
}

impl ViewerApp {
    fn new(data: ViewerData) -> Self {
        Self {
            data,
            selected_idx: 0,
            tab: Tab::Overview,
            color_mode: ColorMode::Classical,
            voronoi_xform: egui::emath::TSTransform::IDENTITY,
            voronoi_xform_ready: false,
            selected_cell: None,
        }
    }

    fn current(&self) -> &IndexView {
        &self.data.indexes[self.selected_idx]
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::light());

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(&self.current().title);
                ui.add_space(20.0);
                if self.data.indexes.len() > 1 {
                    let prev = self.selected_idx;
                    egui::ComboBox::from_id_source("index_picker")
                        .selected_text(&self.current().title)
                        .show_ui(ui, |ui| {
                            for (i, ix) in self.data.indexes.iter().enumerate() {
                                ui.selectable_value(&mut self.selected_idx, i, &ix.title);
                            }
                        });
                    if prev != self.selected_idx {
                        // Reset interaction state on switch.
                        self.voronoi_xform_ready = false;
                        self.selected_cell = None;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Overview, "Overview");
                ui.selectable_value(&mut self.tab, Tab::Voronoi, "Voronoi");
                ui.selectable_value(&mut self.tab, Tab::Histogram, "Histogram");
                ui.selectable_value(&mut self.tab, Tab::AreaCount, "Area vs Count");
            });
        });

        match self.tab {
            Tab::Overview => self.render_overview(ctx),
            Tab::Voronoi => self.render_voronoi(ctx),
            Tab::Histogram => self.render_histogram(ctx),
            Tab::AreaCount => self.render_area_count(ctx),
        }
    }
}

impl ViewerApp {
    fn render_overview(&self, ctx: &egui::Context) {
        let view = self.current();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(6.0);
            egui::Grid::new("overview_kvs")
                .striped(true)
                .num_columns(2)
                .spacing([20.0, 4.0])
                .show(ui, |ui| {
                    for (k, v) in &view.overview {
                        ui.label(egui::RichText::new(k).strong());
                        ui.label(v);
                        ui.end_row();
                    }
                });
        });
    }

    fn render_voronoi(&mut self, ctx: &egui::Context) {
        let canvas_frame = egui::Frame::none().fill(egui::Color32::WHITE);
        egui::CentralPanel::default()
            .frame(canvas_frame)
            .show(ctx, |ui| {
                // --- toolbar ---
                ui.horizontal(|ui| {
                    ui.label("Color:");
                    ui.selectable_value(&mut self.color_mode, ColorMode::Classical, "Classical");
                    ui.selectable_value(&mut self.color_mode, ColorMode::ByCount, "By count");
                    ui.selectable_value(&mut self.color_mode, ColorMode::DeadOnly, "Dead only");
                    ui.separator();
                    if ui.button("Reset view").clicked() {
                        self.voronoi_xform_ready = false;
                        self.selected_cell = None;
                    }
                    ui.label("(drag to pan • scroll to zoom • click a cell)");
                });

                // --- canvas (egui's idiomatic canvas pattern) ---
                let (response, painter) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
                let avail = response.rect;

                // Initial fit (and after Reset).
                if !self.voronoi_xform_ready {
                    self.voronoi_xform = fit_transform(self.current().bbox, avail);
                    self.voronoi_xform_ready = true;
                }

                // --- input: pan / zoom / click ---
                if response.dragged() {
                    self.voronoi_xform.translation += response.drag_delta();
                }
                if response.hovered() {
                    let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.001 {
                        let zoom = (scroll * 0.005).exp();
                        let anchor = response.hover_pos().unwrap_or_else(|| avail.center());
                        let new_scale = (self.voronoi_xform.scaling * zoom).clamp(1e-3, 1e6);
                        let data_anchor = inverse_to_data(anchor, self.voronoi_xform);
                        self.voronoi_xform.translation =
                            anchor.to_vec2() - data_anchor.to_vec2() * new_scale;
                        self.voronoi_xform.scaling = new_scale;
                    }
                }
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let view = self.current();
                        let mut best: Option<(usize, f32)> = None;
                        for (i, c) in view.centroids.iter().enumerate() {
                            let p = self.voronoi_xform * egui::pos2(c.point.x, c.point.y);
                            let d = (p - pos).length();
                            if best.map_or(true, |(_, bd)| d < bd) {
                                best = Some((i, d));
                            }
                        }
                        self.selected_cell = best.map(|(i, _)| i);
                    }
                }

                // --- paint cells ---
                let view = self.current();
                let xform = self.voronoi_xform;
                let stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(40));
                for (i, poly) in view.cells.iter().enumerate() {
                    if poly.len() < 3 {
                        continue;
                    }
                    let count = view.centroids[i].count;
                    let fill = match self.color_mode {
                        ColorMode::Classical => cell_color_classical(i),
                        ColorMode::ByCount => cell_color_by_count(count, view.max_count),
                        ColorMode::DeadOnly => cell_color_dead_only(count),
                    };
                    let screen: Vec<egui::Pos2> = poly.iter().map(|&p| xform * p).collect();
                    painter.add(egui::Shape::convex_polygon(screen, fill, stroke));
                }

                // --- centroid markers + hover ring + tooltip ---
                let hover_pos = response.hover_pos();
                let mut hovered: Option<usize> = None;
                for (i, c) in view.centroids.iter().enumerate() {
                    let p = xform * egui::pos2(c.point.x, c.point.y);
                    painter.circle_filled(p, 2.5, egui::Color32::BLACK);
                    if let Some(hp) = hover_pos {
                        if avail.contains(hp) && (hp - p).length() < 8.0 {
                            hovered = Some(i);
                        }
                    }
                }
                if let Some(i) = self.selected_cell {
                    let c = &view.centroids[i];
                    let p = xform * egui::pos2(c.point.x, c.point.y);
                    painter.circle_stroke(p, 7.0, egui::Stroke::new(2.0, egui::Color32::BLUE));
                }
                if let Some(i) = hovered {
                    let c = &view.centroids[i];
                    let p = xform * egui::pos2(c.point.x, c.point.y);
                    painter.circle_stroke(
                        p,
                        6.0,
                        egui::Stroke::new(1.6, egui::Color32::from_rgb(220, 60, 60)),
                    );
                    egui::show_tooltip_at_pointer(
                        ui.ctx(),
                        ui.layer_id(),
                        egui::Id::new(("voronoi_tip", i)),
                        |ui| {
                            ui.label(format!("centroid #{i}"));
                            ui.label(format!("vectors: {}", c.count));
                        },
                    );
                }

                // --- detail panel for selected cell ---
                if let Some(i) = self.selected_cell {
                    self.paint_detail_panel(&painter, avail, i);
                }
            });
    }

    fn paint_detail_panel(&self, painter: &egui::Painter, avail: egui::Rect, i: usize) {
        let view = self.current();
        let panel_w = 240.0_f32.min(avail.width() * 0.3);
        let panel = egui::Rect::from_min_size(
            egui::pos2(avail.right() - panel_w - 12.0, avail.top() + 12.0),
            egui::vec2(panel_w, 160.0),
        );
        painter.rect(
            panel,
            6.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 235),
            egui::Stroke::new(1.0, egui::Color32::from_gray(160)),
        );

        // Compute rank (1-indexed).
        let rank = view
            .sorted_counts
            .iter()
            .position(|&(_, idx)| idx == i)
            .map(|p| p + 1)
            .unwrap_or(0);
        let count = view.centroids[i].count;

        // Nearest neighbor in original high-dim space.
        let nn = nearest_neighbor(&view.high_dim, i);

        let lines = [
            format!("Centroid #{i}"),
            format!(
                "vectors: {count}    rank: {rank} / {}",
                view.centroids.len()
            ),
            format!("cell area (2D): {:.2}", view.cell_areas[i]),
            match nn {
                Some((j, d)) => format!("nearest centroid: #{j} (d={d:.3})"),
                None => "nearest centroid: n/a".into(),
            },
        ];
        let mut y = panel.top() + 12.0;
        for line in &lines {
            painter.text(
                egui::pos2(panel.left() + 10.0, y),
                egui::Align2::LEFT_TOP,
                line,
                egui::FontId::proportional(13.0),
                egui::Color32::BLACK,
            );
            y += 22.0;
        }
    }

    fn render_histogram(&self, ctx: &egui::Context) {
        let view = self.current();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(format!(
                "Vector count per centroid (sorted descending). N = {}, max = {}.",
                view.sorted_counts.len(),
                view.max_count
            ));
            ui.add_space(4.0);

            let avail = ui.available_rect_before_wrap();
            let painter = ui.painter_at(avail);
            painter.rect_filled(avail, 0.0, egui::Color32::from_gray(252));

            let n = view.sorted_counts.len();
            if n == 0 {
                return;
            }
            let bar_w = (avail.width() / n as f32).max(1.0);
            let h = avail.height() - 16.0;
            let max = view.max_count.max(1) as f32;

            let response = ui.allocate_rect(avail, egui::Sense::hover());
            let hover_pos = response.hover_pos();
            let hovered_bar = hover_pos.and_then(|hp| {
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

            for (i, &(c, _)) in view.sorted_counts.iter().enumerate() {
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
                let (count, orig_idx) = view.sorted_counts[i];
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

    fn render_area_count(&self, ctx: &egui::Context) {
        let view = self.current();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(
                "Cell area (2D) vs vector count. Healthy IVF: roughly proportional. \
                 Outliers above the line = sparse (overprovisioned) cells; \
                 below the line = dense (overloaded) cells.",
            );
            ui.add_space(4.0);

            let avail = ui.available_rect_before_wrap();
            let painter = ui.painter_at(avail);
            painter.rect_filled(avail, 0.0, egui::Color32::from_gray(252));

            let pad = 40.0;
            let plot = egui::Rect::from_min_max(
                egui::pos2(avail.left() + pad, avail.top() + 10.0),
                egui::pos2(avail.right() - 10.0, avail.bottom() - pad),
            );
            painter.rect_stroke(
                plot,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(150)),
            );

            // Build the (count, area) cloud, skip degenerate cells.
            let pts: Vec<(f64, f64, usize)> = view
                .centroids
                .iter()
                .zip(view.cell_areas.iter())
                .enumerate()
                .filter_map(|(i, (c, &a))| {
                    if a > 0.0 {
                        Some((c.count as f64, a, i))
                    } else {
                        None
                    }
                })
                .collect();
            if pts.is_empty() {
                return;
            }
            let xmax = pts.iter().map(|p| p.0).fold(1.0_f64, f64::max);
            let ymax = pts.iter().map(|p| p.1).fold(1.0_f64, f64::max);

            let response = ui.allocate_rect(avail, egui::Sense::hover());
            let hover_pos = response.hover_pos();
            let mut hovered: Option<(usize, f64, f64)> = None;
            let to_screen = |x: f64, y: f64| -> egui::Pos2 {
                let nx = (x / xmax) as f32;
                let ny = (y / ymax) as f32;
                egui::pos2(
                    plot.left() + nx * plot.width(),
                    plot.bottom() - ny * plot.height(),
                )
            };

            for &(x, y, idx) in &pts {
                let p = to_screen(x, y);
                painter.circle_filled(p, 3.0, egui::Color32::from_rgb(70, 110, 180));
                if let Some(hp) = hover_pos {
                    if (hp - p).length() < 6.0 {
                        hovered = Some((idx, x, y));
                    }
                }
            }

            // Y-label / X-label
            painter.text(
                egui::pos2(plot.left() - 8.0, plot.center().y),
                egui::Align2::RIGHT_CENTER,
                format!("area  (max {ymax:.1})"),
                egui::FontId::proportional(11.0),
                egui::Color32::from_gray(80),
            );
            painter.text(
                egui::pos2(plot.center().x, plot.bottom() + 18.0),
                egui::Align2::CENTER_TOP,
                format!("vectors  (max {})", xmax as u64),
                egui::FontId::proportional(11.0),
                egui::Color32::from_gray(80),
            );

            if let Some((i, x, y)) = hovered {
                let p = to_screen(x, y);
                painter.circle_stroke(
                    p,
                    6.0,
                    egui::Stroke::new(1.6, egui::Color32::from_rgb(220, 60, 60)),
                );
                egui::show_tooltip_at_pointer(
                    ctx,
                    ui.layer_id(),
                    egui::Id::new(("ac_tip", i)),
                    |ui| {
                        ui.label(format!("centroid #{i}"));
                        ui.label(format!("vectors: {}", x as u64));
                        ui.label(format!("cell area: {y:.2}"));
                    },
                );
            }
        });
    }
}

// ---- Voronoi computation -------------------------------------------------

/// Build Voronoi cells from 2D centroid positions. Returns one polygon per
/// input centroid (in matching order, possibly empty if degenerate), the
/// per-cell areas (zero for degenerate cells), and the bounding box used.
pub fn build_cells(points_2d: &[Point2]) -> (Vec<Vec<egui::Pos2>>, Vec<f64>, egui::Rect) {
    if points_2d.is_empty() {
        return (Vec::new(), Vec::new(), egui::Rect::ZERO);
    }
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for p in points_2d {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
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

    let pts: Vec<Point> = points_2d
        .iter()
        .map(|p| Point {
            x: p.x as f64,
            y: p.y as f64,
        })
        .collect();

    let diagram = VoronoiDiagram::<Point>::new(&lo, &hi, &pts).expect("Voronoi computation failed");

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
    let areas: Vec<f64> = cells.iter().map(|poly| polygon_area(poly)).collect();

    let bbox = egui::Rect::from_min_max(
        egui::pos2(lo.x as f32, lo.y as f32),
        egui::pos2(hi.x as f32, hi.y as f32),
    );
    (cells, areas, bbox)
}

/// Shoelace formula. Returns absolute area.
fn polygon_area(poly: &[egui::Pos2]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        sum += a.x as f64 * b.y as f64 - b.x as f64 * a.y as f64;
    }
    (sum * 0.5).abs()
}

fn nearest_neighbor(centroids: &[Vec<f32>], i: usize) -> Option<(usize, f32)> {
    if centroids.len() < 2 || i >= centroids.len() {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    let me = &centroids[i];
    for (j, other) in centroids.iter().enumerate() {
        if j == i {
            continue;
        }
        let d2: f32 = me
            .iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let d = d2.sqrt();
        if best.map_or(true, |(_, bd)| d < bd) {
            best = Some((j, d));
        }
    }
    best
}

// ---- Layout helpers ------------------------------------------------------

fn inverse_to_data(screen: egui::Pos2, xform: egui::emath::TSTransform) -> egui::Pos2 {
    let v = (screen.to_vec2() - xform.translation) / xform.scaling;
    v.to_pos2()
}

fn fit_transform(bbox: egui::Rect, target: egui::Rect) -> egui::emath::TSTransform {
    let bw = bbox.width().max(f32::MIN_POSITIVE);
    let bh = bbox.height().max(f32::MIN_POSITIVE);
    let pad = 32.0;
    let scale = ((target.width() - 2.0 * pad) / bw).min((target.height() - 2.0 * pad) / bh);
    let translation = target.center().to_vec2() - (bbox.center().to_vec2() * scale);
    egui::emath::TSTransform::new(translation, scale)
}

// ---- Color schemes -------------------------------------------------------

fn cell_color_classical(i: usize) -> egui::Color32 {
    const GOLDEN_RATIO_CONJ: f32 = 0.618_034;
    let h = ((i as f32) * GOLDEN_RATIO_CONJ).fract();
    hsl_to_rgb(h, 0.55, 0.80)
}

fn cell_color_by_count(count: u64, max: u64) -> egui::Color32 {
    if count == 0 {
        return egui::Color32::from_gray(230);
    }
    let t = (count as f32 / max.max(1) as f32).clamp(0.0, 1.0);
    // Cheap viridis-ish: dark navy → teal → lime.
    let r = (40.0 + t * 200.0) as u8;
    let g = (40.0 + t.sqrt() * 200.0) as u8;
    let b = (110.0 + (1.0 - t) * 100.0) as u8;
    egui::Color32::from_rgb(r, g, b)
}

fn cell_color_dead_only(count: u64) -> egui::Color32 {
    if count == 0 {
        egui::Color32::from_rgb(220, 70, 70)
    } else {
        egui::Color32::from_gray(238)
    }
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
