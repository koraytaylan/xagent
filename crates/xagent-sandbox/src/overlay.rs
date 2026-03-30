//! Visual overlay meshes: heatmap, trail ribbon, selection marker.
//!
//! These are diagnostic visualizations rendered on top of the 3D world.
//! Extracted from main.rs to keep the core application logic focused.

use crate::agent::HEATMAP_RES;
use crate::renderer::Vertex;
use crate::world::Mesh;

/// Build a flat-quad mesh showing the selected agent's position heatmap.
/// Each non-zero cell becomes a colored quad slightly above the terrain.
pub fn build_heatmap_mesh(
    heatmap: &[u32],
    world_size: f32,
    terrain: &crate::world::terrain::TerrainData,
) -> Mesh {
    let max_count = heatmap.iter().copied().max().unwrap_or(1).max(1) as f32;
    let half = world_size / 2.0;
    let cell = world_size / HEATMAP_RES as f32;
    let mut vertices = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for cz in 0..HEATMAP_RES {
        for cx in 0..HEATMAP_RES {
            let count = heatmap[cz * HEATMAP_RES + cx];
            if count == 0 {
                continue;
            }
            let t = (count as f32 / max_count).clamp(0.0, 1.0);
            // blue → yellow → red
            let color = if t < 0.5 {
                let s = t * 2.0;
                [s, s, 1.0 - s]
            } else {
                let s = (t - 0.5) * 2.0;
                [1.0, 1.0 - s, 0.0]
            };

            let x0 = -half + cx as f32 * cell;
            let z0 = -half + cz as f32 * cell;
            let xm = x0 + cell * 0.5;
            let zm = z0 + cell * 0.5;
            let y = terrain.height_at(xm, zm) + 0.5;

            let base = vertices.len() as u32;
            vertices.push(Vertex { position: [x0, y, z0], color });
            vertices.push(Vertex { position: [x0 + cell, y, z0], color });
            vertices.push(Vertex { position: [x0 + cell, y, z0 + cell], color });
            vertices.push(Vertex { position: [x0, y, z0 + cell], color });
            // CCW winding from above (normal pointing +Y)
            indices.extend_from_slice(&[
                base, base + 2, base + 1,
                base, base + 3, base + 2,
            ]);
        }
    }

    Mesh { vertices, indices }
}

/// Build a linear ribbon trail from the agent's distance-sampled control points.
/// Only rebuilt when the trail dirty flag is set.
pub fn build_trail_mesh(
    points: &[[f32; 3]],
    agent_color: &[f32; 3],
) -> Mesh {
    let n = points.len();
    if n < 2 {
        return Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    let num_segments = n - 1;
    let mut vertices = Vec::with_capacity(num_segments * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(num_segments * 6);

    let ribbon_half_width: f32 = 0.3;
    let y_offset: f32 = 0.3;

    for i in 0..num_segments {
        let a = points[i];
        let b = points[i + 1];

        // Gentle fade: oldest 20% fades from 0.3→0.7, rest is 0.7
        let progress = i as f32 / num_segments as f32;
        let brightness = if progress < 0.2 {
            0.3 + (progress / 0.2) * 0.4
        } else {
            0.7
        };
        let color = [
            agent_color[0] * brightness,
            agent_color[1] * brightness,
            agent_color[2] * brightness,
        ];

        // Perpendicular in XZ plane for ribbon width
        let dx = b[0] - a[0];
        let dz = b[2] - a[2];
        let len = (dx * dx + dz * dz).sqrt().max(0.001);
        let px = -dz / len * ribbon_half_width;
        let pz = dx / len * ribbon_half_width;

        let base = vertices.len() as u32;
        vertices.push(Vertex {
            position: [a[0] + px, a[1] + y_offset, a[2] + pz],
            color,
        });
        vertices.push(Vertex {
            position: [a[0] - px, a[1] + y_offset, a[2] - pz],
            color,
        });
        vertices.push(Vertex {
            position: [b[0] - px, b[1] + y_offset, b[2] - pz],
            color,
        });
        vertices.push(Vertex {
            position: [b[0] + px, b[1] + y_offset, b[2] + pz],
            color,
        });

        indices.extend_from_slice(&[
            base, base + 2, base + 1,
            base, base + 3, base + 2,
        ]);
    }

    Mesh { vertices, indices }
}

/// Build a small diamond marker hovering above the given position.
pub fn build_marker_mesh(position: glam::Vec3) -> Mesh {
    let cx = position.x;
    let cy = position.y + 5.0;
    let cz = position.z;
    let r: f32 = 1.2;
    let h: f32 = 1.8;
    let color = [1.0, 1.0, 0.2];

    let top = [cx, cy + h * 0.5, cz];
    let bot = [cx, cy - h * 0.5, cz];
    let n = [cx, cy, cz - r];
    let s = [cx, cy, cz + r];
    let e = [cx + r, cy, cz];
    let w = [cx - r, cy, cz];

    let shade_top = [color[0], color[1], color[2]];
    let shade_side = [color[0] * 0.75, color[1] * 0.75, color[2] * 0.75];
    let shade_bot = [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5];

    let mut vertices = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(24);

    let upper_faces = [(n, e), (e, s), (s, w), (w, n)];
    for (a, b) in &upper_faces {
        let base = vertices.len() as u32;
        vertices.push(Vertex { position: top, color: shade_top });
        vertices.push(Vertex { position: *a, color: shade_side });
        vertices.push(Vertex { position: *b, color: shade_side });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    let lower_faces = [(e, n), (s, e), (w, s), (n, w)];
    for (a, b) in &lower_faces {
        let base = vertices.len() as u32;
        vertices.push(Vertex { position: bot, color: shade_bot });
        vertices.push(Vertex { position: *a, color: shade_side });
        vertices.push(Vertex { position: *b, color: shade_side });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    Mesh { vertices, indices }
}
