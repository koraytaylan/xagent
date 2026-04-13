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
            vertices.push(Vertex {
                position: [x0, y, z0],
                color,
            });
            vertices.push(Vertex {
                position: [x0 + cell, y, z0],
                color,
            });
            vertices.push(Vertex {
                position: [x0 + cell, y, z0 + cell],
                color,
            });
            vertices.push(Vertex {
                position: [x0, y, z0 + cell],
                color,
            });
            // CCW winding from above (normal pointing +Y)
            indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        }
    }

    Mesh { vertices, indices }
}

const MAX_SUBDIVISIONS: usize = 8;

/// Subdivide a trail's control points using Catmull-Rom interpolation.
/// Returns a denser polyline that smooths sharp corners.
fn catmull_rom_subdivide(points: &[[f32; 3]]) -> std::borrow::Cow<'_, [[f32; 3]]> {
    if points.len() < 3 {
        return std::borrow::Cow::Borrowed(points);
    }

    let min_trail_dist_sq = crate::agent::MIN_TRAIL_DIST_SQ;

    let mut result = Vec::with_capacity((points.len() - 1) * MAX_SUBDIVISIONS + 1);
    for i in 0..points.len() - 1 {
        let p0 = points[i.saturating_sub(1)];
        let p1 = points[i];
        let p2 = points[(i + 1).min(points.len() - 1)];
        let p3 = points[(i + 2).min(points.len() - 1)];

        let dist_sq = (p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2);

        let mut subdivisions = MAX_SUBDIVISIONS;
        for k in 1..MAX_SUBDIVISIONS {
            if dist_sq <= (k * k) as f32 * min_trail_dist_sq {
                subdivisions = k;
                break;
            }
        }

        for s in 0..subdivisions {
            let t = s as f32 / subdivisions as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            // Catmull-Rom matrix coefficients
            let x = 0.5
                * ((2.0 * p1[0])
                    + (-p0[0] + p2[0]) * t
                    + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3);
            let y = 0.5
                * ((2.0 * p1[1])
                    + (-p0[1] + p2[1]) * t
                    + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3);
            let z = 0.5
                * ((2.0 * p1[2])
                    + (-p0[2] + p2[2]) * t
                    + (2.0 * p0[2] - 5.0 * p1[2] + 4.0 * p2[2] - p3[2]) * t2
                    + (-p0[2] + 3.0 * p1[2] - 3.0 * p2[2] + p3[2]) * t3);
            result.push([x, y, z]);
        }
    }
    result.push(*points.last().unwrap());
    std::borrow::Cow::Owned(result)
}

/// Build a linear ribbon trail from the agent's distance-sampled control points.
/// Only rebuilt when the trail dirty flag is set.
pub fn build_trail_mesh(points: &[[f32; 3]], agent_color: &[f32; 3]) -> Mesh {
    let n = points.len();
    if n < 2 {
        return Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    let subdivided_points = catmull_rom_subdivide(points);
    let n = subdivided_points.len();

    let num_segments = n - 1;
    let mut vertices = Vec::with_capacity(num_segments * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(num_segments * 6);

    let ribbon_half_width: f32 = 0.3;
    let y_offset: f32 = 0.3;

    for i in 0..num_segments {
        let a = subdivided_points[i];
        let b = subdivided_points[i + 1];

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

        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }

    Mesh { vertices, indices }
}

/// Build a combined ribbon trail for every agent in the simulation.
/// Each agent's trail is coloured with its own colour.  Returns an empty mesh
/// if no agent has ≥2 trail points.
pub fn build_all_trails_mesh(
    agents: &[(&[[f32; 3]], &[f32; 3], bool)], // (trail, color, alive)
) -> Mesh {
    // First pass: count total segments across all agents.
    let total_segments: usize = agents
        .iter()
        .filter(|(trail, _, alive)| *alive && trail.len() > 1)
        .map(|(trail, _, _)| trail.len() - 1)
        .sum();

    if total_segments == 0 {
        return Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    let mut vertices = Vec::with_capacity(total_segments.saturating_mul(4));
    let mut indices: Vec<u32> = Vec::with_capacity(total_segments.saturating_mul(6));

    let ribbon_half_width: f32 = 0.3;
    let y_offset: f32 = 0.3;

    for &(trail, agent_color, alive) in agents {
        if !alive || trail.len() < 2 {
            continue;
        }

        let subdivided_trail = catmull_rom_subdivide(trail);
        let num_segments = subdivided_trail.len() - 1;

        vertices.reserve(num_segments * 4);
        indices.reserve(num_segments * 6);

        for i in 0..num_segments {
            let a = subdivided_trail[i];
            let b = subdivided_trail[i + 1];

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

            indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        }
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
        vertices.push(Vertex {
            position: top,
            color: shade_top,
        });
        vertices.push(Vertex {
            position: *a,
            color: shade_side,
        });
        vertices.push(Vertex {
            position: *b,
            color: shade_side,
        });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    let lower_faces = [(e, n), (s, e), (w, s), (n, w)];
    for (a, b) in &lower_faces {
        let base = vertices.len() as u32;
        vertices.push(Vertex {
            position: bot,
            color: shade_bot,
        });
        vertices.push(Vertex {
            position: *a,
            color: shade_side,
        });
        vertices.push(Vertex {
            position: *b,
            color: shade_side,
        });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    Mesh { vertices, indices }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white() -> [f32; 3] {
        [1.0, 1.0, 1.0]
    }

    // ── build_trail_mesh ─────────────────────────────────────────────

    #[test]
    fn catmull_rom_subdivide_preserves_endpoints() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]];
        let result = catmull_rom_subdivide(&points);
        assert_eq!(result.first().unwrap(), &points[0]);
        assert_eq!(result.last().unwrap(), points.last().unwrap());
    }

    #[test]
    fn catmull_rom_subdivide_increases_point_count() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
            [30.0, 15.0, 0.0],
            [45.0, 15.0, 0.0],
        ];
        let result = catmull_rom_subdivide(&points);
        assert!(result.len() > points.len());
    }

    #[test]
    fn catmull_rom_subdivide_short_input_unchanged() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let result = catmull_rom_subdivide(&points);
        assert_eq!(result.len(), points.len());
    }

    #[test]
    fn trail_mesh_empty_on_zero_points() {
        let m = build_trail_mesh(&[], &white());
        assert!(m.vertices.is_empty());
        assert!(m.indices.is_empty());
    }

    #[test]
    fn trail_mesh_empty_on_single_point() {
        let m = build_trail_mesh(&[[0.0, 0.0, 0.0]], &white());
        assert!(m.vertices.is_empty());
    }

    #[test]
    fn trail_mesh_one_segment() {
        let pts = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let m = build_trail_mesh(&pts, &white());
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.indices.len(), 6);
    }

    #[test]
    fn trail_mesh_three_points_two_segments() {
        let pts = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let m = build_trail_mesh(&pts, &white());

        let expected_segments = catmull_rom_subdivide(&pts).len() - 1;
        assert_eq!(m.vertices.len(), expected_segments * 4);
        assert_eq!(m.indices.len(), expected_segments * 6);

        // Prove subdivision was actually applied
        assert!(expected_segments > pts.len() - 1);
    }

    #[test]
    fn trail_mesh_applies_agent_color() {
        let pts = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let red = [1.0, 0.0, 0.0];
        let m = build_trail_mesh(&pts, &red);
        // Single segment → oldest 20% fade region → brightness ≈ 0.3
        assert!(m.vertices[0].color[0] > 0.0); // red channel present
        assert_eq!(m.vertices[0].color[1], 0.0); // green zero
        assert_eq!(m.vertices[0].color[2], 0.0); // blue zero
    }

    // ── build_all_trails_mesh ────────────────────────────────────────

    #[test]
    fn all_trails_empty_when_no_agents() {
        let m = build_all_trails_mesh(&[]);
        assert!(m.vertices.is_empty());
        assert!(m.indices.is_empty());
    }

    #[test]
    fn all_trails_skips_dead_agents() {
        let trail = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let c = white();
        let agents: Vec<(&[[f32; 3]], &[f32; 3], bool)> = vec![(trail.as_slice(), &c, false)];
        let m = build_all_trails_mesh(&agents);
        assert!(m.indices.is_empty());
    }

    #[test]
    fn all_trails_skips_short_trails() {
        let trail = vec![[0.0, 0.0, 0.0]]; // only 1 point
        let c = white();
        let agents: Vec<(&[[f32; 3]], &[f32; 3], bool)> = vec![(trail.as_slice(), &c, true)];
        let m = build_all_trails_mesh(&agents);
        assert!(m.indices.is_empty());
    }

    #[test]
    fn all_trails_combines_two_agents() {
        let t1 = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]; // 1 segment (too short for subdiv)
        let t2 = vec![[0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [10.0, 0.0, 5.0]]; // 4 segments after subdiv
        let red = [1.0, 0.0, 0.0];
        let blue = [0.0, 0.0, 1.0];
        let agents: Vec<(&[[f32; 3]], &[f32; 3], bool)> =
            vec![(t1.as_slice(), &red, true), (t2.as_slice(), &blue, true)];
        let m = build_all_trails_mesh(&agents);

        let expected_segments =
            (catmull_rom_subdivide(&t1).len() - 1) + (catmull_rom_subdivide(&t2).len() - 1);

        assert_eq!(m.vertices.len(), expected_segments * 4);
        assert_eq!(m.indices.len(), expected_segments * 6);

        let original_segments = (t1.len() - 1) + (t2.len() - 1);
        // Prove subdivision was actually applied
        assert!(expected_segments > original_segments);
    }
    #[test]
    fn all_trails_mixed_alive_and_dead() {
        let t1 = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let t2 = vec![[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]];
        let c = white();
        let agents: Vec<(&[[f32; 3]], &[f32; 3], bool)> = vec![
            (t1.as_slice(), &c, true),  // alive → included
            (t2.as_slice(), &c, false), // dead → skipped
        ];
        let m = build_all_trails_mesh(&agents);
        assert_eq!(m.vertices.len(), 4); // only agent 1
        assert_eq!(m.indices.len(), 6);
    }

    #[test]
    fn all_trails_uses_each_agents_color() {
        let t1 = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let t2 = vec![[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]];
        let red = [1.0, 0.0, 0.0];
        let blue = [0.0, 0.0, 1.0];
        let agents: Vec<(&[[f32; 3]], &[f32; 3], bool)> =
            vec![(t1.as_slice(), &red, true), (t2.as_slice(), &blue, true)];
        let m = build_all_trails_mesh(&agents);
        // First 4 verts belong to agent 1 (red), next 4 to agent 2 (blue)
        assert!(m.vertices[0].color[0] > 0.0); // red channel
        assert_eq!(m.vertices[0].color[2], 0.0); // no blue
        assert_eq!(m.vertices[4].color[0], 0.0); // no red
        assert!(m.vertices[4].color[2] > 0.0); // blue channel
    }
}
