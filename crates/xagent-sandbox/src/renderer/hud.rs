use super::Vertex;

/// A colored bar for the HUD overlay, specified in NDC coordinates.
#[derive(Clone)]
pub struct HudBar {
    /// Left edge in NDC (-1.0 to 1.0).
    pub x: f32,
    /// Top edge in NDC (-1.0 to 1.0, where 1.0 is top).
    pub y: f32,
    /// Width in NDC units.
    pub width: f32,
    /// Height in NDC units (extends downward from y).
    pub height: f32,
    /// Fill ratio 0.0–1.0.
    pub fill: f32,
    /// Foreground RGB color.
    pub color: [f32; 3],
    /// Background RGB color.
    pub bg_color: [f32; 3],
}

/// Build combined vertex + index data for all HUD bars.
///
/// Each bar produces: border outline + background + filled foreground,
/// using the same `Vertex` layout as the 3D pipeline but with NDC positions.
pub fn build_hud_vertices(bars: &[HudBar]) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts: Vec<Vertex> = Vec::with_capacity(bars.len() * 12);
    let mut idxs: Vec<u32> = Vec::with_capacity(bars.len() * 18);

    let border = 0.003;
    let border_color = [0.6, 0.6, 0.6];

    for bar in bars {
        let x0 = bar.x;
        let x1 = bar.x + bar.width;
        let y0 = bar.y;
        let y1 = bar.y - bar.height;

        // Border quad (slightly larger)
        let bx0 = x0 - border;
        let bx1 = x1 + border;
        let by0 = y0 + border;
        let by1 = y1 - border;
        let base = verts.len() as u32;
        verts.push(Vertex { position: [bx0, by0, 0.0], color: border_color });
        verts.push(Vertex { position: [bx1, by0, 0.0], color: border_color });
        verts.push(Vertex { position: [bx1, by1, 0.0], color: border_color });
        verts.push(Vertex { position: [bx0, by1, 0.0], color: border_color });
        idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

        // Background quad
        let base = verts.len() as u32;
        verts.push(Vertex { position: [x0, y0, 0.0], color: bar.bg_color });
        verts.push(Vertex { position: [x1, y0, 0.0], color: bar.bg_color });
        verts.push(Vertex { position: [x1, y1, 0.0], color: bar.bg_color });
        verts.push(Vertex { position: [x0, y1, 0.0], color: bar.bg_color });
        idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

        // Foreground (filled) quad
        let fx1 = x0 + bar.fill.clamp(0.0, 1.0) * bar.width;
        let base = verts.len() as u32;
        verts.push(Vertex { position: [x0, y0, 0.0], color: bar.color });
        verts.push(Vertex { position: [fx1, y0, 0.0], color: bar.color });
        verts.push(Vertex { position: [fx1, y1, 0.0], color: bar.color });
        verts.push(Vertex { position: [x0, y1, 0.0], color: bar.color });
        idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    (verts, idxs)
}

/// Build a semi-transparent dark background panel quad.
pub fn build_panel_vertices(x: f32, y: f32, w: f32, h: f32) -> (Vec<Vertex>, Vec<u32>) {
    let color = [0.05, 0.05, 0.08];
    let x1 = x + w;
    let y1 = y - h;
    let verts = vec![
        Vertex { position: [x, y, 0.0], color },
        Vertex { position: [x1, y, 0.0], color },
        Vertex { position: [x1, y1, 0.0], color },
        Vertex { position: [x, y1, 0.0], color },
    ];
    let idxs = vec![0, 1, 2, 0, 2, 3];
    (verts, idxs)
}
