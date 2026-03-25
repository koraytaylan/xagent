use glam::{Mat4, Vec3};

/// Free-flying spectator camera for observing the simulation.
///
/// Uses Euler angles (yaw/pitch) for orientation. WASD + mouse drag controls.
/// The camera exists purely for observation — agents cannot see through it.
/// Coordinate system: right-handed, Y-up, initial position looks toward -Z.
pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub aspect: f32,
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,

    // Input state
    pub move_forward: bool,
    pub move_backward: bool,
    pub move_left: bool,
    pub move_right: bool,
    pub move_up: bool,
    pub move_down: bool,

    pub is_mouse_dragging: bool,
    pub last_mouse_pos: Option<(f64, f64)>,
}

impl Camera {
    /// Create a camera at position (0, 10, 20) looking roughly toward the world origin.
    ///
    /// Initial yaw is -π/2 (facing -Z) with a slight downward pitch (-0.3 rad ≈ -17°)
    /// to give a good overview of the terrain below.
    pub fn new(aspect: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 10.0, 20.0),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.3,
            aspect,
            fov_y: 45.0_f32.to_radians(),
            z_near: 0.1,
            z_far: 500.0,
            move_forward: false,
            move_backward: false,
            move_left: false,
            move_right: false,
            move_up: false,
            move_down: false,
            is_mouse_dragging: false,
            last_mouse_pos: None,
        }
    }

    /// Compute the camera's forward direction vector from yaw and pitch angles.
    pub fn front(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    /// Compute the camera's right direction vector (perpendicular to front and world-up).
    pub fn right(&self) -> Vec3 {
        self.front().cross(Vec3::Y).normalize()
    }

    /// Update camera position based on active movement keys.
    /// Movement speed is 60 units/s, applied in the camera's local coordinate frame.
    pub fn update(&mut self, dt: f32) {
        let speed = 60.0 * dt;
        let front = self.front();
        let right = self.right();

        if self.move_forward {
            self.position += front * speed;
        }
        if self.move_backward {
            self.position -= front * speed;
        }
        if self.move_left {
            self.position -= right * speed;
        }
        if self.move_right {
            self.position += right * speed;
        }
        if self.move_up {
            self.position += Vec3::Y * speed;
        }
        if self.move_down {
            self.position -= Vec3::Y * speed;
        }
    }

    /// Process mouse movement for look-around. Only active when `is_mouse_dragging` is true.
    /// Sensitivity is 0.003 rad/pixel. Pitch is clamped to ±89° to prevent gimbal lock.
    pub fn process_mouse_move(&mut self, x: f64, y: f64) {
        if !self.is_mouse_dragging {
            self.last_mouse_pos = None;
            return;
        }

        if let Some((last_x, last_y)) = self.last_mouse_pos {
            let sensitivity = 0.003;
            let dx = (x - last_x) as f32;
            let dy = (y - last_y) as f32;
            self.yaw += dx * sensitivity;
            self.pitch -= dy * sensitivity;

            let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
            self.pitch = self.pitch.clamp(-max_pitch, max_pitch);
        }
        self.last_mouse_pos = Some((x, y));
    }

    /// Scroll-wheel zoom: moves the camera forward/backward along its look direction.
    pub fn process_scroll(&mut self, delta: f32) {
        let front = self.front();
        self.position += front * delta * 2.0;
    }

    /// Reset camera to the default overview position above the world origin.
    pub fn reset(&mut self) {
        self.position = Vec3::new(0.0, 10.0, 20.0);
        self.yaw = -std::f32::consts::FRAC_PI_2;
        self.pitch = -0.3;
    }

    /// Compute the view matrix (world-to-camera transform) using right-handed look-at.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.front(), Vec3::Y)
    }

    /// Compute the perspective projection matrix. FOV is 45°, clip planes at 0.1–500 units.
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far)
    }

    /// Combined view-projection matrix for transforming world positions to clip space.
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
}
