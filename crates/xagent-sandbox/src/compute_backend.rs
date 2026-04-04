//! Runtime compute backend selection.
//!
//! Probes for GPU at startup and selects the highest-capability tier.
//! Falls back to CPU with rayon parallelism if no GPU is available.

use log::info;

/// Available compute backend tiers, ordered by capability.
pub enum ComputeBackend {
    /// Rayon + spatial grid (always available)
    CpuOptimized,
    /// Full GPU compute pipeline (vision + encode + recall)
    GpuAccelerated {
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
    },
}

impl ComputeBackend {
    /// Probe the system and return the highest available backend.
    pub fn probe() -> Self {
        if let Some(backend) = Self::try_gpu() {
            backend
        } else {
            info!("[xagent] Compute backend: CpuOptimized (no GPU detected)");
            ComputeBackend::CpuOptimized
        }
    }

    fn try_gpu() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let adapter_name = adapter.get_info().name.clone();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("xagent-compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        info!("[xagent] Compute backend: GpuAccelerated ({})", adapter_name);

        Some(ComputeBackend::GpuAccelerated {
            device,
            queue,
            adapter_name,
        })
    }

    /// Returns true if GPU is available.
    pub fn has_gpu(&self) -> bool {
        matches!(self, ComputeBackend::GpuAccelerated { .. })
    }

    /// Human-readable backend name for logging.
    pub fn name(&self) -> &str {
        match self {
            ComputeBackend::CpuOptimized => "CpuOptimized",
            ComputeBackend::GpuAccelerated { adapter_name, .. } => adapter_name,
        }
    }
}
