//! A kernel built from the config flag alone picks its sense table and emits
//! the matching extra sensory channels.

use xagent_brain::buffers::BrainLayout;
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

const AGENTS: u32 = 4;
const FOOD: usize = 4;
const EXTENDED_SENSE_BLOCKS: usize = 3;
const WARM_TICKS: u32 = 4;

/// Pure decision: `Ok(true)` run, `Ok(false)` skip, `Err` fail closed.
fn gpu_guard_decision(available: bool, required: bool) -> Result<bool, &'static str> {
    if available {
        Ok(true)
    } else if required {
        Err("no GPU adapter available but XAGENT_REQUIRE_GPU is set")
    } else {
        Ok(false)
    }
}

fn gpu_or_skip() -> bool {
    let required = std::env::var_os("XAGENT_REQUIRE_GPU").is_some();
    match gpu_guard_decision(GpuKernel::is_available(), required) {
        Ok(true) => true,
        Ok(false) => {
            eprintln!("Skipping: no GPU/fallback adapter available");
            false
        }
        Err(message) => panic!("{message}"),
    }
}

#[test]
fn second_odour_kernel_builds_with_extended_table() {
    if !gpu_or_skip() {
        return;
    }
    let config = BrainConfig {
        second_odour_sense_enabled: true,
        ..BrainConfig::default()
    };
    let mut kernel = GpuKernel::new(AGENTS, FOOD, &config, &WorldConfig::default());
    let layout = kernel.brain_layout();
    assert_eq!(layout.sense_blocks().len(), EXTENDED_SENSE_BLOCKS);
    assert_eq!(
        layout.sensory_stride,
        BrainLayout::from_config(&config).sensory_stride
    );
    kernel.dispatch_ticks(0, WARM_TICKS);
}

#[test]
fn second_odour_kernel_off_matches_default_layout() {
    if !gpu_or_skip() {
        return;
    }
    let config = BrainConfig::default();
    let kernel = GpuKernel::new(AGENTS, FOOD, &config, &WorldConfig::default());
    assert_eq!(
        format!("{:?}", kernel.brain_layout()),
        format!("{:?}", BrainLayout::from_config(&BrainConfig::default()))
    );
}
