//! The observatory redraw sources must not name GPU, readback or database APIs.

fn assert_clean(name: &str, source: &str, needles: &[&str]) {
    for needle in needles {
        assert!(!source.contains(needle), "{name} names {needle}");
    }
}

#[test]
fn observatory_sources_name_no_gpu_readback_or_database_api() {
    let view_needles = [
        concat!("Gpu", "Kernel"),
        concat!("wg", "pu"),
        concat!("rusql", "ite"),
        concat!("Conn", "ection"),
        concat!("store_", "read::"),
        concat!("read_full_state_", "blocking"),
    ];
    let views = [
        ("tables.rs", include_str!("../src/observatory/tables.rs")),
        ("plots.rs", include_str!("../src/observatory/plots.rs")),
        ("history.rs", include_str!("../src/observatory/history.rs")),
    ];
    for (name, source) in views {
        assert_clean(name, source, &view_needles);
    }

    let ui_needles = [
        concat!("store_read::", "load_run"),
        concat!("store_read::", "detect_supervisor_run"),
        concat!("store_read::", "open_read_only"),
        concat!("Gpu", "Kernel"),
        concat!("read_full_state_", "blocking"),
    ];
    assert_clean("ui.rs", include_str!("../src/ui.rs"), &ui_needles);
}
