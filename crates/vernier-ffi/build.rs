//! Embeds libpython's directory as an rpath so the test binaries (which
//! link libpython) start without `LD_LIBRARY_PATH`. pyo3 emits nothing
//! for `extension-module` builds, so wheels are unchanged.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    pyo3_build_config::add_libpython_rpath_link_args();
}
