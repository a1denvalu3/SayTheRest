fn main() {
    println!("cargo:rerun-if-changed=vendor/sonic/sonic.c");
    println!("cargo:rerun-if-changed=vendor/sonic/sonic.h");
    cc::Build::new()
        .file("vendor/sonic/sonic.c")
        .include("vendor/sonic")
        .opt_level(2)
        .flag_if_supported("-Wno-unused-parameter")
        .compile("sayit_sonic");
}
