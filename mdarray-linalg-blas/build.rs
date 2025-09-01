use std::process::Command;

fn main() {
    let output = Command::new("python3")
        .arg("scripts/gen_matmut.py")
        .output()
        .expect("Failed to execute Python script");

    println!("{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("{}", String::from_utf8_lossy(&output.stderr));

    if !output.status.success() {
        panic!("Python script failed!");
    }

    println!("cargo:rerun-if-changed=script.py");
}
