# mdarray-linalg: linear algebra bindings for Rust mdarray

Efficient, flexible, and idiomatic linear algebra bindings (BLAS, LAPACK, etc.)
to the Rust [mdarray](https://github.com/fre-hu/mdarray) crate.

## Usage
These crates are not yet released on crates.io. To depend on the main crate
and one of the backends (for example `mdarray-linalg-blas`) add the following
to your `Cargo.toml`:
```
[dependencies]
mdarray-linalg = { git = "https://github.com/grothesque/mdarray-linalg" }
mdarray-linalg-blas = { git = "https://github.com/grothesque/mdarray-linalg" }
```

See the tests for a “tutorial”.

## License
Dual-licensed (Apache and MIT) to be compatible with the Rust project.
See the file LICENSE.md in this directory.
