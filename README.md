# mdarray-linalg: Linear algebra operations for mdarray

[`mdarray`](https://crates.io/crates/mdarray) is a Rust crate
providing elegant abstractions for multidimensional arrays.
It takes advantage of modern Rust features such as const generics
and adapts core Rust concepts
(e.g. slices and iterators)
to multidimensional arrays.
Its `Array` type can represent
tiny zero-overhead stack-allocated arrays
just as well as heap-allocated arrays with fully dynamic shapes,
and even hybrid arrays whose shapes are partly known at compile time and partly dynamic.
Since `mdarray` arrays are not limited to numeric elements,
the core library leaves arithmetic (beyond elementwise operations)
to add-on libraries.

This crate, `mdarray-linalg`,
defines traits for linear algebra operations on `mdarray` arrays.
Whole-array operations including
tensor contraction, matrix multiplication, decompositions, and factorizations
are exposed as trait methods.
Crates such as `mdarray-linalg-blas` and `mdarray-linalg-faer`
provide backend types that implement these traits.
Since backends are Rust values
they can carry configuration such as threading settings or library-specific context.

The traits are deliberately generic over the scalar type.
This allows each backend to choose the scalar types it supports:
BLAS/LAPACK backends naturally cover the classic BLAS scalar types,
while other backends may be generic over broader families of scalars.

User code can be generic over both the scalar type and the backend type.
The possibilities for user code thus range from
being tied to a concrete combination scalar and backend type,
to being generic over both:
```rust
T: ...          // Require certain operations for the scalar type T.
B: Contract<T>  // Require a backend that can contract arrays of T.
```

Similarly, a backend can be either generic over some family of scalars,
or tied to one or more concrete scalar types.
By switching backends,
user code that is generic over a broad range of scalars can still take advantage
of whole-array operations that have been optimized for a particular scalar type.
For example, when working with double-double scalars,
a custom matrix multiplication routine
can outperform a generic one merely instantiated for double-double scalars.

## Getting started

Add `mdarray` and `mdarray-linalg` as dependencies to a Rust project:

```bash
cargo add mdarray mdarray-linalg
```

This is enough to use the built-in `Naive` backend.
To use an optimized backend, add one of the backend crates as well.
Note that some backends have non-Rust dependencies that need to be satisfied:
see the backend documentation.

## Local development

This repository is a workspace that contains the main `mdarray-linalg` crate
and multiple `mdarray-linalg-*` backend crates.
Only backends without non-Rust dependencies are listed as `default-members` in `Cargo.toml`.
This means that a simple `cargo test` should run in a fresh git clone.

The tests for the BLAS and LAPACK backends assume that OpenBLAS is available.
On Debian/Ubuntu,
`sudo apt install pkgconf libopenblas-dev` should install all that is necessary.
Install equivalent packages on other systems.

The tests for the TBLIS backend assume that TBLIS is available.
TBLIS can be installed from source following the
[upstream build guide](https://github.com/MatthewsResearchGroup/tblis/wiki/Building).

Developers who wish to use a different setup can carry a small local patch
that reconfigures the workspace according to their preferences.

## License

This crate is dual-licensed under the Apache-2.0 and MIT licenses to be compatible with the Rust project.
See the file LICENSE.md in this directory.
