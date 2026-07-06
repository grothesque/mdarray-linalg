# mdarray-linalg: linear algebra operations for mdarray

[`mdarray`](https://crates.io/crates/mdarray) is a Rust crate
providing elegant abstractions for multidimensional arrays.
It takes advantage of modern Rust features such as const generics
and adapts core Rust concepts
(e.g., slices and iterators)
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
Whole-array operations including tensor contraction, matrix multiplication, decompositions and factorizations
are exposed as trait methods.
Crates such as `mdarray-linalg-blas` and `mdarray-linalg-faer`
provide library-specific backends,
i.e., Rust types that implement these traits.

This backend-based approach is more than just a unified interface to multiple libraries.
Backends are Rust values,
so they can carry configuration such as threading settings or library-specific context.

The operation traits are deliberately generic over the scalar type.
This allows each backend to choose the scalar types it supports:
BLAS/LAPACK backends naturally cover the classic BLAS scalar types,
while other backends may be generic over broader families of scalars.

User code can be generic over both the scalar type
and the backend that provides whole-array operations.
This allows user code to be written in any of the following ways:

- tied to a concrete combination of backend and scalar type;
- generic over the backend for a particular concrete scalar type;
- generic over the scalar type for a concrete backend;
- generic over both the backend and the scalar type.

In the most general case, trait bounds for the scalar type and the backend are expressed independently:
```rust
T: ...          // Require certain operations for the scalar type T.
B: Contract<T>  // Require a backend that can contract arrays of T.
```

This separation also leaves room for backend implementations
optimized for particular scalar types,
e.g., matrix multiplication for double-double scalars.
These can outperform implementations built from generic scalar operations of that type.

## Getting started

Add `mdarray` and `mdarray-linalg` as dependencies to a Rust project:

```bash
cargo add mdarray mdarray-linalg
```

This is enough to use the built-in `Naive` backend.
To use an optimized backend, add one of the backend crates as well.
The [mdarray-linalg documentation](https://docs.rs/mdarray-linalg/)
includes usage examples and a tabular overview of backend functionality.

## License
This crate is dual-licensed under the Apache-2.0 and MIT licenses to be compatible with the Rust project.
See the file LICENSE.md in this directory.
