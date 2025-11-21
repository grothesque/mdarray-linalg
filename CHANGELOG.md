# Changelog of mdarray-linalg

The mdarray-linalg project adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).
This file documents all notable user-visible changes to the crates present in this workspace.
The format follows [keep-a-changelog](https://keepachangelog.com/en/1.1.0/).

## unreleased

### Changed
- **Matrix-vector interface overhaul:**
  Rework `matvec` API  for greater flexibility and consistency.
  Examples:
  - Rank-1 update: *A ← A + α x⊗y<sup>T</sup>*
  ```rust
  bd.outer(&x, &y).scale(alpha).add_to(&mut A);
  ```
  - Matrix-vector product: *y ← Ax + y*
  ```rust
  bd.matvec(&A, &x).add_to_vec(&mut y);
  ```

### Added
- **`argmax_abs` function:**
  The new function `argmax_abs` finds the index of the element with the largest absolute value in tensors of arbitrary dimension.
  It provides a wrapper around the BLAS `iamax` function.
  A `Naive` backend implementation is provided for environments without BLAS.

## [0.1.2](https://github.com/grothesque/mdarray-linalg/releases/tag/v0.1.2) - 2025-11-05
### Added
- `argmax_abs` with backends BLAS and Naive.

## [0.1.1](https://github.com/grothesque/mdarray-linalg/releases/tag/v0.1.1) - 2025-10-24
Initial version
