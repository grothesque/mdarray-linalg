# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-12

### Changed
- **Matrix-vector interface overhaul:**  
  The `matvec` API has been reworked for greater flexibility and consistency.  
  - Rank-1 updates can now be performed via  
    ```rust
	  bd.outer(&x, &y).eval();
	  bd.outer(&x, &y).overwrite(&mut A);
	  bd.outer(&x, &y).add_to(&A);
	  bd.outer(&x, &y).scale(alpha).add_to_overwrite(&mut A); // (A := A + alpha x⊗y^T)
    ```  
    to compute a rank-1 update on matrix `a`.  
  - Matrix-vector products are now expressed as  
    ```rust
	bd.matvec(&A, &x).eval();
	bd.matvec(&A, &x).overwrite(&mut y);
    ```  
    which computes `Ax + y`.

### Added
- **`argmax_abs` function:**  
  Introduced `argmax_abs` to find the index of the element with the largest absolute value in tensors of arbitrary dimension.  
  - Provides a wrapper around the BLAS `iamax` function.  
  - Includes a `Naive` backend implementation for environments without BLAS support.

---

[0.2.0]: https://github.com/grothesque/mdarray-linalg/releases/tag/v0.2.
