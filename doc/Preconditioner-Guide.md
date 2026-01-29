# Preconditioner Guide for Iterative Solvers

## Part 1: What is a Preconditioner?

### The Problem: Ill-Conditioned Systems

When solving `Ax = b` iteratively, the **condition number** κ(A) determines convergence speed:

```
κ(A) = λ_max / λ_min    (ratio of largest to smallest eigenvalue)
```

| Condition Number | Convergence | Iterations |
|------------------|-------------|------------|
| κ ≈ 1 | Excellent | Few |
| κ ≈ 10³ | Slow | Hundreds |
| κ ≈ 10⁶ | Very slow | Thousands+ |

**Example:** A poorly scaled system
```
A = [1     0.1  ]    κ(A) ≈ 1000
    [0.1   1000 ]    
```
Iterative solvers struggle because eigenvalues span a wide range.

### The Solution: Preconditioning

Find matrix **M ≈ A** that is cheap to "invert". Transform the system so the solver sees a better-conditioned matrix.

**Ideal case:** If M = A exactly, then M⁻¹A = I, and we solve in 1 iteration.

**Reality:** We use M ≈ A that balances:
- Closeness to A (better conditioning)
- Cheapness to apply M⁻¹ (less work per iteration)

### How Preconditioning is Applied

#### Left Preconditioning
```
Original:     A x = b
Transformed:  (M⁻¹A) x = M⁻¹b
```
Solve for x directly. Iterate with matrix M⁻¹A.

#### Right Preconditioning
```
Original:     A x = b
Substitute:   x = M⁻¹y
Transformed:  (A M⁻¹) y = b
Recover:      x = M⁻¹y
```
Solve for y, then compute x = M⁻¹y. Iterate with matrix AM⁻¹.

#### Split Preconditioning (Recommended for Symmetric Systems)
```
Original:     A x = b
Substitute:   x = M₂⁻¹y
Transformed:  (M₁⁻¹ A M₂⁻¹) y = M₁⁻¹b
Recover:      x = M₂⁻¹y
```
With M₁ = M₂ = √M, the preconditioned matrix M⁻¹ᐟ² A M⁻¹ᐟ² **remains symmetric**.

### Why Symmetry Matters

| Preconditioning | A symmetric → M⁻¹A symmetric? |
|-----------------|-------------------------------|
| Left (M⁻¹A) | ❌ No |
| Right (AM⁻¹) | ❌ No |
| Split (M⁻¹ᐟ²AM⁻¹ᐟ²) | ✅ Yes |

Symmetric solvers (CG, MINRES, QMR, BLQMR) work best when the preconditioned system stays symmetric.

---

## Part 2: Preconditioner Types

### 2.1 Diagonal (Jacobi) Preconditioner

**Construction:** M = diag(A)

```
A = [4  1  2]      M = [4  0  0]
    [1  5  1]          [0  5  0]
    [2  1  6]          [0  0  6]
```

| Aspect | Details |
|--------|---------|
| **Cost to build** | O(n) - just extract diagonal |
| **Cost to apply** | O(n) - element-wise division |
| **Storage** | O(n) - just the diagonal |
| **Parallelism** | Perfect - fully parallel |

**Pros:**
- Trivial to implement
- No fill-in
- Works for any matrix
- Perfectly parallel

**Cons:**
- Weak - ignores all off-diagonal structure
- Only fixes scaling issues
- Doesn't reduce iteration count much for well-scaled systems

**Best for:**
- Diagonally dominant matrices
- Quick baseline
- Massively parallel systems where communication is expensive

---

### 2.2 Block Diagonal Preconditioner

**Construction:** M = block-diag(A₁₁, A₂₂, ...)

```
A = [A₁₁  A₁₂]      M = [A₁₁   0 ]
    [A₂₁  A₂₂]          [ 0   A₂₂]
```

Each diagonal block is factored (LU or Cholesky) independently.

| Aspect | Details |
|--------|---------|
| **Cost to build** | O(∑ nᵢ³) for block sizes nᵢ |
| **Cost to apply** | O(∑ nᵢ²) |
| **Storage** | O(∑ nᵢ²) |
| **Parallelism** | Parallel across blocks |

**Pros:**
- Stronger than point Jacobi
- Natural for multi-physics problems
- Parallel across blocks

**Cons:**
- Ignores inter-block coupling
- Need to identify good block structure

**Best for:**
- Multi-physics (fluid-structure, coupled PDEs)
- Problems with natural block structure
- Domain decomposition

---

### 2.3 Incomplete LU (ILU) Preconditioner

**Construction:** A ≈ L̃Ũ where L̃, Ũ are sparse approximations.

```
True LU:   A = LU        (L, U may be dense due to fill-in)
ILU:       A ≈ L̃Ũ       (L̃, Ũ forced to stay sparse)
```

#### ILU Variants

| Variant | Fill-in Rule | Storage |
|---------|--------------|---------|
| ILU(0) | Same sparsity as A | O(nnz) |
| ILU(k) | Allow k levels of fill | O(nnz × 2ᵏ) |
| ILUT(τ) | Drop entries < τ | Adaptive |

**ILU(0) Example:**
```
During factorization, if A(i,j) = 0, force L̃(i,j) = Ũ(i,j) = 0
even if the true LU would have fill-in there.
```

| Aspect | Details |
|--------|---------|
| **Cost to build** | O(nnz) for ILU(0), more for ILU(k) |
| **Cost to apply** | O(nnz) - triangular solves |
| **Storage** | O(nnz) for ILU(0) |
| **Parallelism** | Poor - inherently sequential |

**Pros:**
- Much stronger than Jacobi
- Captures matrix structure
- Well-studied, widely available

**Cons:**
- Sequential triangular solves
- Can fail for indefinite matrices
- Fill-in can grow for ILU(k), ILUT

**Best for:**
- General sparse systems
- Moderate-size problems
- When Jacobi is too weak

---

### 2.4 Incomplete Cholesky (IC) Preconditioner

**Construction:** A ≈ L̃L̃ᵀ for symmetric positive definite (SPD) A.

Same idea as ILU but exploits symmetry - only one factor needed.

| Aspect | Details |
|--------|---------|
| **Cost to build** | O(nnz) |
| **Cost to apply** | O(nnz) |
| **Storage** | O(nnz/2) - only L̃ |
| **Parallelism** | Poor |

**Pros:**
- Half the storage of ILU
- Preserves symmetry
- Natural for SPD systems

**Cons:**
- **Only works for SPD matrices**
- Can fail with negative pivot (breakdown)
- Not applicable to indefinite or complex symmetric

**Best for:**
- SPD systems (Laplacian, diffusion, elasticity)
- When memory is tight

---

### 2.5 SSOR (Symmetric Successive Over-Relaxation)

**Construction:** Split A = L + D + U (lower + diagonal + upper)

```
M = (D + ωL) D⁻¹ (D + ωL)ᵀ
```

where ω ∈ (0, 2) is a relaxation parameter.

| Aspect | Details |
|--------|---------|
| **Cost to build** | O(1) - no factorization! |
| **Cost to apply** | O(nnz) - forward/backward sweeps |
| **Storage** | O(1) - uses A directly |
| **Parallelism** | Poor - sequential sweeps |

**Pros:**
- No factorization needed
- No additional storage
- Preserves symmetry
- No breakdown possible

**Cons:**
- Requires tuning ω (optimal ω is problem-dependent)
- Generally weaker than ILU
- Sequential

**Best for:**
- Memory-constrained situations
- When ILU fails (indefinite systems)
- Simple implementation needed

---

### 2.6 Algebraic Multigrid (AMG)

**Construction:** Automatic hierarchy of coarser problems.

```
Level 0: Fine grid      (n unknowns)     ← Smooth
            ↓ Restrict
Level 1: Coarse grid    (n/2 unknowns)   ← Smooth
            ↓ Restrict  
Level 2: Coarser grid   (n/4 unknowns)   ← Smooth
            ↓
Level k: Coarsest       (small)          ← Solve directly
            ↑
      Interpolate back up through levels
```

| Aspect | Details |
|--------|---------|
| **Cost to build** | O(nnz) but with large constant |
| **Cost to apply** | O(nnz) - optimal! |
| **Storage** | O(nnz) across all levels |
| **Parallelism** | Good with careful implementation |

**Pros:**
- Near-optimal O(n) complexity
- Scalable to very large problems
- Black-box (no geometry needed)

**Cons:**
- Complex to implement
- Significant setup cost
- Large memory footprint
- Can struggle with non-elliptic problems

**Best for:**
- Large elliptic PDEs (10⁶+ unknowns)
- Laplacian, diffusion, elasticity
- When scalability is critical

---

### 2.7 Sparse Approximate Inverse (SPAI)

**Construction:** Compute M⁻¹ directly (sparse approximation to A⁻¹).

```
Minimize ‖AM⁻¹ - I‖_F  subject to sparsity constraints on M⁻¹
```

| Aspect | Details |
|--------|---------|
| **Cost to build** | O(nnz²) or more |
| **Cost to apply** | O(nnz) - just matrix-vector multiply! |
| **Storage** | O(nnz) |
| **Parallelism** | **Excellent** - just SpMV |

**Pros:**
- Applying M⁻¹ is just a matrix-vector multiply (parallel!)
- No triangular solves
- Good for GPU/parallel architectures

**Cons:**
- Expensive to construct
- Hard to get good sparsity pattern
- Often weaker than ILU

**Best for:**
- GPU computing
- Massively parallel systems
- When triangular solves are bottleneck

---

### Comparison Summary

| Preconditioner | Strength | Build Cost | Apply Cost | Parallel | Memory |
|----------------|----------|------------|------------|----------|--------|
| Jacobi | Weak | O(n) | O(n) | ★★★★★ | O(n) |
| Block Jacobi | Moderate | O(∑nᵢ³) | O(∑nᵢ²) | ★★★★☆ | O(∑nᵢ²) |
| ILU(0) | Good | O(nnz) | O(nnz) | ★☆☆☆☆ | O(nnz) |
| ILUT | Very Good | O(nnz+) | O(nnz+) | ★☆☆☆☆ | Variable |
| IC | Good | O(nnz) | O(nnz) | ★☆☆☆☆ | O(nnz/2) |
| SSOR | Moderate | O(1) | O(nnz) | ★☆☆☆☆ | O(1) |
| AMG | Excellent | O(nnz) | O(nnz) | ★★★☆☆ | O(nnz) |
| SPAI | Moderate | O(nnz²) | O(nnz) | ★★★★★ | O(nnz) |

---

## Part 3: Best Practices for BLQMR and Symmetric Systems

### 3.1 Why Symmetric Systems are Special

BLQMR solves **complex symmetric** systems: A = Aᵀ (transpose, not conjugate transpose).

These arise in:
- Helmholtz equation (acoustics, electromagnetics)
- Wave propagation
- Frequency-domain problems

**Key challenges:**
- Not Hermitian → standard CG doesn't apply
- Often indefinite → ILU can break down
- Complex arithmetic → some preconditioners don't extend naturally

### 3.2 Split Preconditioning is Essential

For symmetric A, use **split preconditioning** with M₁ = M₂:

```matlab
% For Jacobi:
d = diag(A);
sqrt_d = sqrt(d);
M1 = spdiags(sqrt_d, 0, n, n);
M2 = M1;

[x, flag] = blqmr(A, b, tol, maxit, M1, M2);
```

This transforms:
```
A x = b  →  (D⁻¹ᐟ² A D⁻¹ᐟ²) y = D⁻¹ᐟ² b
```

The preconditioned matrix **stays symmetric**.

### 3.3 Recommended Preconditioners for BLQMR

#### Tier 1: Safe Choices (Always Work)

**Split Jacobi**
```matlab
d = full(diag(A));
d(abs(d) < 1e-14) = 1;  % Handle zeros
sqrt_d = sqrt(d);
M = spdiags(sqrt_d, 0, n, n);
[x, flag] = blqmr(A, b, tol, maxit, M, M);
```
- ✅ Simple, robust, parallel
- ❌ Weak - only fixes scaling

**SSOR**
```matlab
% No explicit construction - apply as sweeps
omega = 1.0;  % Or tune
% Implement as forward/backward Gauss-Seidel sweeps
```
- ✅ No factorization, no breakdown
- ❌ Sequential, needs parameter tuning

#### Tier 2: Stronger but Riskier

**Split ILU (if available)**

For complex symmetric, construct ILU and use symmetrically:
```matlab
[L, U] = ilu(A, struct('type', 'ilutp', 'droptol', 1e-4));
% Use as left/right pair (not truly split, but often works)
```
- ✅ Much stronger than Jacobi
- ❌ Can fail for indefinite systems
- ❌ Not truly symmetric split

**Shifted System**

For Helmholtz (A = K - k²M + iσM), precondition with the shifted/damped system:
```matlab
A_shift = K - k²M + i*beta*M;  % beta > sigma (more damping)
[L, U] = lu(A_shift);           % Or ILU
% Use L, U as preconditioner
```
- ✅ Physics-based, very effective for waves
- ❌ Requires problem-specific knowledge

#### Tier 3: Advanced (Large Problems)

**AMG (if applicable)**

Some AMG implementations handle complex symmetric:
```matlab
% Using external library (e.g., AGMG, HSL_MI20)
amg_setup = amg_setup(A);
% Use amg_apply as preconditioner
```
- ✅ Scalable, near-optimal
- ❌ Not all AMG codes handle complex symmetric
- ❌ Setup cost

### 3.4 Decision Flowchart

```
Start
  │
  ▼
Is the system small (n < 10,000)?
  │
  ├─ Yes → Try direct solver first (A\b)
  │        If too slow, use ILU + BLQMR
  │
  └─ No
      │
      ▼
    Is the diagonal well-scaled (max/min < 100)?
      │
      ├─ Yes → Preconditioning may not help much
      │        Try BLQMR without preconditioner first
      │
      └─ No
          │
          ▼
        Is the system very large (n > 10⁶)?
          │
          ├─ Yes → Consider AMG or domain decomposition
          │        Split Jacobi as fallback
          │
          └─ No
              │
              ▼
            Is ILU stable for your matrix?
              │
              ├─ Yes → Use split ILU (strongest)
              │
              └─ No (indefinite, ILU breaks)
                  │
                  ▼
                Use Split Jacobi or SSOR
                Consider shifted preconditioner for Helmholtz
```

### 3.5 Practical Tips

#### 1. Always Test Without Preconditioner First
```matlab
[x, flag, ~, iter_no_prec] = blqmr(A, b, tol, maxit);
fprintf('No precond: %d iterations\n', iter_no_prec);
```
If this converges quickly, you may not need preconditioning.

#### 2. Check Diagonal Scaling
```matlab
d = full(diag(A));
fprintf('Diagonal range: [%.2e, %.2e], ratio = %.1f\n', ...
    min(abs(d)), max(abs(d)), max(abs(d))/min(abs(d)));
```
If ratio > 100, Jacobi preconditioning should help.

#### 3. Handle Zero/Small Diagonal Entries
```matlab
d = full(diag(A));
small = abs(d) < max(abs(d)) * 1e-14;
d(small) = 1;  % Replace with 1 to avoid division by zero
```

#### 4. Complex Square Roots
For complex diagonal entries, `sqrt(d)` gives complex results. This is correct and necessary for complex symmetric systems.
```matlab
d = full(diag(A));  % Complex
sqrt_d = sqrt(d);   % Also complex - this is fine
```

#### 5. Monitor Convergence
```matlab
[x, flag, relres, iter, resvec] = blqmr(A, b, tol, maxit, M1, M2);
semilogy(resvec);  % Plot convergence history
```
If residual stagnates, try a stronger preconditioner.

#### 6. Verify Solution
```matlab
true_residual = norm(A*x - b) / norm(b);
fprintf('Relative residual: %.2e\n', true_residual);
```
The quasi-residual from BLQMR may differ from the true residual.

### 3.6 Quick Reference

| Situation | Recommended Preconditioner |
|-----------|---------------------------|
| First attempt | None (test baseline) |
| Poorly scaled diagonal | Split Jacobi |
| General sparse, moderate size | Split ILU if stable |
| ILU fails (indefinite) | Split Jacobi or SSOR |
| Helmholtz/wave problems | Shifted preconditioner |
| Very large (>10⁶) | AMG or domain decomposition |
| GPU/parallel priority | Split Jacobi or SPAI |

### 3.7 Code Template

```matlab
function x = solve_with_blqmr(A, b, tol, maxit)
    n = size(A, 1);
    
    % Check diagonal scaling
    d = full(diag(A));
    d(abs(d) < max(abs(d)) * 1e-14) = 1;
    diag_ratio = max(abs(d)) / min(abs(d));
    
    if diag_ratio > 100
        % Use split Jacobi preconditioning
        fprintf('Using split Jacobi (diagonal ratio = %.1f)\n', diag_ratio);
        sqrt_d = sqrt(d);
        M = spdiags(sqrt_d, 0, n, n);
        [x, flag, relres, iter] = blqmr(A, b, tol, maxit, M, M);
    else
        % No preconditioning needed
        fprintf('No preconditioning (diagonal ratio = %.1f)\n', diag_ratio);
        [x, flag, relres, iter] = blqmr(A, b, tol, maxit);
    end
    
    % Report results
    true_res = norm(A*x - b) / norm(b);
    fprintf('flag=%d, iter=%d, relres=%.2e, true_res=%.2e\n', ...
        flag, iter, relres, true_res);
end
```

---

## Summary

1. **Preconditioning** transforms `Ax = b` into an equivalent system that converges faster

2. **For symmetric systems**, use **split preconditioning** (M₁ = M₂) to preserve symmetry

3. **Start simple** (no preconditioner or Jacobi), move to stronger methods if needed

4. **For BLQMR specifically:**
   - Split Jacobi is the safe default
   - Check diagonal scaling to decide if preconditioning helps
   - Use `blqmr(A, b, tol, maxit, M, M)` with M = √diag(A)
