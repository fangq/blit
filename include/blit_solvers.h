/**
 * @file blit_solvers.h
 * @brief C/C++ interface header for the Blit sparse linear solver library
 *
 * Blit - An open-source library for block iterative sparse linear solvers
 *
 * @copyright Copyright 2011,2020 Qianqian Fang <q.fang at neu.edu>
 * @license BSD or LGPL or GPL, see LICENSE_*.txt for more details
 *
 * @details This header provides C and C++ bindings to call the Fortran-based
 *          Block Quasi-Minimal Residual (BLQMR) solver and ILU preconditioner.
 *          The structs defined here must exactly match the Fortran type definitions
 *          with bind(c) attribute for proper interoperability.
 *
 * @see http://blit.sourceforge.net
 *
 * @author Qianqian Fang, PhD
 *         Dept. of Bioengineering, Northeastern University
 *         360 Huntington Ave, ISEC 206, Boston, MA 02115, USA
 */

#ifndef _BLIT_SOLVERS_C_H
#define _BLIT_SOLVERS_C_H

#include <stdio.h>
#include <string.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*===========================================================================*/
/** @name Constants and Enumerations
 *  @{ */
/*===========================================================================*/

/** @brief Null pointer constant for Blit objects */
#define BLIT_NULL       0

/** @brief Type identifier for real (double precision) matrices */
#define MTYPEID_REAL    1

/** @brief Type identifier for complex (double precision) matrices */
#define MTYPEID_COMPLEX 2

/**
 * @brief Error codes returned by Blit operations
 */
enum BlitError {
    beTypeMismatch = -999,  /**< Type mismatch between expected and actual data types */
    beNoLHS        = -998,  /**< Left-hand side matrix (A) not provided or invalid */
    beNoRHS        = -997   /**< Right-hand side vector/matrix (b) not provided or invalid */
};

/** @} */

/*===========================================================================*/
/** @name Data Structures
 *  @{ */
/*===========================================================================*/

/**
 * @brief Incomplete LU (ILU) preconditioner structure
 *
 * @details This structure holds the ILU factorization data computed by UMFPACK.
 *          It stores the symbolic and numeric factorization handles along with
 *          control parameters. Must match the Fortran ILUPcond type exactly.
 *
 * @note The numeric and symbolic fields are opaque pointers to UMFPACK's
 *       internal data structures. They must be void* to correctly represent
 *       Fortran's c_ptr on 64-bit systems.
 */
typedef struct Blit_ILUPcond {
    int n;              /**< Matrix dimension (n x n system) */
    int nz;             /**< Number of non-zero elements in the sparse matrix */
    int status;         /**< Status code from last operation (-1 = uninitialized) */
    int iscomplex;      /**< Flag: 0 = real matrix, 1 = complex matrix */
    void *numeric;      /**< UMFPACK numeric factorization handle (c_ptr) */
    void *symbolic;     /**< UMFPACK symbolic factorization handle (c_ptr) */
    double control[20]; /**< UMFPACK control parameters array */
} ILUPcond;

/**
 * @brief Block Quasi-Minimal Residual (BLQMR) solver structure
 *
 * @details This structure contains all parameters and state information for
 *          the BLQMR iterative solver. It supports both real and complex
 *          sparse linear systems with multiple right-hand sides.
 *
 * @note Field order must exactly match the Fortran BLQMRSolver type with
 *       bind(c) attribute. The ilu field is embedded (not a pointer).
 */
typedef struct Blit_BLQMRSolver {
    /* Integer parameters - must maintain this exact order */
    int n;              /**< Matrix dimension (n x n system) */
    int nrhs;           /**< Number of right-hand side vectors */
    int maxit;          /**< Maximum number of iterations allowed */
    int state;          /**< Solver state: -1=destroyed, 0=created, 1=prepared */
    int flag;           /**< Convergence flag:
                             -1 = not started,
                              0 = converged,
                              1 = max iterations reached,
                              2 = preconditioner failure (NaN detected),
                              3 = stagnation (no progress) */
    int iter;           /**< Actual number of iterations performed */
    int isquasires;     /**< Residual computation mode:
                             0 = true residual (more accurate, slower),
                             1 = quasi-residual (faster, default) */
    int debug;          /**< Debug output level (bitmask):
                             0 = silent,
                             1 = print residual each iteration */
    int pcond_type;     /**< Preconditioner type:
                             0 = none,
                             1 = ILU-left (default),
                             2 = ILU-split,
                             3 = Jacobi-split */

    /* Floating-point parameters */
    double qtol;        /**< Convergence tolerance for relative residual (default: 1e-6) */
    double droptol;     /**< Drop tolerance for ILU factorization (default: 1e-3) */
    double res;         /**< Absolute residual norm at termination */
    double relres;      /**< Relative residual norm at termination (res/res0) */

    /* Embedded preconditioner structure */
    ILUPcond ilu;       /**< ILU preconditioner data (embedded, not a pointer) */
} BLQMRSolver;

/**
 * @brief Fortran 90 compatible complex number structure
 *
 * @details Represents a double-precision complex number with the same
 *          memory layout as Fortran's complex(kind=8) type.
 */
typedef struct Blit_F90Complex {
    double x;           /**< Real part */
    double y;           /**< Imaginary part */
} F90Complex;

/** @} */

/*===========================================================================*/
/** @name Fortran Symbol Name Mappings
 *  @brief Maps C function names to Fortran module procedure mangled names
 *  @{ */
/*===========================================================================*/

/** @brief Create real BLQMR solver */
#define DBLQMRCreate    __blit_blqmr_real_MOD_blqmroncreate
/** @brief Create complex BLQMR solver */
#define ZBLQMRCreate    __blit_blqmr_complex_MOD_blqmroncreate
/** @brief Prepare real BLQMR solver (build preconditioner) */
#define DBLQMRPrep      __blit_blqmr_real_MOD_blqmrprep
/** @brief Prepare complex BLQMR solver (build preconditioner) */
#define ZBLQMRPrep      __blit_blqmr_complex_MOD_blqmrprep
/** @brief Solve real system */
#define DBLQMRSolve     __blit_blqmr_real_MOD_blqmrsolve
/** @brief Solve complex system */
#define ZBLQMRSolve     __blit_blqmr_complex_MOD_blqmrsolve
/** @brief Destroy real BLQMR solver */
#define DBLQMRDestroy   __blit_blqmr_real_MOD_blqmrondestroy
/** @brief Destroy complex BLQMR solver */
#define ZBLQMRDestroy   __blit_blqmr_complex_MOD_blqmrondestroy
/** @brief Print real BLQMR solver state */
#define DBLQMRPrint     __blit_blqmr_real_MOD_blqmrprint
/** @brief Print complex BLQMR solver state */
#define ZBLQMRPrint     __blit_blqmr_complex_MOD_blqmrprint

/** @brief Create ILU preconditioner */
#define ILUPcondCreate  __blit_ilupcond_MOD_ilupcondcreate
/** @brief Prepare ILU preconditioner (factorize) */
#define ILUPcondPrep    __blit_ilupcond_MOD_ilupcondprep
/** @brief Solve with ILU preconditioner */
#define ILUPcondSolve   __blit_ilupcond_MOD_ilupcondsolve
/** @brief Destroy ILU preconditioner */
#define ILUPcondDestroy __blit_ilupcond_MOD_ilupconddestroy

/** @} */

/*===========================================================================*/
/** @name BLQMR Solver Functions (Real)
 *  @brief Functions for solving real sparse linear systems A*x = b
 *  @{ */
/*===========================================================================*/

/**
 * @brief Initialize a real BLQMR solver object
 *
 * @param[in,out] qmr  Pointer to BLQMRSolver structure to initialize
 * @param[in]     n    Pointer to matrix dimension (n x n system)
 *
 * @details Initializes all solver parameters to default values:
 *          - qtol = 1e-6, droptol = 1e-3, maxit = n
 *          - pcond_type = 1 (ILU-left), isquasires = 1
 */
extern void DBLQMRCreate(BLQMRSolver *qmr, int *n);

/**
 * @brief Build the preconditioner for a real sparse matrix
 *
 * @param[in,out] qmr  Pointer to initialized BLQMRSolver
 * @param[in,out] Ap   Column pointers array (size n+1, 1-based indexing)
 * @param[in,out] Ai   Row indices array (size nnz, 1-based indexing)
 * @param[in,out] Ax   Non-zero values array (size nnz)
 * @param[in]     nz   Pointer to number of non-zeros
 *
 * @details The matrix is in Compressed Sparse Column (CSC) format.
 *          After this call, qmr->state is set to 1 (prepared).
 */
extern void DBLQMRPrep(BLQMRSolver *qmr, int *Ap, int *Ai, double *Ax, int *nz);

/**
 * @brief Solve a real sparse linear system A*x = b
 *
 * @param[in,out] qmr   Pointer to prepared BLQMRSolver
 * @param[in]     Ap    Column pointers array (size n+1)
 * @param[in]     Ai    Row indices array (size nnz)
 * @param[in]     Ax    Non-zero values array (size nnz)
 * @param[in]     nz    Pointer to number of non-zeros
 * @param[in,out] x     Solution vectors (size n x nrhs), initial guess on input
 * @param[in]     b     Right-hand side vectors (size n x nrhs)
 * @param[in]     nrhs  Pointer to number of right-hand sides
 *
 * @details On return, qmr->flag indicates convergence status,
 *          qmr->iter contains iteration count, and qmr->relres
 *          contains the relative residual norm.
 */
extern void DBLQMRSolve(BLQMRSolver *qmr, int *Ap, int *Ai, double *Ax,
                        int *nz, double *x, double *b, int *nrhs);

/**
 * @brief Destroy a real BLQMR solver and free resources
 *
 * @param[in,out] qmr  Pointer to BLQMRSolver to destroy
 *
 * @details Frees the ILU preconditioner and any allocated memory.
 *          After this call, qmr->state is set to -1.
 */
extern void DBLQMRDestroy(BLQMRSolver *qmr);

/**
 * @brief Print the state of a real BLQMR solver in JSON format
 *
 * @param[in] qmr  Pointer to BLQMRSolver to print
 */
extern void DBLQMRPrint(BLQMRSolver *qmr);

/** @} */

/*===========================================================================*/
/** @name BLQMR Solver Functions (Complex)
 *  @brief Functions for solving complex sparse linear systems A*x = b
 *  @{ */
/*===========================================================================*/

/**
 * @brief Initialize a complex BLQMR solver object
 *
 * @param[in,out] qmr  Pointer to BLQMRSolver structure to initialize
 * @param[in]     n    Pointer to matrix dimension (n x n system)
 *
 * @see DBLQMRCreate for parameter defaults
 */
extern void ZBLQMRCreate(BLQMRSolver *qmr, int *n);

/**
 * @brief Build the preconditioner for a complex sparse matrix
 *
 * @param[in,out] qmr  Pointer to initialized BLQMRSolver
 * @param[in,out] Ap   Column pointers array (size n+1, 1-based indexing)
 * @param[in,out] Ai   Row indices array (size nnz, 1-based indexing)
 * @param[in,out] Ax   Complex non-zero values array (size nnz)
 * @param[in]     nz   Pointer to number of non-zeros
 */
extern void ZBLQMRPrep(BLQMRSolver *qmr, int *Ap, int *Ai, F90Complex *Ax, int *nz);

/**
 * @brief Solve a complex sparse linear system A*x = b
 *
 * @param[in,out] qmr   Pointer to prepared BLQMRSolver
 * @param[in]     Ap    Column pointers array (size n+1)
 * @param[in]     Ai    Row indices array (size nnz)
 * @param[in]     Ax    Complex non-zero values array (size nnz)
 * @param[in]     nz    Pointer to number of non-zeros
 * @param[in,out] x     Complex solution vectors (size n x nrhs)
 * @param[in]     b     Complex right-hand side vectors (size n x nrhs)
 * @param[in]     nrhs  Pointer to number of right-hand sides
 */
extern void ZBLQMRSolve(BLQMRSolver *qmr, int *Ap, int *Ai, F90Complex *Ax,
                        int *nz, F90Complex *x, F90Complex *b, int *nrhs);

/**
 * @brief Destroy a complex BLQMR solver and free resources
 *
 * @param[in,out] qmr  Pointer to BLQMRSolver to destroy
 */
extern void ZBLQMRDestroy(BLQMRSolver *qmr);

/**
 * @brief Print the state of a complex BLQMR solver in JSON format
 *
 * @param[in] qmr  Pointer to BLQMRSolver to print
 */
extern void ZBLQMRPrint(BLQMRSolver *qmr);

/** @} */

/*===========================================================================*/
/** @name ILU Preconditioner Functions
 *  @brief Standalone ILU preconditioner interface (using UMFPACK)
 *  @{ */
/*===========================================================================*/

/**
 * @brief Initialize an ILU preconditioner object
 *
 * @param[in,out] ilu  Pointer to ILUPcond structure to initialize
 * @param[in]     n    Pointer to matrix dimension
 * @param[in]     nz   Pointer to number of non-zeros
 */
extern void ILUPcondCreate(ILUPcond *ilu, int *n, int *nz);

/**
 * @brief Compute ILU factorization of a sparse matrix
 *
 * @param[in,out] ilu      Pointer to initialized ILUPcond
 * @param[in]     Ap       Column pointers array
 * @param[in]     Ai       Row indices array
 * @param[in]     Ax       Real part of non-zero values
 * @param[in]     droptol  Pointer to drop tolerance for incomplete factorization
 * @param[in]     Az       Imaginary part of non-zero values (NULL for real matrices)
 */
extern void ILUPcondPrep(ILUPcond *ilu, int *Ap, int *Ai, double *Ax,
                         double *droptol, double *Az);

/**
 * @brief Solve a system using the ILU preconditioner: (LU)*x = b
 *
 * @param[in,out] ilu   Pointer to factorized ILUPcond
 * @param[in]     Ap    Column pointers array
 * @param[in]     Ai    Row indices array
 * @param[in]     Ax    Real part of matrix values
 * @param[in]     rows  Pointer to number of rows
 * @param[in]     cols  Pointer to number of columns (right-hand sides)
 * @param[out]    x     Real part of solution
 * @param[in]     b     Real part of right-hand side
 * @param[in]     Az    Imaginary part of matrix values (NULL for real)
 * @param[out]    xz    Imaginary part of solution (NULL for real)
 * @param[in]     bz    Imaginary part of right-hand side (NULL for real)
 */
extern void ILUPcondSolve(ILUPcond *ilu, int *Ap, int *Ai, double *Ax,
                          int *rows, int *cols, double *x, double *b,
                          double *Az, double *xz, double *bz);

/**
 * @brief Destroy an ILU preconditioner and free UMFPACK resources
 *
 * @param[in,out] ilu  Pointer to ILUPcond to destroy
 */
extern void ILUPcondDestroy(ILUPcond *ilu);

/** @} */

#endif /* _BLIT_SOLVERS_C_H */

#ifdef  __cplusplus
}
#endif

/*===========================================================================*/
/** @name C++ Template Classes
 *  @brief Object-oriented C++ wrappers for Blit solvers
 *  @{ */
/*===========================================================================*/

#ifdef  __cplusplus

/**
 * @brief C++ wrapper class for the ILU preconditioner
 *
 * @tparam T  Data type (double for real, F90Complex for complex)
 *
 * @details Provides RAII-style resource management for the ILU preconditioner.
 *
 * @code
 * BlitILU<double> ilu(n, nz);
 * ilu.Run(&Ap, &Ai, &Ax, 0.001);
 * ilu.Solve(n, 1, x, b);
 * @endcode
 */
template <class T>
class BlitILU {

private:
    ILUPcond ilu;       /**< Internal ILU preconditioner structure */
    int    *Ap;         /**< Column pointers (stored reference) */
    int    *Ai;         /**< Row indices (stored reference) */
    double *Ax;         /**< Real part of values (stored reference) */
    double *Az;         /**< Imaginary part of values (stored reference) */
    int nz;             /**< Number of non-zeros */

public:
    /**
     * @brief Construct an ILU preconditioner
     *
     * @param n   Matrix dimension
     * @param nz  Number of non-zero elements
     */
    BlitILU(int n, int nz) {
        Ap = BLIT_NULL;
        Ai = BLIT_NULL;
        Ax = BLIT_NULL;
        Az = BLIT_NULL;
        this->nz = 0;
        ILUPcondCreate(&ilu, &n, &nz);
    }

    /**
     * @brief Destructor - frees UMFPACK resources
     */
    ~BlitILU() noexcept {
        ILUPcondDestroy(&ilu);
    }

    /**
     * @brief Compute the ILU factorization
     *
     * @param App      Pointer to column pointers array
     * @param Aii      Pointer to row indices array
     * @param Axx      Pointer to real values array
     * @param droptol  Drop tolerance for incomplete factorization
     * @param Azz      Pointer to imaginary values array (optional)
     */
    void Run(int **App, int **Aii, double **Axx, double droptol,
             double **Azz = BLIT_NULL) {
        Ap = *App;
        Ai = *Aii;
        Ax = *Axx;
        if (Azz != BLIT_NULL) Az = *Azz;
        ILUPcondPrep(&ilu, Ap, Ai, Ax, &droptol, Az);
    }

    /**
     * @brief Solve a system using the ILU factorization
     *
     * @param nrow  Number of rows
     * @param ncol  Number of right-hand sides
     * @param x     Solution array (output)
     * @param b     Right-hand side array (input)
     * @param xz    Imaginary part of solution (optional)
     * @param bz    Imaginary part of RHS (optional)
     *
     * @throws beNoLHS if matrix not factorized
     * @throws beNoRHS if x or b is null
     */
    void Solve(int nrow, int ncol, double *x, double *b,
               double *xz = BLIT_NULL, double *bz = BLIT_NULL) {
        if (nz == 0 || Ap == 0) throw(beNoLHS);
        if (x == 0 || b == 0) throw(beNoRHS);
        ILUPcondSolve(&ilu, Ap, Ai, Ax, &nrow, &ncol, x, b, Az, xz, bz);
    }
};

/**
 * @brief C++ wrapper class for the BLQMR iterative solver
 *
 * @tparam T  Data type (double for real, F90Complex for complex)
 *
 * @details Provides RAII-style resource management and type-safe interface
 *          to the Block QMR solver. Automatically selects real or complex
 *          Fortran routines based on template parameter.
 *
 * @code
 * // Solve a real system
 * BlitBLQMR<double> solver(n);
 * solver.Prepare(Ap, Ai, Ax, nz);
 * solver.Solve(x, b, nrhs);
 * solver.Print();
 *
 * // Solve a complex system
 * BlitBLQMR<F90Complex> csolver(n);
 * csolver.Prepare(Ap, Ai, Ax_complex, nz);
 * csolver.Solve(x_complex, b_complex, nrhs);
 * @endcode
 */
template <class T>
class BlitBLQMR {

private:
    BLQMRSolver qmr;    /**< Internal solver structure */
    int *Ap;            /**< Column pointers (owned copy) */
    int *Ai;            /**< Row indices (owned copy) */
    T   *Ax;            /**< Matrix values (owned copy) */
    int nz;             /**< Number of non-zeros */

public:
    /**
     * @brief Construct a BLQMR solver with default parameters
     *
     * @param n  Matrix dimension
     *
     * @throws beTypeMismatch if T is neither double nor F90Complex
     */
    BlitBLQMR(int n) {
        Ap = BLIT_NULL;
        Ai = BLIT_NULL;
        Ax = BLIT_NULL;
        nz = 0;
        if (sizeof(T) == sizeof(double))
            DBLQMRCreate(&qmr, &n);
        else if (sizeof(T) == sizeof(F90Complex))
            ZBLQMRCreate(&qmr, &n);
        else
            throw beTypeMismatch;
    }

    /**
     * @brief Construct a BLQMR solver with custom parameters
     *
     * @param n           Matrix dimension
     * @param nrhs        Number of right-hand sides
     * @param maxit       Maximum iterations (default: 100)
     * @param droptol     ILU drop tolerance (default: 1e-3)
     * @param isquasires  Use quasi-residual (default: 0 = true residual)
     * @param debug       Debug output level (default: 1)
     *
     * @throws beTypeMismatch if T is neither double nor F90Complex
     */
    BlitBLQMR(int n, int nrhs, int maxit = 100, double droptol = 1e-3,
              int isquasires = 0, int debug = 1) {
        Ap = BLIT_NULL;
        Ai = BLIT_NULL;
        Ax = BLIT_NULL;
        nz = 0;
        if (sizeof(T) == sizeof(double))
            DBLQMRCreate(&qmr, &qmr.n);
        else if (sizeof(T) == sizeof(F90Complex))
            ZBLQMRCreate(&qmr, &qmr.n);
        else
            throw beTypeMismatch;
        qmr.n = n;
        qmr.nrhs = nrhs;
        qmr.maxit = maxit;
        qmr.isquasires = isquasires;
        qmr.debug = debug;
        qmr.droptol = droptol;
        qmr.pcond_type = (droptol >= 0.0);
    }

    /**
     * @brief Destructor - frees solver and matrix storage
     *
     * @note noexcept since C++11 destructors default to noexcept.
     *       Type validation occurs in constructor, so no mismatch possible here.
     */
    ~BlitBLQMR() noexcept {
        if (sizeof(T) == sizeof(double))
            DBLQMRDestroy(&qmr);
        else if (sizeof(T) == sizeof(F90Complex))
            ZBLQMRDestroy(&qmr);
        /* Note: Type mismatch is impossible here - constructor would have thrown */

        if (Ap) delete[] Ap;
        if (Ai) delete[] Ai;
        if (Ax) delete[] Ax;
    }

    /**
     * @brief Prepare the solver with a sparse matrix
     *
     * @param App  Column pointers array (copied internally)
     * @param Aii  Row indices array (copied internally)
     * @param Axx  Matrix values array (copied internally)
     * @param nzz  Number of non-zero elements
     *
     * @details Makes internal copies of all matrix data and builds
     *          the preconditioner. Can be called multiple times to
     *          update the matrix.
     */
    void Prepare(int *App, int *Aii, T *Axx, int nzz) {
        if (Ap) delete[] Ap;
        if (Ai) delete[] Ai;
        if (Ax) delete[] Ax;

        nz = nzz;
        Ap = new int[qmr.n + 1];
        Ai = new int[nz];
        Ax = new T[nz];
        memcpy(Ap, App, sizeof(int) * (qmr.n + 1));
        memcpy(Ai, Aii, sizeof(int) * nz);
        memcpy(Ax, Axx, sizeof(T) * nz);

        printf("Ap address before: %lx\n", (unsigned long)Ap);
        if (sizeof(T) == sizeof(double))
            DBLQMRPrep(&qmr, Ap, Ai, (double *)Ax, &nz);
        else if (sizeof(T) == sizeof(F90Complex))
            ZBLQMRPrep(&qmr, Ap, Ai, (F90Complex *)Ax, &nz);
        else
            throw beTypeMismatch;
        printf("Ap address after: %lx\n", (unsigned long)Ap);
    }

    /**
     * @brief Solve the linear system A*x = b
     *
     * @param x     Solution vectors (size n x nrhs, initial guess on input)
     * @param b     Right-hand side vectors (size n x nrhs)
     * @param nrhs  Number of right-hand sides
     *
     * @throws beNoLHS if Prepare() has not been called
     * @throws beNoRHS if x or b is null
     *
     * @details After solving, check qmr.flag for convergence status
     *          and qmr.relres for the achieved relative residual.
     */
    void Solve(T *x, T *b, int nrhs) {
        if (nz == 0 || Ap == BLIT_NULL) throw(beNoLHS);
        if (x == BLIT_NULL || b == BLIT_NULL) throw(beNoRHS);

        if (sizeof(T) == sizeof(double))
            DBLQMRSolve(&qmr, Ap, Ai, (double *)Ax, &nz, (double *)x,
                        (double *)b, &nrhs);
        else if (sizeof(T) == sizeof(F90Complex))
            ZBLQMRSolve(&qmr, Ap, Ai, (F90Complex *)Ax, &nz, (F90Complex *)x,
                        (F90Complex *)b, &nrhs);
        else
            throw beTypeMismatch;
    }

    /**
     * @brief Print solver state in JSON format
     */
    void Print() {
        if (sizeof(T) == sizeof(double))
            DBLQMRPrint(&qmr);
        else if (sizeof(T) == sizeof(F90Complex))
            ZBLQMRPrint(&qmr);
        else
            throw beTypeMismatch;
    }
};

/** @} */

#endif /* __cplusplus */