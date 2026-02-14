/*************************************************************************
 *
 *  blqmr_mex.c - MATLAB MEX gateway for BLIT BLQMR Fortran solver
 *
 *  Builds as: blqmr_.mex*   (called from blqmr.m when available)
 *
 *  Usage (called internally by blqmr.m):
 *    [x, flag, relres, iter] = blqmr_(A, B, qtol, maxit, pcond_type, droptol, nblock)
 *
 *  Inputs:
 *    A          - Sparse n x n matrix (real or complex)
 *    B          - RHS matrix, n x nrhs (real or complex)
 *    qtol       - Convergence tolerance (scalar, default 1e-6)
 *    maxit      - Maximum iterations (scalar, default n)
 *    pcond_type - Preconditioner: 0=none, 1=ILU-left, 2=ILU-split,
 *                 3=Jacobi-split (scalar, default 1)
 *    droptol    - ILU drop tolerance (scalar, default 0.001)
 *    nblock     - Block size for OpenMP parallelism (scalar, default 0)
 *                 0 = all RHS in one block; >0 = partition into chunks
 *
 *  Outputs:
 *    x       - Solution matrix (n x nrhs)
 *    flag    - Convergence flag (0=success, 1=maxiter, 2=precond fail, 3=stagnated)
 *    relres  - Relative residual
 *    iter    - Iterations performed
 *
 *  Place in: matlab/blqmr_mex.c
 *
 *************************************************************************/

#include "mex.h"
#include <string.h>
#include <math.h>

/* ---- Fortran subroutine declarations (trailing underscore) ---- */

extern void blqmr_solve_real_(
    int *n, int *nnz, int *Ap, int *Ai, double *Ax, double *b, double *x,
    int *maxit, double *qtol, double *droptol, int *pcond_type,
    int *flag, int *iter, double *relres);

extern void blqmr_solve_real_multi_(
    int *n, int *nnz, int *nrhs, int *Ap, int *Ai, double *Ax,
    double *B, double *X,
    int *maxit, double *qtol, double *droptol, int *pcond_type,
    int *flag, int *iter, double *relres);

extern void blqmr_solve_real_multi_omp_(
    int *n, int *nnz, int *nrhs, int *nblock,
    int *Ap, int *Ai, double *Ax, double *B, double *X,
    int *maxit, double *qtol, double *droptol, int *pcond_type,
    int *flag, int *iter, double *relres);

extern void blqmr_solve_complex_(
    int *n, int *nnz, int *Ap, int *Ai, double *Ax, double *b, double *x,
    int *maxit, double *qtol, double *droptol, int *pcond_type,
    int *flag, int *iter, double *relres);

extern void blqmr_solve_complex_multi_(
    int *n, int *nnz, int *nrhs, int *Ap, int *Ai, double *Ax,
    double *B, double *X,
    int *maxit, double *qtol, double *droptol, int *pcond_type,
    int *flag, int *iter, double *relres);

extern void blqmr_solve_complex_multi_omp_(
    int *n, int *nnz, int *nrhs, int *nblock,
    int *Ap, int *Ai, double *Ax, double *B, double *X,
    int *maxit, double *qtol, double *droptol, int *pcond_type,
    int *flag, int *iter, double *relres);


/* ---- Helper: interleave separate real/imag arrays into Fortran complex ---- */
static double *interleave_complex(const double *re, const double *im, mwSize len)
{
    double *out = (double *)mxMalloc(2 * len * sizeof(double));
    mwSize i;
    for (i = 0; i < len; i++) {
        out[2 * i]     = re[i];
        out[2 * i + 1] = im ? im[i] : 0.0;
    }
    return out;
}

/* ---- Helper: de-interleave Fortran complex into separate real/imag ---- */
static void deinterleave_complex(const double *src, double *re, double *im, mwSize len)
{
    mwSize i;
    for (i = 0; i < len; i++) {
        re[i] = src[2 * i];
        im[i] = src[2 * i + 1];
    }
}


/*
 * Pin MATLAB's internal BLAS (MKL) to single-threaded mode.
 * MATLAB hijacks standard BLAS symbols (dgemm etc.) to its own MKL.
 * MKL's threaded dgemm can crash with stack overflow when called from
 * a MEX Fortran solver. We use maxNumCompThreads(1) to force sequential.
 */
static double matlab_set_comp_threads(int n) {
    mxArray *lhs[1] = {NULL}, *rhs[1];
    double prev = 1;
    rhs[0] = mxCreateDoubleScalar((double)n);
    if (mexCallMATLAB(1, lhs, 1, rhs, "maxNumCompThreads") == 0 && lhs[0])
        prev = mxGetScalar(lhs[0]);
    mxDestroyArray(rhs[0]);
    if (lhs[0]) mxDestroyArray(lhs[0]);
    return prev;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs_in, const mxArray *prhs[])
{
    mwIndex *Ap_mw, *Ai_mw;
    double *Ax_pr;
    int *Ap_f, *Ai_f;
    int n, nnz, num_rhs, maxit, pcond_type, nblock;
    int flag, iter, is_complex;
    double qtol, droptol, relres;
    mwSize i;
    double saved_threads;

    /* ---- Validate minimum inputs ---- */
    if (nrhs_in < 2)
        mexErrMsgIdAndTxt("blqmr:nrhs",
            "Usage: [x,flag,relres,iter] = blqmr_(A, B, qtol, maxit, pcond_type, droptol, nblock)");

    if (!mxIsSparse(prhs[0]))
        mexErrMsgIdAndTxt("blqmr:sparse", "A must be a sparse matrix");
    if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))
        mexErrMsgIdAndTxt("blqmr:type", "A and B must be double");

    /* ---- Extract sparse matrix A in CSC format ---- */
    n   = (int)mxGetM(prhs[0]);
    if ((int)mxGetN(prhs[0]) != n)
        mexErrMsgIdAndTxt("blqmr:square", "A must be square");

    Ap_mw = mxGetJc(prhs[0]);
    Ai_mw = mxGetIr(prhs[0]);
    nnz   = (int)Ap_mw[n];

    is_complex = mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]);

    /* RHS */
    num_rhs = (int)mxGetN(prhs[1]);
    if ((int)mxGetM(prhs[1]) != n)
        mexErrMsgIdAndTxt("blqmr:dim", "B must have %d rows", n);

    /* ---- Parse optional parameters with defaults ---- */
    qtol       = (nrhs_in > 2 && !mxIsEmpty(prhs[2])) ? mxGetScalar(prhs[2]) : 1e-6;
    maxit      = (nrhs_in > 3 && !mxIsEmpty(prhs[3])) ? (int)mxGetScalar(prhs[3]) : n;
    pcond_type = (nrhs_in > 4 && !mxIsEmpty(prhs[4])) ? (int)mxGetScalar(prhs[4]) : 1;
    droptol    = (nrhs_in > 5 && !mxIsEmpty(prhs[5])) ? mxGetScalar(prhs[5]) : 0.001;
    nblock     = (nrhs_in > 6 && !mxIsEmpty(prhs[6])) ? (int)mxGetScalar(prhs[6]) : 0;

    /* ---- Convert MATLAB mwIndex (0-based) to Fortran int32 (1-based) ---- */
    Ap_f = (int *)mxMalloc((n + 1) * sizeof(int));
    Ai_f = (int *)mxMalloc(nnz * sizeof(int));

    for (i = 0; i <= (mwSize)n; i++)
        Ap_f[i] = (int)Ap_mw[i] + 1;
    for (i = 0; i < (mwSize)nnz; i++)
        Ai_f[i] = (int)Ai_mw[i] + 1;

    /* ---- Pin MATLAB's BLAS to 1 thread ---- */
    saved_threads = matlab_set_comp_threads(1);
    /* Also set MKL env var directly - maxNumCompThreads may not affect MKL */
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("MKL_DYNAMIC", "FALSE", 1);

    /* ---- Minimum nrhs padding for MKL AVX2 compatibility ---- */
    /* MKL's AVX2 dgemm kernel reads 32 bytes (4 doubles) at a time.
     * The Fortran solver allocates (nrhs x nrhs) workspace matrices on the
     * stack. When nrhs < 4, these are smaller than 128 bytes and MKL
     * over-reads past the allocation. Pad nrhs to at least 4 and use
     * the multi-RHS entry point for all cases. */
#define BLQMR_MIN_NRHS 4

    /* ---- Dispatch: real vs complex ---- */
    if (!is_complex) {
        /* ============ REAL PATH ============ */
        double *Ax_vals = mxGetPr(prhs[0]);
        double *B_vals  = mxGetPr(prhs[1]);
        int nrhs_actual = num_rhs;
        int nrhs_padded = (num_rhs < BLQMR_MIN_NRHS) ? BLQMR_MIN_NRHS : num_rhs;

        double *B_pad = (double *)mxCalloc((mwSize)n * nrhs_padded, sizeof(double));
        double *X_pad = (double *)mxCalloc((mwSize)n * nrhs_padded, sizeof(double));
        memcpy(B_pad, B_vals, (mwSize)n * nrhs_actual * sizeof(double));

        plhs[0] = mxCreateDoubleMatrix(n, nrhs_actual, mxREAL);

        if (nblock > 0 && nblock < nrhs_padded) {
            blqmr_solve_real_multi_omp_(
                &n, &nnz, &nrhs_padded, &nblock, Ap_f, Ai_f, Ax_vals,
                B_pad, X_pad,
                &maxit, &qtol, &droptol, &pcond_type,
                &flag, &iter, &relres);
        } else {
            blqmr_solve_real_multi_(
                &n, &nnz, &nrhs_padded, Ap_f, Ai_f, Ax_vals,
                B_pad, X_pad,
                &maxit, &qtol, &droptol, &pcond_type,
                &flag, &iter, &relres);
        }

        memcpy(mxGetPr(plhs[0]), X_pad, (mwSize)n * nrhs_actual * sizeof(double));
        mxFree(B_pad);
        mxFree(X_pad);
    } else {
        /* ============ COMPLEX PATH ============ */
        /* Pack MATLAB complex data into Fortran interleaved format */
        double *Ax_cplx, *B_cplx, *X_cplx;

#if MX_HAS_INTERLEAVED_COMPLEX
        /* R2018a+: already interleaved, can cast directly */
        Ax_cplx = (double *)mxGetComplexDoubles(prhs[0]);
        B_cplx  = (double *)mxGetComplexDoubles(prhs[1]);
#else
        /* Pre-R2018a: separate real/imag parts */
        Ax_cplx = interleave_complex(mxGetPr(prhs[0]), mxGetPi(prhs[0]), nnz);
        B_cplx  = interleave_complex(mxGetPr(prhs[1]), mxGetPi(prhs[1]),
                                     (mwSize)n * num_rhs);
#endif

        int nrhs_actual = num_rhs;
        int nrhs_padded = (num_rhs < BLQMR_MIN_NRHS) ? BLQMR_MIN_NRHS : num_rhs;

        X_cplx = (double *)mxCalloc(2 * (mwSize)n * nrhs_padded, sizeof(double));

        /* Pad B if needed */
        double *B_solve = B_cplx;
        double *B_pad_cplx = NULL;
        if (nrhs_padded > nrhs_actual) {
            B_pad_cplx = (double *)mxCalloc(2 * (mwSize)n * nrhs_padded, sizeof(double));
            memcpy(B_pad_cplx, B_cplx, 2 * (mwSize)n * nrhs_actual * sizeof(double));
            B_solve = B_pad_cplx;
        }

        if (nblock > 0 && nblock < nrhs_padded) {
            blqmr_solve_complex_multi_omp_(
                &n, &nnz, &nrhs_padded, &nblock, Ap_f, Ai_f, Ax_cplx,
                B_solve, X_cplx,
                &maxit, &qtol, &droptol, &pcond_type,
                &flag, &iter, &relres);
        } else {
            blqmr_solve_complex_multi_(
                &n, &nnz, &nrhs_padded, Ap_f, Ai_f, Ax_cplx,
                B_solve, X_cplx,
                &maxit, &qtol, &droptol, &pcond_type,
                &flag, &iter, &relres);
        }

        if (B_pad_cplx) mxFree(B_pad_cplx);

        /* Copy result back to MATLAB complex output */
        plhs[0] = mxCreateDoubleMatrix(n, num_rhs, mxCOMPLEX);
#if MX_HAS_INTERLEAVED_COMPLEX
        memcpy(mxGetComplexDoubles(plhs[0]), X_cplx,
               2 * (mwSize)n * num_rhs * sizeof(double));
#else
        deinterleave_complex(X_cplx, mxGetPr(plhs[0]), mxGetPi(plhs[0]),
                             (mwSize)n * num_rhs);
        mxFree(Ax_cplx);
        mxFree(B_cplx);
#endif
        mxFree(X_cplx);
    }

    /* ---- Cleanup ---- */
    mxFree(Ap_f);
    mxFree(Ai_f);

    /* ---- Restore MATLAB thread count ---- */
    matlab_set_comp_threads((int)saved_threads);

    /* ---- Return scalar outputs ---- */
    if (nlhs > 1) plhs[1] = mxCreateDoubleScalar((double)flag);
    if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(relres);
    if (nlhs > 3) plhs[3] = mxCreateDoubleScalar((double)iter);
}