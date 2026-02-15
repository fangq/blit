
#include "mex.h"
#include <string.h>
#include <stddef.h>

extern void blqmr_solve_real_(
    int* n, int* nnz, int* Ap, int* Ai, double* Ax, double* b, double* x,
    int* maxit, double* qtol, double* droptol, int* pcond_type,
    int* flag, int* iter, double* relres);

extern void blqmr_solve_real_multi_(
    int* n, int* nnz, int* nrhs, int* Ap, int* Ai, double* Ax,
    double* B, double* X, int* maxit, double* qtol, double* droptol,
    int* pcond_type, int* flag, int* iter, double* relres);

extern void blqmr_solve_real_multi_omp_(
    int* n, int* nnz, int* nrhs, int* nblock, int* Ap, int* Ai, double* Ax,
    double* B, double* X, int* maxit, double* qtol, double* droptol,
    int* pcond_type, int* flag, int* iter, double* relres);

extern void blqmr_solve_complex_(
    int* n, int* nnz, int* Ap, int* Ai, double* Ax, double* b, double* x,
    int* maxit, double* qtol, double* droptol, int* pcond_type,
    int* flag, int* iter, double* relres);

extern void blqmr_solve_complex_multi_(
    int* n, int* nnz, int* nrhs, int* Ap, int* Ai, double* Ax,
    double* B, double* X, int* maxit, double* qtol, double* droptol,
    int* pcond_type, int* flag, int* iter, double* relres);

extern void blqmr_solve_complex_multi_omp_(
    int* n, int* nnz, int* nrhs, int* nblock, int* Ap, int* Ai, double* Ax,
    double* B, double* X, int* maxit, double* qtol, double* droptol,
    int* pcond_type, int* flag, int* iter, double* relres);

void mexFunction(int nlhs, mxArray* plhs[], int nrhs_in, const mxArray* prhs[]) {
    mwIndex* Ap_mw, *Ai_mw;
    int* Ap_f, *Ai_f;
    int n, nnz_val, num_rhs, maxit, pcond_type, nblock;
    int flag, iter, is_complex;
    double qtol, droptol, relres;
    mwSize i;

    if (nrhs_in < 2)
        mexErrMsgIdAndTxt("blqmr:nrhs",
                          "Usage: [x,flag,relres,iter] = blqmr_(A,B,qtol,maxit,pcond_type,droptol,nblock)");

    if (!mxIsSparse(prhs[0])) {
        mexErrMsgIdAndTxt("blqmr:sparse", "A must be a sparse matrix");
    }

    if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt("blqmr:type", "A and B must be double");
    }

    n = (int)mxGetM(prhs[0]);

    if ((int)mxGetN(prhs[0]) != n) {
        mexErrMsgIdAndTxt("blqmr:square", "A must be square");
    }

    Ap_mw = mxGetJc(prhs[0]);
    Ai_mw = mxGetIr(prhs[0]);
    nnz_val = (int)Ap_mw[n];
    is_complex = mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]);
    num_rhs = (int)mxGetN(prhs[1]);

    if ((int)mxGetM(prhs[1]) != n) {
        mexErrMsgIdAndTxt("blqmr:dim", "B must have n rows");
    }

    qtol       = (nrhs_in > 2 && !mxIsEmpty(prhs[2])) ? mxGetScalar(prhs[2]) : 1e-6;
    maxit      = (nrhs_in > 3 && !mxIsEmpty(prhs[3])) ? (int)mxGetScalar(prhs[3]) : n;
    pcond_type = (nrhs_in > 4 && !mxIsEmpty(prhs[4])) ? (int)mxGetScalar(prhs[4]) : 1;
    droptol    = (nrhs_in > 5 && !mxIsEmpty(prhs[5])) ? mxGetScalar(prhs[5]) : 0.001;
    nblock     = (nrhs_in > 6 && !mxIsEmpty(prhs[6])) ? (int)mxGetScalar(prhs[6]) : 0;

    Ap_f = (int*)mxMalloc((n + 1) * sizeof(int));
    Ai_f = (int*)mxMalloc(nnz_val * sizeof(int));

    for (i = 0; i <= (mwSize)n; i++) {
        Ap_f[i] = (int)Ap_mw[i] + 1;
    }

    for (i = 0; i < (mwSize)nnz_val; i++) {
        Ai_f[i] = (int)Ai_mw[i] + 1;
    }

    if (!is_complex) {
        /* ===== REAL PATH ===== */
        double* Ax = mxGetPr(prhs[0]);
        double* B  = mxGetPr(prhs[1]);

        if (num_rhs == 1) {
            plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
            blqmr_solve_real_(&n, &nnz_val, Ap_f, Ai_f, Ax, B,
                              mxGetPr(plhs[0]),
                              &maxit, &qtol, &droptol, &pcond_type,
                              &flag, &iter, &relres);
        } else {
            plhs[0] = mxCreateDoubleMatrix(n, num_rhs, mxREAL);

            if (nblock > 0 && nblock < num_rhs) {
                blqmr_solve_real_multi_omp_(
                    &n, &nnz_val, &num_rhs, &nblock, Ap_f, Ai_f, Ax, B,
                    mxGetPr(plhs[0]),
                    &maxit, &qtol, &droptol, &pcond_type,
                    &flag, &iter, &relres);
            } else {
                blqmr_solve_real_multi_(
                    &n, &nnz_val, &num_rhs, Ap_f, Ai_f, Ax, B,
                    mxGetPr(plhs[0]),
                    &maxit, &qtol, &droptol, &pcond_type,
                    &flag, &iter, &relres);
            }
        }
    } else {
        /* ===== COMPLEX PATH =====
         * Two storage formats:
         *   Separate (MATLAB R2017b): mxGetPr -> real[nnz], mxGetPi -> imag[nnz]
         *   Interleaved (Octave):     mxGetData -> [re,im,re,im,...] (2*nnz doubles)
         *                             mxGetPi returns NULL
         *
         * Detection: complex matrix with mxGetPi==NULL means interleaved.
         */
        double* Ax_cplx, *B_cplx, *X_cplx;
        mwSize total_b = (mwSize)n * num_rhs;

        /* --- Pack A values --- */
        Ax_cplx = (double*)mxMalloc(2 * nnz_val * sizeof(double));
        {
            double* api = mxIsComplex(prhs[0]) ? mxGetPi(prhs[0]) : NULL;

            if (api == NULL && mxIsComplex(prhs[0])) {
                /* Interleaved (Octave): raw data is already [re,im,...] */
                memcpy(Ax_cplx, mxGetData(prhs[0]), 2 * nnz_val * sizeof(double));
            } else {
                /* Separate (MATLAB): interleave manually */
                double* apr = mxGetPr(prhs[0]);

                for (i = 0; i < (mwSize)nnz_val; i++) {
                    Ax_cplx[2 * i]   = apr[i];
                    Ax_cplx[2 * i + 1] = api ? api[i] : 0.0;
                }
            }
        }

        /* --- Pack B values --- */
        B_cplx = (double*)mxMalloc(2 * total_b * sizeof(double));
        {
            double* bpi = mxIsComplex(prhs[1]) ? mxGetPi(prhs[1]) : NULL;

            if (bpi == NULL && mxIsComplex(prhs[1])) {
                /* Interleaved */
                memcpy(B_cplx, mxGetData(prhs[1]), 2 * total_b * sizeof(double));
            } else {
                /* Separate */
                double* bpr = mxGetPr(prhs[1]);

                for (i = 0; i < total_b; i++) {
                    B_cplx[2 * i]   = bpr[i];
                    B_cplx[2 * i + 1] = bpi ? bpi[i] : 0.0;
                }
            }
        }

        /* --- Solve --- */
        X_cplx = (double*)mxCalloc(2 * total_b, sizeof(double));

        if (num_rhs == 1) {
            blqmr_solve_complex_(&n, &nnz_val, Ap_f, Ai_f, Ax_cplx,
                                 B_cplx, X_cplx,
                                 &maxit, &qtol, &droptol, &pcond_type,
                                 &flag, &iter, &relres);
        } else {
            if (nblock > 0 && nblock < num_rhs) {
                blqmr_solve_complex_multi_omp_(
                    &n, &nnz_val, &num_rhs, &nblock, Ap_f, Ai_f, Ax_cplx,
                    B_cplx, X_cplx,
                    &maxit, &qtol, &droptol, &pcond_type,
                    &flag, &iter, &relres);
            } else {
                blqmr_solve_complex_multi_(
                    &n, &nnz_val, &num_rhs, Ap_f, Ai_f, Ax_cplx,
                    B_cplx, X_cplx,
                    &maxit, &qtol, &droptol, &pcond_type,
                    &flag, &iter, &relres);
            }
        }

        /* --- Unpack X --- */
        plhs[0] = mxCreateDoubleMatrix(n, num_rhs, mxCOMPLEX);
        {
            double* xpi = mxGetPi(plhs[0]);

            if (xpi == NULL) {
                /* Interleaved output (Octave) */
                memcpy(mxGetData(plhs[0]), X_cplx, 2 * total_b * sizeof(double));
            } else {
                /* Separate output (MATLAB) */
                double* xpr = mxGetPr(plhs[0]);

                for (i = 0; i < total_b; i++) {
                    xpr[i] = X_cplx[2 * i];
                    xpi[i] = X_cplx[2 * i + 1];
                }
            }
        }

        mxFree(Ax_cplx);
        mxFree(B_cplx);
        mxFree(X_cplx);
    }

    mxFree(Ap_f);
    mxFree(Ai_f);

    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleScalar((double)flag);
    }

    if (nlhs > 2) {
        plhs[2] = mxCreateDoubleScalar(relres);
    }

    if (nlhs > 3) {
        plhs[3] = mxCreateDoubleScalar((double)iter);
    }
}