/*************************************************************************
 *
 *  Blit - An open-source library for block iterative sparse linear solvers
 *
 *  Runtime BLAS thread control for OpenMP compatibility.
 *
 *  Detects OpenBLAS, MKL, or Apple Accelerate at runtime and
 *  provides get/set thread count functions callable from Fortran.
 *
 *  Fortran interface:
 *    call blit_set_blas_threads(nthreads)
 *    call blit_get_blas_threads(nthreads)
 *
 *************************************************************************/

#include <stdlib.h>

/*
 * Weak symbol declarations — these resolve to NULL if the BLAS library
 * doesn't provide them, avoiding link errors.
 */
#ifdef __GNUC__

    /* OpenBLAS */
    #pragma weak openblas_set_num_threads
    #pragma weak openblas_get_num_threads
    extern void openblas_set_num_threads(int num_threads);
    extern int  openblas_get_num_threads(void);

    /* MKL */
    #pragma weak MKL_Set_Num_Threads
    #pragma weak MKL_Get_Max_Threads
    extern void MKL_Set_Num_Threads(int num_threads);
    extern int  MKL_Get_Max_Threads(void);

    /* BLIS */
    #pragma weak bli_thread_set_num_threads
    #pragma weak bli_thread_get_num_threads
    extern void bli_thread_set_num_threads(int num_threads);
    extern int  bli_thread_get_num_threads(void);

#else
    /* Non-GCC: disable weak symbols, use environment variables only */
    #define openblas_set_num_threads NULL
    #define openblas_get_num_threads NULL
    #define MKL_Set_Num_Threads      NULL
    #define MKL_Get_Max_Threads      NULL
    #define bli_thread_set_num_threads NULL
    #define bli_thread_get_num_threads NULL
#endif

/**
 * Set BLAS thread count at runtime.
 * Tries OpenBLAS, MKL, and BLIS in order.
 * Fortran: call blit_set_blas_threads(n)
 */
void blit_set_blas_threads_(int* nthreads) {
    int n = *nthreads;

    if (n < 1) {
        n = 1;
    }

#ifdef __GNUC__

    if (openblas_set_num_threads) {
        openblas_set_num_threads(n);
        return;
    }

    if (MKL_Set_Num_Threads) {
        MKL_Set_Num_Threads(n);
        return;
    }

    if (bli_thread_set_num_threads) {
        bli_thread_set_num_threads(n);
        return;
    }

#endif
    /* Fallback: nothing to do — BLAS may be single-threaded or uncontrollable */
}

/**
 * Get current BLAS thread count.
 * Tries OpenBLAS, MKL, and BLIS in order.
 * Fortran: call blit_get_blas_threads(n)
 */
void blit_get_blas_threads_(int* nthreads) {
#ifdef __GNUC__

    if (openblas_get_num_threads) {
        *nthreads = openblas_get_num_threads();
        return;
    }

    if (MKL_Get_Max_Threads) {
        *nthreads = MKL_Get_Max_Threads();
        return;
    }

    if (bli_thread_get_num_threads) {
        *nthreads = bli_thread_get_num_threads();
        return;
    }

#endif
    /* Fallback: assume 1 thread */
    *nthreads = 1;
}
