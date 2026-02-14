!*************************************************************************
!
!  blqmr_mex.f90 - Fortran MEX gateway for BLIT BLQMR solver
!
!  Builds as: blqmr_.mex*  (called from blqmr.m when available)
!
!  Usage (from MATLAB, called by blqmr.m):
!    [x, flag, relres, iter] = blqmr_(A, B, qtol, maxit, pcond_type, droptol, nblock)
!
!  Place in: matlab/src/blqmr_mex.f90
!
!*************************************************************************

#include "fintrf.h"

subroutine mexFunction(nlhs, plhs, nrhs, prhs)
    use blit_precision
    implicit none

    mwPointer :: plhs(*), prhs(*)
    integer :: nlhs, nrhs

    mwPointer :: mxGetPr, mxGetPi, mxCreateDoubleMatrix
    mwPointer :: mxGetJc, mxGetIr
    mwSize    :: mxGetM, mxGetN
    integer*4 :: mxIsComplex, mxIsSparse, mxIsDouble, mxIsEmpty
    real*8    :: mxGetScalar

    mwPointer :: Ap_ptr, Ai_ptr, Ax_ptr
    mwPointer :: B_ptr, X_ptr
    mwSize :: n_mw, nrhs_mw, nnz_mw, one_mw, np1_mw
    integer :: nn, nnz_val, num_rhs, maxit, pcond_type, nblock
    real(kind=Kdouble) :: qtol, droptol, relres
    integer :: flag_out, iter_out, is_complex
    mwSize :: i_mw

    integer, allocatable :: Ap(:), Ai(:)
    real(kind=Kdouble), allocatable :: Ax_real(:), B_real(:,:), X_real(:,:)
    complex(kind=Kdouble), allocatable :: Ax_cmplx(:), B_cmplx(:,:), X_cmplx(:,:)
    real(kind=Kdouble), allocatable :: tmp_re(:), tmp_im(:)
    real(kind=Kdouble) :: tmp_scalar(1)
    integer*8, allocatable :: Ap_i8(:), Ai_i8(:)

    one_mw = 1

    ! ----- Validate inputs -----
    if (nrhs < 2) then
        call mexErrMsgIdAndTxt('blqmr:nrhs', &
            'Usage: [x,flag,relres,iter] = blqmr_(A,B,qtol,maxit,pcond_type,droptol,nblock)')
        return
    endif

    if (mxIsSparse(prhs(1)) /= 1) then
        call mexErrMsgIdAndTxt('blqmr:sparse', 'A must be a sparse matrix')
        return
    endif
    if (mxIsDouble(prhs(1)) /= 1 .or. mxIsDouble(prhs(2)) /= 1) then
        call mexErrMsgIdAndTxt('blqmr:type', 'A and B must be double')
        return
    endif

    ! ----- Get matrix dimensions -----
    n_mw = mxGetM(prhs(1))
    nn = int(n_mw)
    if (mxGetN(prhs(1)) /= n_mw) then
        call mexErrMsgIdAndTxt('blqmr:square', 'A must be square')
        return
    endif

    nrhs_mw = mxGetN(prhs(2))
    num_rhs = int(nrhs_mw)
    if (mxGetM(prhs(2)) /= n_mw) then
        call mexErrMsgIdAndTxt('blqmr:dim', 'B row count must match A')
        return
    endif

    is_complex = mxIsComplex(prhs(1)) + mxIsComplex(prhs(2))

    ! ----- Parse optional parameters -----
    qtol = 1.0d-6
    if (nrhs >= 3 .and. mxIsEmpty(prhs(3)) == 0) qtol = mxGetScalar(prhs(3))
    maxit = nn
    if (nrhs >= 4 .and. mxIsEmpty(prhs(4)) == 0) maxit = int(mxGetScalar(prhs(4)))
    pcond_type = 1
    if (nrhs >= 5 .and. mxIsEmpty(prhs(5)) == 0) pcond_type = int(mxGetScalar(prhs(5)))
    droptol = 0.001d0
    if (nrhs >= 6 .and. mxIsEmpty(prhs(6)) == 0) droptol = mxGetScalar(prhs(6))
    nblock = 0
    if (nrhs >= 7 .and. mxIsEmpty(prhs(7)) == 0) nblock = int(mxGetScalar(prhs(7)))

    ! ----- Extract CSC sparse structure -----
    Ap_ptr = mxGetJc(prhs(1))
    Ai_ptr = mxGetIr(prhs(1))
    Ax_ptr = mxGetPr(prhs(1))

    ! Column pointers: mwIndex (8 bytes) -> integer*4 (1-based)
    allocate(Ap_i8(nn + 1))
    np1_mw = n_mw + 1
    call mxCopyPtrToInteger8(Ap_ptr, Ap_i8, np1_mw)
    nnz_mw = Ap_i8(nn + 1)
    nnz_val = int(nnz_mw)

    allocate(Ap(nn + 1))
    do i_mw = 1, nn + 1
        Ap(i_mw) = int(Ap_i8(i_mw)) + 1
    enddo
    deallocate(Ap_i8)

    ! Row indices: mwIndex (8 bytes) -> integer*4 (1-based)
    allocate(Ai_i8(nnz_val))
    call mxCopyPtrToInteger8(Ai_ptr, Ai_i8, nnz_mw)
    allocate(Ai(nnz_val))
    do i_mw = 1, nnz_val
        Ai(i_mw) = int(Ai_i8(i_mw)) + 1
    enddo
    deallocate(Ai_i8)

    ! ----- Dispatch: real vs complex -----
    if (is_complex == 0) then
        allocate(Ax_real(nnz_val))
        call mxCopyPtrToReal8(Ax_ptr, Ax_real, nnz_mw)

        allocate(B_real(nn, num_rhs))
        B_ptr = mxGetPr(prhs(2))
        call mxCopyPtrToReal8(B_ptr, B_real, n_mw * nrhs_mw)

        allocate(X_real(nn, num_rhs))
        X_real = 0.0_Kdouble

        if (num_rhs == 1) then
            call blqmr_solve_real(nn, nnz_val, Ap, Ai, Ax_real, &
                B_real(:,1), X_real(:,1), maxit, qtol, droptol, pcond_type, &
                flag_out, iter_out, relres)
        else if (nblock > 0 .and. nblock < num_rhs) then
            call blqmr_solve_real_multi_omp(nn, nnz_val, num_rhs, nblock, &
                Ap, Ai, Ax_real, B_real, X_real, &
                maxit, qtol, droptol, pcond_type, &
                flag_out, iter_out, relres)
        else
            call blqmr_solve_real_multi(nn, nnz_val, num_rhs, &
                Ap, Ai, Ax_real, B_real, X_real, &
                maxit, qtol, droptol, pcond_type, &
                flag_out, iter_out, relres)
        endif

        plhs(1) = mxCreateDoubleMatrix(n_mw, nrhs_mw, 0)
        X_ptr = mxGetPr(plhs(1))
        call mxCopyReal8ToPtr(X_real, X_ptr, n_mw * nrhs_mw)
        deallocate(Ax_real, B_real, X_real)
    else
        allocate(Ax_cmplx(nnz_val))
        allocate(tmp_re(nnz_val), tmp_im(nnz_val))
        call mxCopyPtrToReal8(mxGetPr(prhs(1)), tmp_re, nnz_mw)
        if (mxIsComplex(prhs(1)) == 1) then
            call mxCopyPtrToReal8(mxGetPi(prhs(1)), tmp_im, nnz_mw)
        else
            tmp_im = 0.0_Kdouble
        endif
        Ax_cmplx = cmplx(tmp_re, tmp_im, kind=Kdouble)
        deallocate(tmp_re, tmp_im)

        allocate(B_cmplx(nn, num_rhs))
        allocate(tmp_re(nn * num_rhs), tmp_im(nn * num_rhs))
        call mxCopyPtrToReal8(mxGetPr(prhs(2)), tmp_re, n_mw * nrhs_mw)
        if (mxIsComplex(prhs(2)) == 1) then
            call mxCopyPtrToReal8(mxGetPi(prhs(2)), tmp_im, n_mw * nrhs_mw)
        else
            tmp_im = 0.0_Kdouble
        endif
        B_cmplx = reshape(cmplx(tmp_re, tmp_im, kind=Kdouble), [nn, num_rhs])
        deallocate(tmp_re, tmp_im)

        allocate(X_cmplx(nn, num_rhs))
        X_cmplx = (0.0_Kdouble, 0.0_Kdouble)

        if (num_rhs == 1) then
            call blqmr_solve_complex(nn, nnz_val, Ap, Ai, Ax_cmplx, &
                B_cmplx(:,1), X_cmplx(:,1), maxit, qtol, droptol, pcond_type, &
                flag_out, iter_out, relres)
        else if (nblock > 0 .and. nblock < num_rhs) then
            call blqmr_solve_complex_multi_omp(nn, nnz_val, num_rhs, nblock, &
                Ap, Ai, Ax_cmplx, B_cmplx, X_cmplx, &
                maxit, qtol, droptol, pcond_type, &
                flag_out, iter_out, relres)
        else
            call blqmr_solve_complex_multi(nn, nnz_val, num_rhs, &
                Ap, Ai, Ax_cmplx, B_cmplx, X_cmplx, &
                maxit, qtol, droptol, pcond_type, &
                flag_out, iter_out, relres)
        endif

        plhs(1) = mxCreateDoubleMatrix(n_mw, nrhs_mw, 1)
        allocate(tmp_re(nn * num_rhs), tmp_im(nn * num_rhs))
        tmp_re = reshape(real(X_cmplx, Kdouble), [nn * num_rhs])
        tmp_im = reshape(aimag(X_cmplx), [nn * num_rhs])
        call mxCopyReal8ToPtr(tmp_re, mxGetPr(plhs(1)), n_mw * nrhs_mw)
        call mxCopyReal8ToPtr(tmp_im, mxGetPi(plhs(1)), n_mw * nrhs_mw)
        deallocate(tmp_re, tmp_im)
        deallocate(Ax_cmplx, B_cmplx, X_cmplx)
    endif

    ! ----- Cleanup -----
    deallocate(Ap, Ai)

    ! ----- Scalar outputs -----
    if (nlhs > 1) then
        plhs(2) = mxCreateDoubleMatrix(one_mw, one_mw, 0)
        tmp_scalar(1) = dble(flag_out)
        call mxCopyReal8ToPtr(tmp_scalar, mxGetPr(plhs(2)), one_mw)
    endif
    if (nlhs > 2) then
        plhs(3) = mxCreateDoubleMatrix(one_mw, one_mw, 0)
        tmp_scalar(1) = relres
        call mxCopyReal8ToPtr(tmp_scalar, mxGetPr(plhs(3)), one_mw)
    endif
    if (nlhs > 3) then
        plhs(4) = mxCreateDoubleMatrix(one_mw, one_mw, 0)
        tmp_scalar(1) = dble(iter_out)
        call mxCopyReal8ToPtr(tmp_scalar, mxGetPr(plhs(4)), one_mw)
    endif

end subroutine mexFunction