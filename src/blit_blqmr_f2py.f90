!!*************************************************************************
!!
!!  Blit - An open-source library for block iterative sparse linear solvers
!!  Python F2PY Wrapper - Standalone subroutines (no derived types exposed)
!!
!!  Copyright 2011,2020 Qianqian Fang <q.fang at neu.edu>
!!
!!  Place this file in: src/blit_blqmr_f2py.f90
!!
!!*************************************************************************

!> @brief Solve sparse system Ax = b (single RHS) - F2PY interface
!>
!> @param[in]  n        Matrix dimension
!> @param[in]  nnz      Number of non-zeros
!> @param[in]  Ap       Column pointers (1-based, size n+1)
!> @param[in]  Ai       Row indices (1-based, size nnz)
!> @param[in]  Ax       Non-zero values (size nnz)
!> @param[in]  b        Right-hand side vector (size n)
!> @param[out] x        Solution vector (size n)
!> @param[in]  maxit    Maximum iterations
!> @param[in]  qtol     Convergence tolerance
!> @param[in]  droptol  ILU drop tolerance
!> @param[in]  pcond_type  Use preconditioner (1=yes, 0=no)
!> @param[out] flag     Convergence flag (0=success)
!> @param[out] iter     Iterations performed
!> @param[out] relres   Relative residual

subroutine blqmr_solve_real(n, nnz, Ap, Ai, Ax, b, x, &
                            maxit, qtol, droptol, pcond_type, &
                            flag, iter, relres)
    use blit_precision
    use blit_blqmr_real
    implicit none

    !f2py intent(in) n, nnz, maxit, pcond_type
    !f2py intent(in) qtol, droptol
    !f2py intent(in) Ap, Ai, Ax, b
    !f2py intent(out) x, flag, iter, relres
    !f2py depend(n) Ap, b, x
    !f2py depend(nnz) Ai, Ax

    integer, intent(in) :: n, nnz, maxit, pcond_type
    real(kind=Kdouble), intent(in) :: qtol, droptol
    integer, intent(in) :: Ap(n+1), Ai(nnz)
    real(kind=Kdouble), intent(in) :: Ax(nnz), b(n)
    real(kind=Kdouble), intent(out) :: x(n), relres
    integer, intent(out) :: flag, iter

    ! Local variables - copies of input arrays since Fortran routines may modify
    type(BLQMRSolver) :: qmr
    integer :: nnz_local
    integer :: Ap_local(n+1), Ai_local(nnz)
    real(kind=Kdouble) :: Ax_local(nnz), b_local(n,1), x_local(n,1)

    ! Copy all input arrays
    nnz_local = nnz
    Ap_local = Ap
    Ai_local = Ai
    Ax_local = Ax
    b_local(:,1) = b
    x_local = 0.0_Kdouble

    ! Initialize solver
    call BLQMRCreate(qmr, n)
    qmr%maxit = maxit
    qmr%qtol = qtol
    qmr%droptol = droptol
    qmr%pcond_type = pcond_type
    qmr%isquasires = 0

    ! Prepare preconditioner and solve
    call BLQMRPrep(qmr, Ap_local, Ai_local, Ax_local, nnz_local)
    call BLQMRSolve(qmr, Ap_local, Ai_local, Ax_local, nnz_local, x_local, b_local, 1)

    ! Extract results
    x = x_local(:,1)
    flag = qmr%flag
    iter = qmr%iter
    relres = qmr%relres

    ! Cleanup
    call BLQMRDestroy(qmr)

end subroutine blqmr_solve_real


!> @brief Solve sparse system AX = B (multiple RHS) - F2PY interface

subroutine blqmr_solve_real_multi(n, nnz, nrhs, Ap, Ai, Ax, B, X, &
                                   maxit, qtol, droptol, pcond_type, &
                                   flag, iter, relres)
    use blit_precision
    use blit_blqmr_real
    implicit none

    !f2py intent(in) n, nnz, nrhs, maxit, pcond_type
    !f2py intent(in) qtol, droptol
    !f2py intent(in) Ap, Ai, Ax, B
    !f2py intent(out) X, flag, iter, relres
    !f2py depend(n) Ap
    !f2py depend(nnz) Ai, Ax
    !f2py depend(n,nrhs) B, X

    integer, intent(in) :: n, nnz, nrhs, maxit, pcond_type
    real(kind=Kdouble), intent(in) :: qtol, droptol
    integer, intent(in) :: Ap(n+1), Ai(nnz)
    real(kind=Kdouble), intent(in) :: Ax(nnz), B(n, nrhs)
    real(kind=Kdouble), intent(out) :: X(n, nrhs), relres
    integer, intent(out) :: flag, iter

    ! Local variables
    type(BLQMRSolver) :: qmr
    integer :: nnz_local
    integer :: Ap_local(n+1), Ai_local(nnz)
    real(kind=Kdouble) :: Ax_local(nnz), B_local(n, nrhs), X_local(n, nrhs)

    ! Copy arrays
    nnz_local = nnz
    Ap_local = Ap
    Ai_local = Ai
    Ax_local = Ax
    B_local = B
    X_local = 0.0_Kdouble

    ! Initialize solver
    call BLQMRCreate(qmr, n)
    qmr%maxit = maxit
    qmr%qtol = qtol
    qmr%droptol = droptol
    qmr%pcond_type = pcond_type
    qmr%isquasires = 0

    ! Solve
    call BLQMRPrep(qmr, Ap_local, Ai_local, Ax_local, nnz_local)
    call BLQMRSolve(qmr, Ap_local, Ai_local, Ax_local, nnz_local, X_local, B_local, nrhs)

    ! Results
    X = X_local
    flag = qmr%flag
    iter = qmr%iter
    relres = qmr%relres

    ! Cleanup
    call BLQMRDestroy(qmr)

end subroutine blqmr_solve_real_multi


!> @brief Solve complex sparse system (single RHS)

subroutine blqmr_solve_complex(n, nnz, Ap, Ai, Ax, b, x, &
                               maxit, qtol, droptol, pcond_type, &
                               flag, iter, relres)
    use blit_precision
    use blit_blqmr_complex
    implicit none

    !f2py intent(in) n, nnz, maxit, pcond_type
    !f2py intent(in) qtol, droptol
    !f2py intent(in) Ap, Ai, Ax, b
    !f2py intent(out) x, flag, iter, relres
    !f2py depend(n) Ap, b, x
    !f2py depend(nnz) Ai, Ax

    integer, intent(in) :: n, nnz, maxit, pcond_type
    real(kind=Kdouble), intent(in) :: qtol, droptol
    integer, intent(in) :: Ap(n+1), Ai(nnz)
    complex(kind=Kdouble), intent(in) :: Ax(nnz), b(n)
    complex(kind=Kdouble), intent(out) :: x(n)
    real(kind=Kdouble), intent(out) :: relres
    integer, intent(out) :: flag, iter

    ! Local variables
    type(BLQMRSolver) :: qmr
    integer :: nnz_local
    integer :: Ap_local(n+1), Ai_local(nnz)
    complex(kind=Kdouble) :: Ax_local(nnz), b_local(n,1), x_local(n,1)

    nnz_local = nnz
    Ap_local = Ap
    Ai_local = Ai
    Ax_local = Ax
    b_local(:,1) = b
    x_local = (0.0_Kdouble, 0.0_Kdouble)

    call BLQMRCreate(qmr, n)
    qmr%maxit = maxit
    qmr%qtol = qtol
    qmr%droptol = droptol
    qmr%pcond_type = pcond_type
    qmr%isquasires = 0

    call BLQMRPrep(qmr, Ap_local, Ai_local, Ax_local, nnz_local)
    call BLQMRSolve(qmr, Ap_local, Ai_local, Ax_local, nnz_local, x_local, b_local, 1)

    x = x_local(:,1)
    flag = qmr%flag
    iter = qmr%iter
    relres = qmr%relres

    call BLQMRDestroy(qmr)

end subroutine blqmr_solve_complex


!> @brief Solve complex sparse system AX = B (multiple RHS) - F2PY interface

subroutine blqmr_solve_complex_multi(n, nnz, nrhs, Ap, Ai, Ax, B, X, &
                                      maxit, qtol, droptol, pcond_type, &
                                      flag, iter, relres)
    use blit_precision
    use blit_blqmr_complex
    implicit none

    !f2py intent(in) n, nnz, nrhs, maxit, pcond_type
    !f2py intent(in) qtol, droptol
    !f2py intent(in) Ap, Ai, Ax, B
    !f2py intent(out) X, flag, iter, relres
    !f2py depend(n) Ap
    !f2py depend(nnz) Ai, Ax
    !f2py depend(n,nrhs) B, X

    integer, intent(in) :: n, nnz, nrhs, maxit, pcond_type
    real(kind=Kdouble), intent(in) :: qtol, droptol
    integer, intent(in) :: Ap(n+1), Ai(nnz)
    complex(kind=Kdouble), intent(in) :: Ax(nnz), B(n, nrhs)
    complex(kind=Kdouble), intent(out) :: X(n, nrhs)
    real(kind=Kdouble), intent(out) :: relres
    integer, intent(out) :: flag, iter

    ! Local variables
    type(BLQMRSolver) :: qmr
    integer :: nnz_local
    integer :: Ap_local(n+1), Ai_local(nnz)
    complex(kind=Kdouble) :: Ax_local(nnz), B_local(n, nrhs), X_local(n, nrhs)

    ! Copy arrays
    nnz_local = nnz
    Ap_local = Ap
    Ai_local = Ai
    Ax_local = Ax
    B_local = B
    X_local = (0.0_Kdouble, 0.0_Kdouble)

    ! Initialize solver
    call BLQMRCreate(qmr, n)
    qmr%maxit = maxit
    qmr%qtol = qtol
    qmr%droptol = droptol
    qmr%pcond_type = pcond_type
    qmr%isquasires = 0

    ! Solve with multiple RHS (true block method)
    call BLQMRPrep(qmr, Ap_local, Ai_local, Ax_local, nnz_local)
    call BLQMRSolve(qmr, Ap_local, Ai_local, Ax_local, nnz_local, X_local, B_local, nrhs)

    ! Results
    X = X_local
    flag = qmr%flag
    iter = qmr%iter
    relres = qmr%relres

    ! Cleanup
    call BLQMRDestroy(qmr)

end subroutine blqmr_solve_complex_multi