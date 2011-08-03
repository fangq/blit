!==========================================================================
! Block QMR Solver - for real systems
!==========================================================================

#define MTYPEID_REAL    1
#define MTYPEID_COMPLEX 2

module blit_blqmr_real

        use blit_precision
        use blit_ilupcond
        use blit_sparseutil_real
        use blit_matrixutil_real

        implicit none

        private
        public :: BLQMRSolver, BLQMRCreate, BLQMRDestroy, BLQMRPrep, &
                  BLQMRSolve
        interface BLQMRCreate; module procedure BLQMROnCreate; end interface
        interface BLQMRDestroy; module procedure BLQMROnDestroy; end interface

        type BLQMRSolver
                integer :: n, nrhs, maxit, state, dopcond, flag, iter
                real(kind=Kdouble) :: qtol, droptol, res, relres ! convergence tolerance
        end type BLQMRSolver
save

#define MTYPE       real
        MTYPE(kind=Kdouble),dimension(:,:),allocatable   :: vt,alpha,theta,zeta,zetat,eta,tao,taot,x0
        MTYPE(kind=Kdouble),dimension(:,:,:),allocatable :: beta,Qa,Qc,v,omega,Qb,Qd,p
        type (ILUPcond) :: ilu ! private ILU preconditioner
#undef MTYPE

contains

#define MTYPE       real
#define MTYPEID     MTYPEID_REAL
#include "blit_blqmr_sub.f90"
#undef MTYPEID
#undef MTYPE

end module blit_blqmr_real

!==========================================================================
! Block QMR Solver - for complex systems
!==========================================================================

module blit_blqmr_complex

        use blit_precision
        use blit_ilupcond
        use blit_sparseutil_complex
        use blit_matrixutil_complex

        implicit none

        private
        public :: BLQMRSolver, BLQMRCreate, BLQMRDestroy, BLQMRPrep, &
                  BLQMRSolve
        interface BLQMRCreate; module procedure BLQMROnCreate; end interface
        interface BLQMRDestroy; module procedure BLQMROnDestroy; end interface


        type BLQMRSolver
                integer :: n, nrhs, maxit, state, dopcond, flag, iter
                real(kind=Kdouble) :: qtol, droptol, res, relres ! convergence tolerance
        end type BLQMRSolver
save

#define MTYPE       complex
        MTYPE(kind=Kdouble),dimension(:,:),allocatable   :: vt,alpha,theta,zeta,zetat,eta,tao,taot,x0
        MTYPE(kind=Kdouble),dimension(:,:,:),allocatable :: beta,Qa,Qc,v,omega,Qb,Qd,p
        type (ILUPcond) :: ilu ! private ILU preconditioner
#undef MTYPE

contains

#define MTYPE       complex
#define MTYPEID     MTYPEID_COMPLEX
#include "blit_blqmr_sub.f90"
#undef MTYPEID
#undef MTYPE

end module blit_blqmr_complex


!==========================================================================
! Regression test program
!==========================================================================

#ifdef BLIT_SELF_TEST

program test_blit_blqmr
use blit_precision
use blit_blqmr_real
implicit none

       integer, parameter :: n=5, nz=12
       type (BLQMRSolver) :: qmr
       integer :: Ap(n+1), Ai(nz)
       real(kind=Kdouble)   :: Ax(nz), x(n,2), b(n,2)

       call BLQMRCreate(qmr,n,2)

       Ap=(/0, 2, 5, 9, 10, 12/)+1
       Ai=(/0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4/)+1
       Ax=(/2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1./)
       b(:,1)=(/8.0,  45.000,  -3.000,   3.000,  19.000/)
       b(:,2)=(/18.0,  45.000,  -3.000,   3.000,  19.000/)
       x=0._Kdouble

       qmr%maxit=100
       qmr%qtol=1e-5_Kdouble
       qmr%dopcond=1
       qmr%droptol=0.001_Kdouble

       call BLQMRPrep(qmr, Ap, Ai, Ax, nz)
       call BLQMRSolve(qmr,Ap,Ai,Ax, nz, x, b)

       print *, qmr%iter, qmr%flag, qmr%res, qmr%relres
       write (*,'(F8.3)') x

       call BLQMRSolve(qmr,Ap,Ai,Ax, nz, x(:,1), b(:,2))

       print *, qmr%iter, qmr%flag, qmr%res, qmr%relres
       write (*,'(F8.3)') x(:,1)
       call BLQMRDestroy(qmr)

end program test_blit_blqmr

#endif
