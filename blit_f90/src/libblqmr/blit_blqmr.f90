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
                real(kind=Kdouble) :: qtol, droptol ! convergence tolerance
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
                real(kind=Kdouble) :: qtol, droptol ! convergence tolerance
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
use blit_blqmr_complex
implicit none

       integer, parameter :: n=5, nz=12
       type (BLQMRSolver) :: qmr
       integer :: Ap(n+1), Ai(nz)
       complex(kind=Kdouble)   :: Ax(nz), x(n), b(n)

       call BLQMRCreate(qmr,n,1,nz)

       Ap=(/0, 2, 5, 9, 10, 12/)+1
       Ai=(/0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4/)+1
       Ax=(/2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1./)
       b=(/8.0,  45.000,  -3.000,   3.000,  19.000/)
       b=dcmplx(real(b),real(b)*2.0)
       x=(/0., 0., 1., 0., 0./)

       qmr%maxit=1000
       qmr%qtol=1e-9_Kdouble
       qmr%dopcond=1
       qmr%droptol=0.2_Kdouble

       call BLQMRPrep(qmr, Ap, Ai, Ax, nz, x, b)
       call BLQMRSolve(qmr,Ap,Ai,Ax, nz, x, b)

       print *, qmr%iter, qmr%flag
       write (*,'(F8.3 F8.3)') x

       call BLQMRDestroy(qmr)

end program test_blit_blqmr

#endif
