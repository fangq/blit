!!*************************************************************************
!!
!!  Blit - An open-source library for block iterative sparse linear solvers
!!
!!  Copyright 2011,2020 Qianqian Fang <q.fang at neu.edu>
!!
!!  URL: http://blit.sourceforge.net
!!
!!  Project maintainer: 
!!      Qianqian Fang, PhD
!!      Dept. of Bioengineering
!!      Northeastern University
!!      360 Huntington Ave, ISEC 206
!!      Boston, MA 02115, USA
!!
!!  License:
!!      BSD or LGPL or GPL, see LICENSE_*.txt for more details
!!
!!*************************************************************************

!==========================================================================
!>\brief Incomplete LU decomposition preconditioner (using UMFPACK)
!==========================================================================

!--------------------------------------------------------------------------
!>\class blit_ilupcond
!>\brief module for incomplete LU decomposition of a sparse matrix (preconditioner)
!--------------------------------------------------------------------------
module blit_ilupcond
use iso_c_binding, only: c_char,c_size_t,c_int,c_double,c_ptr,c_null_ptr
use blit_precision
implicit none

        private
        public :: ILUPcond, ILUPcondCreate, ILUPcondDestroy, &
                  ILUPcondPrep, ILUPcondSolve, ILUPcondSolveL, ILUPcondSolveU

        integer,parameter :: UMFP_DROPTOL = 19

        type, bind(c) :: ILUPcond
                integer(c_int) :: n, nz, status, iscomplex
                ! CRITICAL FIX: Use c_ptr for UMFPACK handles (64-bit pointers)
                type(c_ptr) :: numeric = c_null_ptr
                type(c_ptr) :: symbolic = c_null_ptr
                real(c_double),dimension(20) :: control
        end type ILUPcond

contains

!--------------------------------------------------------------------------
!> \fn ILUPcondCreate(this,n,nz)
!> \brief initialization of the ILU preconditioner
!--------------------------------------------------------------------------

        subroutine ILUPcondCreate(this,n,nz)
        implicit none
        type(ILUPcond), intent(inout) :: this
        integer :: n, nz

        this%n=n
        this%nz=nz
        this%numeric=c_null_ptr
        this%symbolic=c_null_ptr
        this%status=-1
        this%iscomplex=0

        end subroutine ILUPcondCreate

!--------------------------------------------------------------------------
!> \fn ILUPcondDestroy(this)
!> \brief destroy of the ILU preconditioner object
!--------------------------------------------------------------------------

        subroutine ILUPcondDestroy(this)
        use iso_c_binding, only: c_associated
        implicit none

        type(ILUPcond), intent(inout) :: this

        if(c_associated(this%numeric)) then
                if(this%iscomplex==0) then
                        call umf4fnum(this%numeric)
                else
                        call zumf4fnum(this%numeric)
                endif
        endif
        this%numeric=c_null_ptr

        end subroutine ILUPcondDestroy

!--------------------------------------------------------------------------
!> \fn ILUPcondPrep(this,Ap,Ai,Ax,droptol,Az)
!> \brief precondition the sparse left-hand-side matrix
!--------------------------------------------------------------------------

        subroutine ILUPcondPrep(this,Ap,Ai,Ax,droptol,Az)
        implicit none

        type(ILUPcond), intent(inout) :: this
        integer :: Ap(this%n+1), Ai(this%nz)
        real(kind=Kdouble)  :: droptol, Ax(this%nz)
        real(kind=Kdouble),dimension(90) :: info
        real(kind=Kdouble),intent(in),optional  :: Az(this%nz)

        if(.not. present(Az)) then
                call umf4def(this%control)
        else
                this%iscomplex=1
                call zumf4def(this%control)
        endif
        this%control(UMFP_DROPTOL) = droptol

        if(.not. present(Az)) then
                call umf4sym(this%n, this%n, Ap-1, Ai-1, Ax, this%symbolic, this%control, info)
        else
                call zumf4sym(this%n, this%n, Ap-1, Ai-1, Ax, Az, this%symbolic, this%control, info)
        endif
        if (info(1) < 0) then
            print *, "Error occurred in umf4sym: ", info(1)
            stop
        endif

        if(.not. present(Az)) then
                call umf4num(Ap-1, Ai-1, Ax, this%symbolic, this%numeric, this%control, info)
        else
                call zumf4num(Ap-1, Ai-1, Ax, Az, this%symbolic, this%numeric, this%control, info)
        endif
        if (info(1) < 0) then
            print *, "Error occurred in umf4num: ", info(1)
            stop
        endif

        if(.not. present(Az)) then
                call umf4fsym(this%symbolic)
        else
                call zumf4fsym(this%symbolic)
        endif

        end subroutine ILUPcondPrep

!--------------------------------------------------------------------------
!> \fn ILUPcondSolve(this,Ap,Ai,Ax,rows,cols,x,b,Az,xz,bz)
!> \brief solving x in A*x=b using the preconditioned A matrix
!--------------------------------------------------------------------------

        subroutine ILUPcondSolve(this,Ap,Ai,Ax,rows,cols,x,b,Az,xz,bz)
        implicit none
        type(ILUPcond), intent(inout) :: this
        integer :: i, sys, rows, cols, Ap(this%n+1), Ai(this%nz)
        real(kind=Kdouble),intent(out) :: x(rows, cols)
        real(kind=Kdouble),intent(in)  :: b(rows, cols), Ax(this%nz)
        real(kind=Kdouble),dimension(90)   :: info
        real(kind=Kdouble),optional,intent(in)  :: Az(this%nz), bz(rows, cols)
        real(kind=Kdouble),optional,intent(out) :: xz(rows, cols)

        sys = 0

        do i=1, cols
            if(.not. present(bz)) then
                call umf4solr(sys, Ap-1, Ai-1, Ax, x(:,i), b(:,i), this%numeric, this%control, info)
            else
                call zumf4solr(sys, Ap-1, Ai-1, Ax, Az, x(:,i), xz(:,i), b(:,i), bz(:,i), this%numeric, this%control, info)
            endif
        enddo
        if (info(1) < 0) then
            print *, "Error occurred in umf4solr: ", info(1)
            stop
        endif

        end subroutine ILUPcondSolve

!--------------------------------------------------------------------------
!> \fn ILUPcondSolveL(this,Ap,Ai,Ax,rows,cols,x,b,Az,xz,bz)
!> \brief Solve L*x = Pb (lower triangular solve with row permutation)
!>
!> UMFPACK factorizes as P*A*Q = L*U where P,Q are permutation matrices.
!> sys=3 solves: L*x = P*b (applies row permutation, then L solve)
!--------------------------------------------------------------------------

        subroutine ILUPcondSolveL(this,Ap,Ai,Ax,rows,cols,x,b,Az,xz,bz)
        implicit none
        type(ILUPcond), intent(inout) :: this
        integer :: i, sys, rows, cols, Ap(this%n+1), Ai(this%nz)
        real(kind=Kdouble),intent(out) :: x(rows, cols)
        real(kind=Kdouble),intent(in)  :: b(rows, cols), Ax(this%nz)
        real(kind=Kdouble),dimension(90)   :: info
        real(kind=Kdouble),optional,intent(in)  :: Az(this%nz), bz(rows, cols)
        real(kind=Kdouble),optional,intent(out) :: xz(rows, cols)

        ! UMFPACK sys values for triangular solves:
        !   sys=3: solve P'Lx = b  =>  Lx = Pb  (L solve with row perm)
        !   sys=4: solve L'Px = b  (L' solve)
        !   sys=5: solve L.'Px = b (L.' solve, for complex)
        sys = 3  ! Solve Lx = Pb
        
        do i=1, cols
            if(.not. present(bz)) then
                call umf4solr(sys, Ap-1, Ai-1, Ax, x(:,i), b(:,i), this%numeric, this%control, info)
            else
                call zumf4solr(sys, Ap-1, Ai-1, Ax, Az, x(:,i), xz(:,i), b(:,i), bz(:,i), this%numeric, this%control, info)
            endif
        enddo
        if (info(1) < 0) then
            print *, "Error in ILUPcondSolveL (sys=3): ", info(1)
        endif
        
        end subroutine ILUPcondSolveL

!--------------------------------------------------------------------------
!> \fn ILUPcondSolveU(this,Ap,Ai,Ax,rows,cols,x,b,Az,xz,bz)
!> \brief Solve U*Q'*x = b (upper triangular solve with column permutation)
!>
!> UMFPACK factorizes as P*A*Q = L*U where P,Q are permutation matrices.
!> sys=6 solves: U*Q'*x = b (U solve, then applies column permutation)
!--------------------------------------------------------------------------

        subroutine ILUPcondSolveU(this,Ap,Ai,Ax,rows,cols,x,b,Az,xz,bz)
        implicit none
        type(ILUPcond), intent(inout) :: this
        integer :: i, sys, rows, cols, Ap(this%n+1), Ai(this%nz)
        real(kind=Kdouble),intent(out) :: x(rows, cols)
        real(kind=Kdouble),intent(in)  :: b(rows, cols), Ax(this%nz)
        real(kind=Kdouble),dimension(90)   :: info
        real(kind=Kdouble),optional,intent(in)  :: Az(this%nz), bz(rows, cols)
        real(kind=Kdouble),optional,intent(out) :: xz(rows, cols)

        ! UMFPACK sys values for triangular solves:
        !   sys=6: solve UQ'x = b  (U solve with col perm)
        !   sys=7: solve QU'x = b  (U' solve)
        !   sys=8: solve QU.'x = b (U.' solve, for complex)
        sys = 6  ! Solve UQ'x = b
        
        do i=1, cols
            if(.not. present(bz)) then
                call umf4solr(sys, Ap-1, Ai-1, Ax, x(:,i), b(:,i), this%numeric, this%control, info)
            else
                call zumf4solr(sys, Ap-1, Ai-1, Ax, Az, x(:,i), xz(:,i), b(:,i), bz(:,i), this%numeric, this%control, info)
            endif
        enddo
        if (info(1) < 0) then
            print *, "Error in ILUPcondSolveU (sys=6): ", info(1)
        endif
        
        end subroutine ILUPcondSolveU

end module blit_ilupcond

!==========================================================================
!!Regression test program
!==========================================================================

#ifdef BLIT_SELF_TEST

program test_blit_ilupcond
use blit_ilupcond
use blit_precision
implicit none

       integer, parameter :: n=5, nz=12
       type (ILUPcond) :: ilu
       integer :: Ap(n+1), Ai(nz)
       real(kind=Kdouble) :: Ax(nz), Az(nz), x(n), b(n), xz(n), bz(n)

       ilu%iscomplex=1
       call ILUPcondCreate(ilu,n,nz)

       Ap=(/0, 2, 5, 9, 10, 12/)+1
       Ai=(/0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4/)+1
       Ax=(/2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1./)
       Az=(/2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1./)
       b=(/8.000,  45.000,  -3.000,   3.000,  19.000/)
       bz=0.0_Kdouble

       call ILUPcondPrep(ilu,Ap,Ai,Ax,0.2_Kdouble, Az)
       call ILUPcondSolve(ilu,Ap,Ai,Ax,n,1,x,b,Az,xz,bz)
       write (*,'(5F8.3)') x
       write (*,'(5F8.3)') xz

       call ILUPcondDestroy(ilu)

end program test_blit_ilupcond

#endif