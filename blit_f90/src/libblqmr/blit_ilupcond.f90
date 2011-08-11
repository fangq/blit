!==========================================================================
! Incomplete LU decomposition preconditioner (using UMFPACK)
!==========================================================================

module blit_ilupcond
use blit_precision
implicit none

        private
        public :: ILUPcond, ILUPcondCreate, ILUPcondDestroy, &
                  ILUPcondPrep, ILUPcondSolve

        integer,parameter :: UMFP_DROPTOL = 19

        type ILUPcond
                integer :: n, nz, status, iscomplex    ! Fortran-variables
                integer :: numeric, symbolic ! UMFPACK-variables
                !integer,dimension(:),allocatable             :: Ap,Ai
                !real(kind=Kdouble),dimension(:), allocatable :: Ax,Az
                real(kind=Kdouble),dimension(20)         :: control
        end type ILUPcond

contains

!=========================================================
!  
!=========================================================

        subroutine ILUPcondCreate(this,n,nz)
        implicit none
        type(ILUPcond), intent(inout) :: this
        integer :: n, nz

        this%n=n
        this%nz=nz
        this%numeric=-1
        this%status=-1
        this%iscomplex=0

        end subroutine ILUPcondCreate

!=========================================================
!  
!=========================================================

        subroutine ILUPcondDestroy(this)
        implicit none

        type(ILUPcond), intent(inout) :: this

        if(this%numeric>0) then
                if(this%iscomplex==0) then
                        call umf4fnum(this%numeric)
                else
                        call zumf4fnum(this%numeric)
                endif
        endif
        this%numeric=-1

        end subroutine ILUPcondDestroy

!=========================================================
!  
!=========================================================

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
        !this%control(1) = 2
        this%control(UMFP_DROPTOL) = droptol
        !call umf4pcon(this%control)

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

!=========================================================
!  
!=========================================================

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

end module blit_ilupcond

!==========================================================================
! Regression test program
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
