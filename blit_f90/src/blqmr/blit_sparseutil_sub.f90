!!*************************************************************************
!!
!!  Blit - An open-source library for block iterative sparse linear solvers
!!
!!  Copyright 2011, Qianqian Fang <fangq at nmr.mgh.harvard.edu>
!!
!!  URL: http://blit.sourceforge.net
!!
!!  Project maintainer: 
!!      Qianqian Fang, PhD
!!      Martinos Center for Biomedical Imaging
!!      Massachusetts General Hospital
!!      Harvard Medical School
!!      149 13th Street, Charlestown, MA 02129
!!
!!  License:
!!      BSD or LGPL or GPL, see LICENSE_*.txt for more details
!!
!!*************************************************************************

!--------------------------------------------------------------------------
!> \fn sp_cc_x_vec(Ap,Ai,Ax,x,b)
!> \brief multiply a column-compressed sparse matrix to a vector
!--------------------------------------------------------------------------

        subroutine sp_cc_x_vec(Ap,Ai,Ax,x,b)
        implicit none

        integer,dimension(:),intent(in)             :: Ai,Ap
        MTYPE(kind=Kdouble),dimension(:),intent(in) :: Ax,x
        MTYPE(kind=Kdouble),dimension(:),intent(out):: b
        integer             :: n,i,p,j

        n = size(Ap)-1
        b = 0.0_Kdouble
        do j = 1,n
          do p = Ap(j), Ap(j+1)-1
            i = Ai(p)
            b(i) = b(i) + Ax(p) * x(j)
          enddo
        enddo
        end subroutine sp_cc_x_vec

!--------------------------------------------------------------------------
!> \fn sp_cc_x_mat(Ap,Ai,Ax,x,b)
!> \brief multiply a column-compressed sparse matrix to a matrix
!--------------------------------------------------------------------------

        subroutine sp_cc_x_mat(Ap,Ai,Ax,x,b)
        implicit none

        integer,dimension(:),intent(in)               :: Ai,Ap
        MTYPE(kind=Kdouble),dimension(:),intent(in)   :: Ax
        MTYPE(kind=Kdouble),dimension(:,:),intent(in) :: x
        MTYPE(kind=Kdouble),dimension(:,:),intent(out):: b
        integer             :: n,i,j,p

        n = size(Ap)-1
        b = 0.0_Kdouble
        do j = 1,n
          do p = Ap(j), Ap(j+1)-1
            i = Ai(p)
            b(i,:)=b(i,:)+Ax(p)*x(j,:)
          enddo
        enddo
        end subroutine sp_cc_x_mat
