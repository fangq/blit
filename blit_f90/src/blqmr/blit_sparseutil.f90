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

!==========================================================================
!>\brief Sparse Matrix Utilities
!==========================================================================

!--------------------------------------------------------------------------
!>\class blit_sparseutil_real
!>\brief Sparse Matrix Utilities - for real sparse matrices
!--------------------------------------------------------------------------

module blit_sparseutil_real
use blit_precision
implicit none

        private
        public :: spmulvec, spmulmat
        interface spmulvec; module procedure sp_cc_x_vec; end interface
        interface spmulmat; module procedure sp_cc_x_mat; end interface

contains

#define MTYPE real
#include "blit_sparseutil_sub.f90"
#undef MTYPE

end module blit_sparseutil_real

!--------------------------------------------------------------------------
!>\class blit_sparseutil_complex
!>\brief Sparse Matrix Utilities - for complex sparse matrices
!--------------------------------------------------------------------------

module blit_sparseutil_complex
use blit_precision
implicit none

        private
        public :: spmulvec, spmulmat
        interface spmulvec; module procedure sp_cc_x_vec; end interface
        interface spmulmat; module procedure sp_cc_x_mat; end interface

contains

#define MTYPE complex
#include "blit_sparseutil_sub.f90"
#undef MTYPE

end module blit_sparseutil_complex

!==========================================================================
!!Regression test program
!==========================================================================

#ifdef BLIT_SELF_TEST

program test_blit_sparseutil
use blit_sparseutil_real
use blit_sparseutil_complex
use blit_precision

implicit none

       integer, parameter :: n=5, nz=12
       integer :: Ap(n+1), Ai(nz)
       complex(kind=Kdouble) :: Ax(nz), b(n), x(n)

       Ap=(/0, 2, 5, 9, 10, 12/)+1
       Ai=(/0, 1, 0,2, 4, 1, 2, 3,4, 2, 1, 4/)+1
       Ax=(/2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1./)
       b=(/1,2,3,4,5/)

       call spmulvec(Ap,Ai,Ax,b,x)
       write (*,'(5F8.3)') x

end program test_blit_sparseutil

#endif
