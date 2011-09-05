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
!>\brief Full Matrix Utilities 
!==========================================================================

!--------------------------------------------------------------------------
!>\class blit_matrixutil_real
!>\brief Full Matrix Utilities - for real matrices
!--------------------------------------------------------------------------

#define MTYPEID_REAL    1
#define MTYPEID_COMPLEX 2

module blit_matrixutil_real
use blit_precision
implicit none

        private
        public :: eye, inv, qr, qqr, disp
        interface eye; module procedure IdentityMatrix; end interface
        interface inv; module procedure MatrixInversion; end interface
        interface qr; module procedure QRDecomposition; end interface
        interface qqr; module procedure QuasiQR; end interface
        interface disp; module procedure PrintMatrix; end interface

contains

#define MTYPE       real
#define MTYPEID     MTYPEID_REAL
#include "blit_matrixutil_sub.f90"
#undef MTYPEID
#undef MTYPE

end module blit_matrixutil_real

!--------------------------------------------------------------------------
!>\class blit_matrixutil_complex
!>\brief Full Matrix Utilities - for complex matrices
!--------------------------------------------------------------------------

module blit_matrixutil_complex
use blit_precision
implicit none

        private
        public :: eye, inv, qr, qqr, disp
        interface eye; module procedure IdentityMatrix; end interface
        interface inv; module procedure MatrixInversion; end interface
        interface qr; module procedure QRDecomposition; end interface
        interface qqr; module procedure QuasiQR; end interface
        interface disp; module procedure PrintMatrix; end interface

contains

#define MTYPE       complex
#define MTYPEID     MTYPEID_COMPLEX
#include "blit_matrixutil_sub.f90"
#undef MTYPEID
#undef MTYPE

end module blit_matrixutil_complex


!==========================================================================
!!Regression test program
!==========================================================================

#ifdef BLIT_SELF_TEST

program test_blit_matrixutil
use blit_matrixutil_real
use blit_matrixutil_complex
use blit_precision

implicit none

       integer, parameter :: n=5
       real(kind=Kdouble), dimension(n,n) :: A, invA, Q, R
       real(kind=Kdouble), dimension(n,n) :: Q2
       real(kind=Kdouble) :: R2(n,3)
       complex(kind=Kdouble), dimension(n,n) :: Ac
       integer :: i

       call eye(n,A)
       call disp(A,'I(5) =')

       forall (i=1:n) A(i,i)=i
       A(2,1)=10._Kdouble
       A(2,3)=-5._Kdouble
       A(3,4)=1._Kdouble
       A(5,1)=9._Kdouble
       
       Ac=cmplx(A,A*0.5_Kdouble)
       call disp(Ac, 'A_complex =')

       call inv(A,invA)
       call disp(invA, 'inv(A) =')
       
       A=A+invA
       call disp(A, 'A+inv(A) =')

       call qr(A,Q,R,0)
       call disp(Q, 'Q =')
       call disp(R, 'R =')
       
       A=matmul(Q,R)
       call disp(A, 'Q*R =')

       call qr(A(:,1:3),Q2,R2, 1)
       call disp(Q2, 'Q2 =')
       call disp(R2, 'R2 =')
       
       call disp(matmul(transpose(Q2),Q2), 'Q2''*Q2 =')

end program test_blit_matrixutil

#endif
