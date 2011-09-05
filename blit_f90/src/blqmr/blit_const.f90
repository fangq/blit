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
!>  \brief Global File Names For All Modules
!==========================================================================

!--------------------------------------------------------------------------
!>\class blit_precision
!>\brief module to define global constants
!--------------------------------------------------------------------------

module blit_precision
save
    integer,parameter :: Kshort  = selected_int_kind(4)
    integer,parameter :: Kint    = selected_int_kind(8)
    integer,parameter :: Ksingle = selected_real_kind(6)
    integer,parameter :: Kdouble = selected_real_kind(15)
end module blit_precision
