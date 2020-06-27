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
