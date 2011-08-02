!==========================================================================
!   Global File Names For All Modules
!==========================================================================

module blit_precision
save
    integer,parameter :: Kshort  = selected_int_kind(4)
    integer,parameter :: Kint    = selected_int_kind(8)
    integer,parameter :: Ksingle = selected_real_kind(6)
    integer,parameter :: Kdouble = selected_real_kind(15)
end module blit_precision
