!=========================================================
!  
!=========================================================

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

!=========================================================
!  
!=========================================================

        subroutine sp_cc_x_mat(Ap,Ai,Ax,x,b)
        implicit none

        integer,dimension(:),intent(in)               :: Ai,Ap
        MTYPE(kind=Kdouble),dimension(:),intent(in)   :: Ax
        MTYPE(kind=Kdouble),dimension(:,:),intent(in) :: x
        MTYPE(kind=Kdouble),dimension(:,:),intent(out):: b
        integer             :: n,m,i,j,p

        n = size(Ap)-1
        m = size(b,2)

        b = 0.0_Kdouble
        do j = 1,n
          do p = Ap(j), Ap(j+1)-1
            i = Ai(p)
            b(i,:)=b(i,:)+Ax(p)*x(j,:)
          enddo
        enddo
        end subroutine sp_cc_x_mat
