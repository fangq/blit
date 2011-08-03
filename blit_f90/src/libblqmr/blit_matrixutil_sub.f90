
!=========================================================
!  
!=========================================================

        subroutine IdentityMatrix(n,A)
        implicit none
        integer :: n,i
        MTYPE(kind=Kdouble), dimension(n,n) :: A

        A=0.0_Kdouble
        forall (i = 1:n) A(i,i) = 1.0_Kdouble

        end subroutine IdentityMatrix

!=========================================================
!  
!=========================================================

        subroutine MatrixInversion(A, invA)
        implicit none

        MTYPE(kind=Kdouble),dimension(:,:),intent(in) :: A
        MTYPE(kind=Kdouble),dimension(:,:),intent(out):: invA
        integer :: n, info
        MTYPE(kind=Kdouble),dimension(size(A,1)) :: work
        integer,dimension(size(A,1)) :: ipiv

        if(size(A,1)/=size(A,2)) then
                stop
        endif
        n=size(A,1)
        invA=A
#if MTYPEID == MTYPEID_REAL
        call dgetrf(n, n, invA, n, ipiv, info)
#else
        call zgetrf(n, n, invA, n, ipiv, info)
#endif
        if(info/=0) then
                print *, 'error encontered when calling dgetrf in MatrixInversion, code:', info
		call PrintMatrix(A,'A = ')
                stop
        endif
#if MTYPEID == MTYPEID_REAL
        call dgetri(n, invA, n, ipiv, work, n, info)
#else
        call zgetri(n, invA, n, ipiv, work, n, info)
#endif
        if(info/=0) then
                print *, 'error encontered when calling dgetri in MatrixInversion, code:', info
		call PrintMatrix(A,'A = ')
                stop
        endif

        end subroutine MatrixInversion

!=========================================================
!  
!=========================================================

        subroutine QRDecomposition(A, Q, R, iseconomic)
        implicit none

        MTYPE(kind=Kdouble),dimension(:,:),intent(in)  :: A
        MTYPE(kind=Kdouble),dimension(:,:),intent(out) :: Q, R
        MTYPE(kind=Kdouble),dimension(size(A,1),size(A,2))  :: tmp
        MTYPE(kind=Kdouble),dimension(min(size(A,1),size(A,2)))  :: tau
        MTYPE(kind=Kdouble),dimension(max(size(A,1),1))  :: work

        integer :: m, n, info, len, i, j, iseconomic

        tmp=A
        m=size(A,1)
        n=size(A,2)
        len=max(m,1)

        R=0.0_Kdouble
#if MTYPEID == MTYPEID_REAL
        call dgeqrf(m, n, tmp, m, tau, work, len, info)
#else
        call zgeqrf(m, n, tmp, m, tau, work, len, info)
#endif
        if(info/=0) then
                print *, 'error encontered when calling dgeqrf in QRDecomposition, code:', info
                stop
        endif
        Q=tmp
        do i=1,n
           do j=i,n
             R(i,j)=tmp(i,j)
           enddo
        enddo

#if MTYPEID == MTYPEID_REAL
        if(iseconomic==0) then
             call dorgqr(m, n, min(m, n), Q, size(tmp,1), tau, work, len, info)
        else
             call dorgqr(m, m, min(m, n), Q, m, tau, work, len, info)
        endif
#else
        if(iseconomic==0) then
             call zungqr(m, n, min(m, n), Q, size(tmp,1), tau, work, len, info)
        else
             call zungqr(m, m, min(m, n), Q, m, tau, work, len, info)
        endif
#endif
        if(info/=0) then
                print *, 'error encontered when calling dorgqr in QRDecomposition, code:', info
                stop
        endif

        end subroutine QRDecomposition

!=========================================================
!  
!=========================================================

        subroutine QuasiQR(A, Q, R) ! economic form only
        implicit none

        MTYPE(kind=Kdouble),dimension(:,:),intent(in)  :: A
        MTYPE(kind=Kdouble),dimension(:,:),intent(out) :: R
        MTYPE(kind=Kdouble),dimension(size(A,1),size(A,2))  :: Q

        integer :: m, n, k,j

        m=size(A,1)
        n=size(A,2)
        R=0.0_Kdouble

        Q=A
        do k=1,n
#if MTYPEID == MTYPEID_REAL
            R(k,k)=dsqrt(sum(Q(:,k)*Q(:,k)))
#else
            R(k,k)=zsqrt(sum(Q(:,k)*Q(:,k)))
#endif
            Q(:,k)=Q(:,k)*(1.0_Kdouble/R(k,k))
            do j=k+1,n
                R(k,j)=sum(Q(:,k)*Q(:,j))
                Q(:,j)=Q(:,j)-R(k,j)*Q(:,k)
            enddo
        enddo

        end subroutine QuasiQR

!=========================================================
!  
!=========================================================

        subroutine PrintMatrix(A, info)
        implicit none
        integer :: i, j
        MTYPE(kind=Kdouble), dimension(:,:), intent(in) :: A
        character(len=*), optional :: info

        if(present(info)) then
                print *, info
        else
                print *, 'ans ='
        endif
        do i=1, size(A,1)
            do j=1, size(A,2)
#if MTYPEID == MTYPEID_REAL
                write (*,'(F8.3)',advance="no") A(i,j)
#else
                write (*,'(A,F8.3,F8.3,A,1X)',advance="no") '(',real(A(i,j)),imag(A(i,j)),')'
#endif
            enddo
            write (*,'(A)') ''
        enddo

        end subroutine PrintMatrix
