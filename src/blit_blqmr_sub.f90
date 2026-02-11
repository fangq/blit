!!*************************************************************************
!!
!!  Blit - An open-source library for block iterative sparse linear solvers
!!
!!  Copyright 2011,2020 Qianqian Fang <q.fang at neu.edu>
!!
!!  URL: http://blit.sourceforge.net
!!
!!  License:
!!      BSD or LGPL or GPL, see LICENSE_*.txt for more details
!!
!!*************************************************************************

!--------------------------------------------------------------------------
!> \fn BLQMROnCreate(this, n)
!> \brief initialize a block-QMR solver object for an n-by-n sparse system
!--------------------------------------------------------------------------

        subroutine BLQMROnCreate(this, n)
        use iso_c_binding, only: c_null_ptr
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer :: n

        this%n=n
        this%nrhs=1
        this%qtol=1e-6_Kdouble
        this%droptol=0.001_Kdouble
        this%maxit=this%n
        this%state=0
        this%pcond_type=1
#ifdef _OPENMP
        this%nblock=1          ! Default: 1 RHS per OMP thread (best parallel scaling)
#else
        this%nblock=0          ! Default: all RHS in one block (no splitting)
#endif
        this%isquasires=1
        this%debug=0
        this%flag=-1
        this%res=1e100_Kdouble
        this%relres=1e100_Kdouble 

        this%ilu%n=n
        this%ilu%numeric=c_null_ptr
        this%ilu%symbolic=c_null_ptr
        this%ilu%nz=0

        end subroutine BLQMROnCreate

!--------------------------------------------------------------------------
!> \fn BLQMROnDestroy(this, isresize)
!> \brief desctroy or resize a BLQMR solver object
!--------------------------------------------------------------------------

        subroutine BLQMROnDestroy(this, isresize)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer, optional :: isresize

        if(.not. present(isresize)) &
                call ILUPcondDestroy(this%ilu)
        if (allocated(jacobi_sqrt_diag)) &
                deallocate(jacobi_sqrt_diag)

        this%state=-1;

        end subroutine BLQMROnDestroy

!--------------------------------------------------------------------------
!> \fn BLQMRPrep(this, Ap, Ai, Ax, nnz)
!> \brief create preconditioner and get ready for solving a block system
!--------------------------------------------------------------------------

        subroutine BLQMRPrep(this, Ap, Ai, Ax, nnz)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer, intent(inout) :: nnz, Ap(this%n+1), Ai(nnz)
        MTYPE(kind=Kdouble), intent(inout) :: Ax(nnz)
        integer :: i, j
        MTYPE(kind=Kdouble) :: diag_val

        if(this%pcond_type > 0) then
            select case(this%pcond_type)
            
            case(1, 2)  ! ILU-left or ILU-split
                call ILUPcondCreate(this%ilu, this%n, nnz)
#if MTYPEID == MTYPEID_COMPLEX
                this%ilu%iscomplex = 1
                call ILUPcondPrep(this%ilu, Ap, Ai, real(Ax), this%droptol, aimag(Ax))
#else
                call ILUPcondPrep(this%ilu, Ap, Ai, Ax, this%droptol)
#endif

            case(3)  ! Jacobi-split
                if (allocated(jacobi_sqrt_diag)) deallocate(jacobi_sqrt_diag)
                allocate(jacobi_sqrt_diag(this%n))
                
                do i = 1, this%n
                    diag_val = 0.0_Kdouble
                    do j = Ap(i), Ap(i+1)-1
                        if (Ai(j) == i) then
                            diag_val = Ax(j)
                            exit
                        endif
                    enddo
                    if (abs(diag_val) < 1.0d-14) diag_val = 1.0_Kdouble
                    jacobi_sqrt_diag(i) = sqrt(diag_val)
                enddo
                
            case default
                this%pcond_type = 0
                
            end select
        endif

        this%state = 1
        this%iter = -1
        this%flag = -1

        end subroutine BLQMRPrep

!--------------------------------------------------------------------------
!> \fn BLQMRSolveOMP(this, Ap, Ai, Ax, nnz, x, b, nrhs)
!> \brief Solve multiple RHS in parallel using OpenMP block decomposition.
!>
!> When nblock < nrhs, partitions the RHS columns into chunks of size
!> nblock and solves each chunk with BLQMRSolve in parallel via OpenMP.
!> The last chunk handles the remainder columns.
!>
!> When nblock >= nrhs or nblock == 0, falls back to a single BLQMRSolve.
!>
!> The preconditioner is prepared once and shared read-only across threads:
!> - ILU: UMFPACK numeric handle is only read by umf4solr
!> - Jacobi: jacobi_sqrt_diag is module-level and read-only during solve
!--------------------------------------------------------------------------

        subroutine BLQMRSolveOMP(this, Ap, Ai, Ax, nnz, x, b, nrhs)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer, intent(in) :: nrhs
        integer :: Ap(this%n+1), nnz, Ai(nnz)
        MTYPE(kind=Kdouble) :: Ax(nnz), b(this%n, nrhs), x(this%n, nrhs)

        integer :: nb, nchunks, ichunk, col_start, col_end, blk_nrhs
        integer :: max_flag, max_iter, nn, nthreads, tid
        integer :: saved_blas_threads
        real(kind=Kdouble) :: max_relres, max_res
        type(BLQMRSolver), allocatable :: qmr_pool(:)
        integer :: omp_get_max_threads, omp_get_thread_num

        nn = this%n

        ! Determine effective block size
        nb = this%nblock
        if (nb <= 0 .or. nb >= nrhs) then
            ! No splitting needed - solve all RHS at once
            call BLQMRSolve(this, Ap, Ai, Ax, nnz, x, b, nrhs)
            return
        endif

        ! Prepare preconditioner ONCE on the parent solver
        if(this%state==0) call BLQMRPrep(this, Ap, Ai, Ax, nnz)

        ! Compute number of chunks (ceiling division)
        nchunks = (nrhs + nb - 1) / nb

        ! Allocate per-thread solver pool (pre-initialized outside parallel region)
        nthreads = 1
        !$ nthreads = omp_get_max_threads()
        allocate(qmr_pool(0:nthreads-1))

        do tid = 0, nthreads-1
            qmr_pool(tid)%n          = nn
            qmr_pool(tid)%nrhs       = 1
            qmr_pool(tid)%maxit      = this%maxit
            qmr_pool(tid)%qtol       = this%qtol
            qmr_pool(tid)%droptol    = this%droptol
            qmr_pool(tid)%pcond_type = this%pcond_type
            qmr_pool(tid)%isquasires = this%isquasires
            qmr_pool(tid)%debug      = this%debug
            qmr_pool(tid)%nblock     = 0
            qmr_pool(tid)%flag       = -1
            qmr_pool(tid)%res        = 1e100_Kdouble
            qmr_pool(tid)%relres     = 1e100_Kdouble
            ! Share the parent's preconditioner (read-only during solve)
            ! - ILU: numeric/symbolic handles are only read by umf4solr
            ! - Jacobi: jacobi_sqrt_diag is module-level and read-only
            qmr_pool(tid)%ilu        = this%ilu
            qmr_pool(tid)%state      = 1  ! Mark as prepared (skip BLQMRPrep)
        enddo

        ! Initialize worst-case convergence tracking
        max_flag = 0
        max_iter = 0
        max_relres = 0.0_Kdouble
        max_res = 0.0_Kdouble

        ! Pin BLAS to 1 thread to avoid oversubscription inside OMP parallel region
        ! Save current setting, set to 1, restore after parallel region
        saved_blas_threads = 1
        call blit_get_blas_threads(saved_blas_threads)
        call blit_set_blas_threads(1)

        !$omp parallel do default(none) &
        !$omp shared(nchunks, nb, nrhs, nn, Ap, Ai, Ax, nnz, x, b, qmr_pool) &
        !$omp private(ichunk, col_start, col_end, blk_nrhs, tid) &
        !$omp reduction(max: max_flag, max_iter, max_relres, max_res) &
        !$omp schedule(dynamic)
        do ichunk = 1, nchunks
            tid = 0
            !$ tid = omp_get_thread_num()

            col_start = (ichunk - 1) * nb + 1
            col_end   = min(ichunk * nb, nrhs)
            blk_nrhs  = col_end - col_start + 1

            qmr_pool(tid)%nrhs   = blk_nrhs
            qmr_pool(tid)%flag   = -1
            qmr_pool(tid)%res    = 1e100_Kdouble
            qmr_pool(tid)%relres = 1e100_Kdouble

            ! Solve this chunk
            call BLQMRSolve(qmr_pool(tid), Ap, Ai, Ax, nnz, &
                            x(:, col_start:col_end), &
                            b(:, col_start:col_end), blk_nrhs)

            ! Track worst-case convergence info
            max_flag   = max(max_flag, qmr_pool(tid)%flag)
            max_iter   = max(max_iter, qmr_pool(tid)%iter)
            max_relres = max(max_relres, qmr_pool(tid)%relres)
            max_res    = max(max_res, qmr_pool(tid)%res)
        enddo
        !$omp end parallel do

        ! Restore BLAS thread count
        call blit_set_blas_threads(saved_blas_threads)

        ! Cleanup - do NOT call BLQMRDestroy since pool shares parent's ILU
        deallocate(qmr_pool)

        ! Store aggregate convergence info
        this%flag   = max_flag
        this%iter   = max_iter
        this%relres = max_relres
        this%res    = max_res
        this%nrhs   = nrhs

        end subroutine BLQMRSolveOMP

!--------------------------------------------------------------------------
!> \fn BLQMRSolve(this, Ap, Ai, Ax, nnz, x, b)
!> \brief solving a real or complex system using BLQMR algorithm
!--------------------------------------------------------------------------

        subroutine BLQMRSolve(this, Ap, Ai, Ax, nnz, x, b, nrhs)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer :: k,m,nn,t3,t3p,t3n,t3nn, Ap(this%n+1), nnz, Ai(nnz), nrhs
        integer, parameter :: DEBUG_RES=1
        real(kind=Kdouble) :: Qres, Qres1, Qres0
        MTYPE(kind=Kdouble) :: Ax(nnz), b(this%n,nrhs), x(this%n,nrhs)
        MTYPE(kind=Kdouble),dimension(nrhs,nrhs) :: tmp0,tmp1,tmp2
        MTYPE(kind=Kdouble),dimension(nrhs*2,nrhs) :: ZZ,zetafull
        MTYPE(kind=Kdouble),dimension(nrhs*2,nrhs*2) :: QQ
        MTYPE(kind=Kdouble),dimension(this%n,nrhs) :: tmp, tmp2_vec, omegat
        MTYPE(kind=Kdouble),dimension(nrhs,nrhs)  :: alpha,theta,zeta,zetat,eta,tau,taut
        MTYPE(kind=Kdouble),dimension(nrhs,nrhs,3):: beta,Qa,Qb,Qc,Qd,omega
        MTYPE(kind=Kdouble),dimension(this%n,nrhs)  :: vt
        MTYPE(kind=Kdouble),dimension(this%n,nrhs,3)  :: v,p
        MTYPE(kind=Kdouble) :: one, zero, negone
#if MTYPEID == MTYPEID_COMPLEX
        real(kind=Kdouble),dimension(this%n,nrhs,2) :: rtmp
#endif

        this%nrhs = nrhs
        m = this%nrhs
        nn = this%n
        one = 1.0_Kdouble
        zero = 0.0_Kdouble
        negone = -1.0_Kdouble

        if(this%state==0) call BLQMRPrep(this, Ap, Ai, Ax, nnz)

        ! Initialize all arrays
        v=0.0_Kdouble; vt=0.0_Kdouble; p=0.0_Kdouble
        alpha=0.0_Kdouble; theta=0.0_Kdouble; zeta=0.0_Kdouble
        zetat=0.0_Kdouble; eta=0.0_Kdouble; tau=0.0_Kdouble; taut=0.0_Kdouble
        beta=0.0_Kdouble; Qa=0.0_Kdouble; Qb=0.0_Kdouble
        Qc=0.0_Kdouble; Qd=0.0_Kdouble; omega=0.0_Kdouble

        t3=modulo(0,3)+1; t3n=modulo(-1,3)+1
        call eye(m, Qa(:,:,t3))
        Qd(:,:,t3n) = Qa(:,:,t3)
        Qd(:,:,t3) = Qa(:,:,t3)

        this%relres = 1._Kdouble
        Qres1 = -1._Kdouble
        
        ! Compute initial residual: vt = b - A*x
        call spmulmat(Ap, Ai, Ax, x, tmp)
        vt = b - tmp

        ! Apply preconditioner to initial residual
        call apply_precond_residual()
        if(this%flag == 2) return

        t3p = modulo(1,3)+1
        call qqr(vt, v(:,:,t3p), beta(:,:,t3p))
        call compute_omega(v(:,:,t3p), omega(:,:,t3p))
        taut = matmul(omega(:,:,t3p), beta(:,:,t3p))

        if(this%isquasires > 0) then
            Qres0 = compute_qres_mm(taut)
        else
            call compute_omegat(v(:,:,t3p), omega(:,:,t3p), omegat)
            Qres0 = compute_qres_nm(vt)
        endif

        if (Qres0 < this%qtol .or. Qres0 < 1.0d-14) then
            this%flag = 0; this%iter = 0
            this%relres = 0.0_Kdouble; this%res = Qres0
            return
        endif

        !======================================================================
        ! Main iteration loop
        !======================================================================
        this%flag = 1
        do k = 1, this%maxit
            t3 = modulo(k,3)+1
            t3p = modulo(k+1,3)+1
            t3n = modulo(k-1,3)+1
            t3nn = modulo(k-2,3)+1

            ! Apply preconditioned matrix-vector product
            call apply_precond_matvec(v(:,:,t3), vt)
            
            ! vt = vt - v(:,:,t3n) * beta(:,:,t3)^T
            call gemm_nnt(v(:,:,t3n), beta(:,:,t3), vt, negone, one)

            ! alpha = v(:,:,t3)^T * vt
            call gemm_tn(v(:,:,t3), vt, alpha)

            ! vt = vt - v(:,:,t3) * alpha
            call gemm_nn(v(:,:,t3), alpha, vt, negone, one)

            call qqr(vt, v(:,:,t3p), beta(:,:,t3p))
            call compute_omega(v(:,:,t3p), omega(:,:,t3p))

            ! Small matrix operations (m x m)
            tmp0 = matmul(omega(:,:,t3n), transpose(beta(:,:,t3)))
            theta = matmul(Qb(:,:,t3nn), tmp0)
            tmp1 = matmul(Qd(:,:,t3nn), tmp0)
            tmp2 = matmul(omega(:,:,t3), alpha)
            eta = matmul(Qa(:,:,t3n), tmp1) + matmul(Qb(:,:,t3n), tmp2)
            zetat = matmul(Qc(:,:,t3n), tmp1) + matmul(Qd(:,:,t3n), tmp2)

            ZZ(1:m, 1:m) = zetat
            ZZ(m+1:2*m, :) = matmul(omega(:,:,t3p), beta(:,:,t3p))

            call qr(ZZ, QQ, zetafull, 1)
            zeta = zetafull(1:m, 1:m)
#if MTYPEID == MTYPEID_REAL
            QQ = transpose(QQ)
#else
            QQ = conjg(transpose(QQ))
#endif

            Qa(:,:,t3) = QQ(1:m, 1:m)
            Qb(:,:,t3) = QQ(1:m, 1+m:2*m)
            Qc(:,:,t3) = QQ(1+m:2*m, 1:m)
            Qd(:,:,t3) = QQ(1+m:2*m, 1+m:2*m)

            call inv(zeta, tmp0)

            tmp = v(:,:,t3)
            call gemm_nn(p(:,:,t3n), eta, tmp, negone, one)
            call gemm_nn(p(:,:,t3nn), theta, tmp, negone, one)
            call gemm_nn(tmp, tmp0, p(:,:,t3), one, zero)

            tau = matmul(Qa(:,:,t3), taut)
            
            ! x = x + p(:,:,t3) * tau
            call gemm_nn(p(:,:,t3), tau, x, one, one)

            taut = matmul(Qc(:,:,t3), taut)

            if(this%isquasires > 0) then
                Qres = compute_qres_mm(taut)
            else
                call update_omegat()
                call gemm_nn(omegat, taut, tmp, one, zero)
#if MTYPEID == MTYPEID_COMPLEX
                tmp = conjg(tmp)
#endif
                Qres = compute_qres_nm(tmp)
            endif
            
            this%res = Qres
            this%iter = k

            if (isnan(Qres)) then
                this%relres = abs(Qres1) / max(abs(Qres0), 1.0d-300)
                if (Qres1 < this%qtol * Qres0) this%flag = 0
                exit
            endif

            this%relres = abs(Qres) / max(abs(Qres0), 1.0d-300)

            if(iand(this%debug, DEBUG_RES) > 0) &
                write(*,'(A,I4,A,E16.8,A,E16.8)') 'Iteration [',k,'] MaxResidual=', Qres, ', Relative=', this%relres

            if(k > 1 .and. abs(Qres-Qres1) < epsilon(Qres)) then
                this%flag = 3
                exit
            endif

            Qres1 = Qres

            if (Qres < 1.0d-14 .or. this%relres < this%qtol) then
                this%flag = 0
                exit
            endif
        enddo

        ! Solution recovery for split preconditioning
        call recover_solution()

        contains

        !------------------------------------------------------------------
        ! BLAS wrappers - internal procedures have access to nn, m
        !------------------------------------------------------------------
        
        subroutine gemm_nn(A, B, C, alpha_val, beta_val)
        ! C = alpha * A * B + beta * C
        ! A is (nn x m), B is (m x m), C is (nn x m)
        MTYPE(kind=Kdouble), intent(in) :: A(nn,m), B(m,m)
        MTYPE(kind=Kdouble), intent(inout) :: C(nn,m)
        MTYPE(kind=Kdouble), intent(in) :: alpha_val, beta_val
#if MTYPEID == MTYPEID_REAL
        call dgemm('N', 'N', nn, m, m, alpha_val, A, nn, B, m, beta_val, C, nn)
#else
        call zgemm('N', 'N', nn, m, m, alpha_val, A, nn, B, m, beta_val, C, nn)
#endif
        end subroutine gemm_nn

        subroutine gemm_nnt(A, B, C, alpha_val, beta_val)
        ! C = alpha * A * B^T + beta * C
        ! A is (nn x m), B is (m x m), C is (nn x m)
        MTYPE(kind=Kdouble), intent(in) :: A(nn,m), B(m,m)
        MTYPE(kind=Kdouble), intent(inout) :: C(nn,m)
        MTYPE(kind=Kdouble), intent(in) :: alpha_val, beta_val
#if MTYPEID == MTYPEID_REAL
        call dgemm('N', 'T', nn, m, m, alpha_val, A, nn, B, m, beta_val, C, nn)
#else
        call zgemm('N', 'T', nn, m, m, alpha_val, A, nn, B, m, beta_val, C, nn)
#endif
        end subroutine gemm_nnt

        subroutine gemm_tn(A, B, C)
        ! C = A^T * B
        ! A is (nn x m), B is (nn x m), C is (m x m)
        MTYPE(kind=Kdouble), intent(in) :: A(nn,m), B(nn,m)
        MTYPE(kind=Kdouble), intent(out) :: C(m,m)
#if MTYPEID == MTYPEID_REAL
        call dgemm('T', 'N', m, m, nn, 1.0d0, A, nn, B, nn, 0.0d0, C, m)
#else
        call zgemm('T', 'N', m, m, nn, (1.0d0,0.0d0), A, nn, B, nn, (0.0d0,0.0d0), C, m)
#endif
        end subroutine gemm_tn

        subroutine compute_omega(vin, omegaout)
        MTYPE(kind=Kdouble), intent(in) :: vin(nn,m)
        MTYPE(kind=Kdouble), intent(out) :: omegaout(m,m)
        integer :: j
        omegaout = 0.0_Kdouble
#if MTYPEID == MTYPEID_REAL
        forall (j=1:m) omegaout(j,j) = sqrt(sum(vin(:,j)*vin(:,j)))
#else
        forall (j=1:m) omegaout(j,j) = sqrt(sum(conjg(vin(:,j))*vin(:,j)))
#endif
        end subroutine compute_omega

        function compute_qres_mm(A) result(Qres)
        MTYPE(kind=Kdouble), intent(in) :: A(m,m)
        real(kind=Kdouble) :: Qres
#if MTYPEID == MTYPEID_REAL
        Qres = maxval(sqrt(sum(A*A, 1)))
#else
        Qres = maxval(sqrt(sum(abs(conjg(A)*A), 1)))
#endif
        end function compute_qres_mm

        function compute_qres_nm(A) result(Qres)
        MTYPE(kind=Kdouble), intent(in) :: A(nn,m)
        real(kind=Kdouble) :: Qres
#if MTYPEID == MTYPEID_REAL
        Qres = maxval(sqrt(sum(A*A, 1)))
#else
        Qres = maxval(sqrt(sum(abs(conjg(A)*A), 1)))
#endif
        end function compute_qres_nm

        subroutine compute_omegat(vin, omegain, omegatout)
        MTYPE(kind=Kdouble), intent(in) :: vin(nn,m), omegain(m,m)
        MTYPE(kind=Kdouble), intent(out) :: omegatout(nn,m)
        integer :: j
        do j = 1, m
            omegatout(:,j) = vin(:,j) / omegain(j,j)
        enddo
        end subroutine compute_omegat

        subroutine update_omegat()
        integer :: j
        do j = 1, m
            tmp0(j,:) = Qd(:,j,t3) / omega(j,j,t3p)
        enddo
#if MTYPEID == MTYPEID_REAL
        call dgemm('N', 'T', nn, m, m, 1.0d0, omegat, nn, Qc(1,1,t3), m, 0.0d0, tmp, nn)
        call dgemm('N', 'N', nn, m, m, 1.0d0, v(1,1,t3p), nn, tmp0, m, 1.0d0, tmp, nn)
#else
        call zgemm('N', 'C', nn, m, m, (1.0d0,0.0d0), omegat, nn, Qc(1,1,t3), m, (0.0d0,0.0d0), tmp, nn)
        call zgemm('N', 'N', nn, m, m, (1.0d0,0.0d0), v(1,1,t3p), nn, conjg(tmp0), m, (1.0d0,0.0d0), tmp, nn)
#endif
        omegat = tmp
        end subroutine update_omegat

        subroutine apply_precond_residual()
        integer :: j, ii
        if(this%pcond_type <= 0) return

        select case(this%pcond_type)
        case(1)  ! ILU-left
            tmp = vt
#if MTYPEID == MTYPEID_REAL
            call ILUPcondSolve(this%ilu, Ap, Ai, Ax, nn, m, vt, tmp)
#else
            call ILUPcondSolve(this%ilu, Ap, Ai, real(Ax,Kdouble), nn, m, &
                    rtmp(:,:,1), real(tmp,Kdouble), aimag(Ax), rtmp(:,:,2), aimag(tmp))
            vt = cmplx(rtmp(:,:,1), rtmp(:,:,2), kind=Kdouble)
#endif

        case(2)  ! ILU-split
            tmp = vt
#if MTYPEID == MTYPEID_REAL
            call ILUPcondSolveL(this%ilu, Ap, Ai, Ax, nn, m, vt, tmp)
#else
            call ILUPcondSolveL(this%ilu, Ap, Ai, real(Ax,Kdouble), nn, m, &
                    rtmp(:,:,1), real(tmp,Kdouble), aimag(Ax), rtmp(:,:,2), aimag(tmp))
            vt = cmplx(rtmp(:,:,1), rtmp(:,:,2), kind=Kdouble)
#endif

        case(3)  ! Jacobi-split
            do j = 1, m
                do ii = 1, nn
                    vt(ii,j) = vt(ii,j) / jacobi_sqrt_diag(ii)
                enddo
            enddo
        end select

        if(isnan(real(sum(vt-vt)))) this%flag = 2
        end subroutine apply_precond_residual

        subroutine apply_precond_matvec(vin, vtout)
        MTYPE(kind=Kdouble), intent(in) :: vin(nn,m)
        MTYPE(kind=Kdouble), intent(out) :: vtout(nn,m)
        integer :: j

        select case(this%pcond_type)
        case(1)  ! ILU-left
            call spmulmat(Ap, Ai, Ax, vin, tmp)
#if MTYPEID == MTYPEID_REAL
            call ILUPcondSolve(this%ilu, Ap, Ai, Ax, nn, m, vtout, tmp)
#else
            call ILUPcondSolve(this%ilu, Ap, Ai, real(Ax,Kdouble), nn, m, &
                  rtmp(:,:,1), real(tmp,Kdouble), aimag(Ax), rtmp(:,:,2), aimag(tmp))
            vtout = cmplx(rtmp(:,:,1), rtmp(:,:,2), kind=Kdouble)
#endif

        case(2)  ! ILU-split
#if MTYPEID == MTYPEID_REAL
            call ILUPcondSolveU(this%ilu, Ap, Ai, Ax, nn, m, tmp, vin)
#else
            call ILUPcondSolveU(this%ilu, Ap, Ai, real(Ax,Kdouble), nn, m, &
                  rtmp(:,:,1), real(vin,Kdouble), aimag(Ax), rtmp(:,:,2), aimag(vin))
            tmp = cmplx(rtmp(:,:,1), rtmp(:,:,2), kind=Kdouble)
#endif
            call spmulmat(Ap, Ai, Ax, tmp, tmp2_vec)
#if MTYPEID == MTYPEID_REAL
            call ILUPcondSolveL(this%ilu, Ap, Ai, Ax, nn, m, vtout, tmp2_vec)
#else
            call ILUPcondSolveL(this%ilu, Ap, Ai, real(Ax,Kdouble), nn, m, &
                  rtmp(:,:,1), real(tmp2_vec,Kdouble), aimag(Ax), rtmp(:,:,2), aimag(tmp2_vec))
            vtout = cmplx(rtmp(:,:,1), rtmp(:,:,2), kind=Kdouble)
#endif

        case(3)  ! Jacobi-split
            do j = 1, m
                tmp(:,j) = vin(:,j) / jacobi_sqrt_diag(:)
            enddo
            call spmulmat(Ap, Ai, Ax, tmp, tmp2_vec)
            do j = 1, m
                vtout(:,j) = tmp2_vec(:,j) / jacobi_sqrt_diag(:)
            enddo

        case default
            call spmulmat(Ap, Ai, Ax, vin, vtout)
        end select
        end subroutine apply_precond_matvec

        subroutine recover_solution()
        integer :: j
        if(this%pcond_type <= 0) return

        select case(this%pcond_type)
        case(2)  ! ILU-split
            tmp = x
#if MTYPEID == MTYPEID_REAL
            call ILUPcondSolveU(this%ilu, Ap, Ai, Ax, nn, m, x, tmp)
#else
            call ILUPcondSolveU(this%ilu, Ap, Ai, real(Ax,Kdouble), nn, m, &
                  rtmp(:,:,1), real(tmp,Kdouble), aimag(Ax), rtmp(:,:,2), aimag(tmp))
            x = cmplx(rtmp(:,:,1), rtmp(:,:,2), kind=Kdouble)
#endif

        case(3)  ! Jacobi-split
            do j = 1, m
                x(:,j) = x(:,j) / jacobi_sqrt_diag(:)
            enddo
        end select
        end subroutine recover_solution

        end subroutine BLQMRSolve

!--------------------------------------------------------------------------
!> \fn BLQMRPrint(this)
!> \brief print the internal variables inside a qmr solver structure
!--------------------------------------------------------------------------

        subroutine BLQMRPrint(this)
        implicit none

        type(BLQMRSolver), intent(in) :: this

        write(*,'(A)') '{'
        write(*,'(2A,I8,A)') char(9), '"n":', this%n, ','
        write(*,'(2A,I8,A)') char(9), '"nrhs":', this%nrhs, ','
        write(*,'(2A,E20.10E4,A)') char(9), '"qtol":', this%qtol, ','
        write(*,'(2A,E20.10E4,A)') char(9), '"droptol":', this%droptol, ','
        write(*,'(2A,I8,A)') char(9), '"maxit":', this%maxit, ','
        write(*,'(2A,I4,A)') char(9), '"state":', this%state, ','
        write(*,'(2A,I4,A)') char(9), '"pcond_type":', this%pcond_type, ','
        write(*,'(2A,I8,A)') char(9), '"nblock":', this%nblock, ','
        write(*,'(2A,I4,A)') char(9), '"isquasires":', this%isquasires, ','
        write(*,'(2A,E20.10E4,A)') char(9), '"res":', this%res, ','
        write(*,'(2A,I4,A)') char(9), '"debug":', this%debug,','
        write(*,'(2A,I4)') char(9), '"flag":', this%flag
        write(*,'(A)') '}'
  
        end subroutine BLQMRPrint