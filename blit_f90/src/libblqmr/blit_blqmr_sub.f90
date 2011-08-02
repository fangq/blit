!=========================================================
!  
!=========================================================

        subroutine BLQMROnCreate(this, n, nrhs, nnz)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer :: n, nrhs, nnz, t3, t3n

        this%n=n
        this%nrhs=nrhs

        allocate(vt(n,nrhs),alpha(nrhs,nrhs),zeta(nrhs,nrhs))
        allocate(zetat(nrhs,nrhs),eta(nrhs,nrhs),tao(nrhs,nrhs))
        allocate(taot(nrhs,nrhs),theta(nrhs,nrhs),x0(n,nrhs))

        ! only need 2 copies, allocate 3 to simplify
        allocate(beta(nrhs,nrhs,3),Qa(nrhs,nrhs,3),Qc(nrhs,nrhs,3)) 
        allocate(v(n,nrhs,3),omega(nrhs,nrhs,3))
        allocate(Qb(nrhs,nrhs,3),Qd(nrhs,nrhs,3),p(n,nrhs,3))

        v=0.0_Kdouble
        vt=0.0_Kdouble
        alpha=0.0_Kdouble
        theta=0.0_Kdouble
        zeta=0.0_Kdouble
        zetat=0.0_Kdouble
        eta=0.0_Kdouble
        tao=0.0_Kdouble
        taot=0.0_Kdouble
        beta=0.0_Kdouble
        Qa=0.0_Kdouble
        Qb=0.0_Kdouble
        Qc=0.0_Kdouble
        Qd=0.0_Kdouble
        p=0.0_Kdouble
        omega=0.0_Kdouble
        this%dopcond=1

        t3=modulo(0,3)+1;t3n=modulo(-1,3)+1
        call eye(nrhs,Qa(:,:,t3))
        Qd(:,:,t3n)=Qa(:,:,t3)
        Qd(:,:,t3) =Qa(:,:,t3)
        
        this%qtol=1e-6_Kdouble
        this%droptol=0.8_Kdouble
        this%maxit=this%n
        this%state=0
        
        end subroutine BLQMROnCreate

!=========================================================
!  
!=========================================================

        subroutine BLQMROnDestroy(this)
        implicit none

        type(BLQMRSolver), intent(in) :: this

        call ILUPcondDestroy(ilu)

        if(allocated(v))     deallocate(v)
        if(allocated(vt))    deallocate(vt)
        if(allocated(beta))  deallocate(beta)
        
        if(allocated(alpha)) deallocate(alpha)
        if(allocated(omega)) deallocate(omega)
        if(allocated(theta)) deallocate(theta)
        if(allocated(Qa))    deallocate(Qa)
        if(allocated(Qb))    deallocate(Qb)
        if(allocated(Qc))    deallocate(Qc)
        if(allocated(Qd))    deallocate(Qd)
        if(allocated(zeta))  deallocate(zeta)

        if(allocated(zetat)) deallocate(zetat)
        if(allocated(eta))   deallocate(eta)
        if(allocated(tao))   deallocate(tao)
        if(allocated(taot))  deallocate(taot)
        if(allocated(p))     deallocate(p)

        end subroutine BLQMROnDestroy

!=========================================================
!  
!=========================================================

        subroutine BLQMRPrep(this, Ap, Ai, Ax, nnz, xini, b)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer :: nnz, status, i, t3, t3p, Ap(this%n+1), Ai(nnz)
        MTYPE(kind=Kdouble) :: Ax(nnz)
        MTYPE(kind=Kdouble),dimension(this%n,this%nrhs) :: tmp, b, xini
        MTYPE(kind=Kdouble),dimension(min(this%n,this%nrhs)) :: tau
        MTYPE(kind=Kdouble),dimension(this%n) :: work
#if MTYPEID == MTYPEID_COMPLEX
        real(kind=Kdouble),dimension(this%n,this%nrhs,2) :: rtmp
#endif

        if(this%dopcond > 0) then
                call ILUPcondCreate(ilu,this%n,nnz,0)
#if MTYPEID == MTYPEID_COMPLEX
                call ILUPcondPrep(ilu,Ap, Ai, real(Ax), this%droptol,imag(Ax))
#else
                call ILUPcondPrep(ilu,Ap, Ai, Ax, this%droptol)
#endif
        endif

	x0=xini
        call spmulmat(Ap,Ai,Ax,x0,tmp)
        vt=b-tmp

        if(this%dopcond>0) then
                tmp=vt
#if MTYPEID == MTYPEID_REAL
                call ILUPcondSolve(ilu,Ap,Ai,Ax,this%n,this%nrhs,vt,tmp)
#else
                call ILUPcondSolve(ilu,Ap,Ai,real(Ax),this%n,this%nrhs,&
                        rtmp(:,:,1),real(tmp),imag(Ax),rtmp(:,:,2),imag(tmp))
                vt=dcmplx(rtmp(:,:,1),rtmp(:,:,2))
#endif
        endif

        t3p=modulo(1,3)+1

        call qqr(vt,v(:,:,t3p),beta(:,:,t3p))
        
#if MTYPEID == MTYPEID_REAL
        forall (i=1:this%nrhs) omega(i,i,t3p)=sqrt(sum(v(:,i,t3p)*v(:,i,t3p)))
#else
        forall (i=1:this%nrhs) omega(i,i,t3p)=sqrt(sum(conjg(v(:,i,t3p))*v(:,i,t3p)))
#endif
        taot=matmul(omega(:,:,t3p),beta(:,:,t3p))

        this%state=1
        this%iter=-1
        this%flag=-1

        end subroutine BLQMRPrep

!=========================================================
!  
!=========================================================

        subroutine BLQMRSolve(this, Ap, Ai, Ax, nnz, x, b)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer :: status,i,k,m,t3,t3p,t3n,t3nn, Ap(:), Ai(:), nnz
        real(kind=Kdouble) :: Qres, Qres1, Qres0, relres
        MTYPE(kind=Kdouble) :: Ax(:)
        MTYPE(kind=Kdouble),dimension(this%nrhs,this%nrhs) :: tmp0,tmp1,tmp2,tmp3
        MTYPE(kind=Kdouble),dimension(this%nrhs*2,this%nrhs) :: ZZ,zetafull
        MTYPE(kind=Kdouble),dimension(this%nrhs*2,this%nrhs*2) :: QQ
        MTYPE(kind=Kdouble),dimension(this%n,this%nrhs) :: tmp, b, x
#if MTYPEID == MTYPEID_COMPLEX
        real(kind=Kdouble),dimension(this%n,this%nrhs,2) :: rtmp
#endif

        m=this%nrhs
        if(this%state==0) call BLQMRPrep(this, Ap, Ai, Ax, nnz, x, b)

        this%flag=1
        do k=1,this%maxit
                t3=modulo(k,3)+1;t3p=modulo(k+1,3)+1;t3n=modulo(k-1,3)+1;t3nn=modulo(k-2,3)+1

                call spmulmat(Ap,Ai,Ax,v(:,:,t3),tmp)

                if(this%dopcond > 0) then
#if MTYPEID == MTYPEID_REAL
                        call ILUPcondSolve(ilu,Ap,Ai,Ax,this%n,this%nrhs,vt,tmp)
#else
                        call ILUPcondSolve(ilu,Ap,Ai,real(Ax),this%n,this%nrhs,rtmp(:,:,1),real(tmp),imag(Ax),rtmp(:,:,2),imag(tmp))
                        vt=dcmplx(rtmp(:,:,1),rtmp(:,:,2))
#endif
                        vt=vt-matmul(v(:,:,t3n),transpose(beta(:,:,t3)))
                else
                        vt=tmp-matmul(v(:,:,t3n),transpose(beta(:,:,t3)))
                endif


                alpha=matmul(transpose(v(:,:,t3)),vt)
                vt=vt-matmul(v(:,:,t3),alpha)

                call qqr(vt,v(:,:,t3p),beta(:,:,t3p))

#if MTYPEID == MTYPEID_REAL
                forall (i=1:this%nrhs) omega(i,i,t3p)=sqrt(sum(v(:,i,t3p)*v(:,i,t3p)))
#else
                forall (i=1:this%nrhs) omega(i,i,t3p)=sqrt(sum(conjg(v(:,i,t3p))*v(:,i,t3p)))
#endif
                tmp0 =matmul(omega(:,:,t3n), transpose(beta(:,:,t3)))
                theta=matmul(Qb(:,:,t3nn),tmp0)
                tmp1 =matmul(Qd(:,:,t3nn),tmp0)
                tmp2 =matmul(omega(:,:,t3),alpha)
                eta  =matmul(Qa(:,:,t3n),tmp1)+matmul(Qb(:,:,t3n),tmp2)
                zetat=matmul(Qc(:,:,t3n),tmp1)+matmul(Qd(:,:,t3n),tmp2)

                ZZ(1:m,1:m)=zetat
                ZZ(m+1:2*m,:)=matmul(omega(:,:,t3p),beta(:,:,t3p))

                call qr(ZZ,QQ,zetafull,1)
                zeta=zetafull(1:m,1:m)
#if MTYPEID == MTYPEID_REAL
                QQ=transpose(QQ)
#else
                QQ=conjg(transpose(QQ))
#endif

                Qa(:,:,t3)=QQ(1:m,1:m)
                Qb(:,:,t3)=QQ(1:m,1+m:2*m)
                Qc(:,:,t3)=QQ(1+m:2*m,1:m)
                Qd(:,:,t3)=QQ(1+m:2*m,1+m:2*m)

                call inv(zeta,tmp0)

                p(:,:,t3)=matmul((v(:,:,t3)-matmul(p(:,:,t3n),eta)-matmul(p(:,:,t3nn),theta)),tmp0)
                tao=matmul(Qa(:,:,t3),taot)
                x=x+matmul(p(:,:,t3),tao)
                taot=matmul(Qc(:,:,t3),taot)

#if MTYPEID == MTYPEID_REAL
                Qres=maxval(sqrt(sum(taot*taot,1)))
#else
                Qres=maxval(sqrt(sum(abs(conjg(taot)*taot),1)))
#endif
                if(k>1 .and. Qres==Qres1) then
                     this%flag=3
                     exit
                endif

                if(k==1) Qres0=Qres
                Qres1=Qres
                relres=Qres/Qres0
                this%iter=k
                if(relres < this%qtol) then
                        this%flag=0
                        exit
                endif
        enddo
        this%state=2

        end subroutine BLQMRSolve