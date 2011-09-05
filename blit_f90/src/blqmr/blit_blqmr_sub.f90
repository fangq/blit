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
!> \fn BLQMROnCreate(this, n)
!> \brief initialize a block-QMR solver object for an n-by-n sparse system
!--------------------------------------------------------------------------

        subroutine BLQMROnCreate(this, n)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer :: n

        this%n=n
        this%qtol=1e-6_Kdouble
        this%droptol=0.001_Kdouble
        this%maxit=this%n
        this%state=0
        this%dopcond=1
        this%isquasires=1
        this%debug=0

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

        this%state=-1;

        end subroutine BLQMROnDestroy

!--------------------------------------------------------------------------
!> \fn BLQMRPrep(this, Ap, Ai, Ax, nnz)
!> \brief create preconditioner and get ready for solving a block system
!--------------------------------------------------------------------------

        subroutine BLQMRPrep(this, Ap, Ai, Ax, nnz)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer :: nnz, Ap(this%n+1), Ai(nnz)
        MTYPE(kind=Kdouble) :: Ax(nnz)

        if(this%dopcond > 0) then
                call ILUPcondCreate(this%ilu,this%n,nnz)
#if MTYPEID == MTYPEID_COMPLEX
                this%ilu%iscomplex=1
                call ILUPcondPrep(this%ilu,Ap, Ai, real(Ax), this%droptol,aimag(Ax))
#else
                call ILUPcondPrep(this%ilu,Ap, Ai, Ax, this%droptol)
#endif
        endif

        this%state=1
        this%iter=-1
        this%flag=-1

        end subroutine BLQMRPrep

!--------------------------------------------------------------------------
!> \fn BLQMRSolve(this, Ap, Ai, Ax, nnz, x, b)
!> \brief solving a real or complex system using BLQMR algorithm
!--------------------------------------------------------------------------

        subroutine BLQMRSolve(this, Ap, Ai, Ax, nnz, x, b)
        implicit none

        type(BLQMRSolver), intent(inout) :: this
        integer :: i,k,m,t3,t3p,t3n,t3nn, Ap(:), Ai(:), nnz
        integer, parameter :: DEBUG_RES=1
        real(kind=Kdouble) :: Qres, Qres1, Qres0
        MTYPE(kind=Kdouble) :: Ax(:), b(:,:), x(:,:)
        MTYPE(kind=Kdouble),dimension(size(b,2),size(b,2)) :: tmp0,tmp1,tmp2
        MTYPE(kind=Kdouble),dimension(size(b,2)*2,size(b,2)) :: ZZ,zetafull
        MTYPE(kind=Kdouble),dimension(size(b,2)*2,size(b,2)*2) :: QQ
        MTYPE(kind=Kdouble),dimension(this%n,size(b,2)) :: tmp, omegat
        MTYPE(kind=Kdouble),dimension(size(b,2),size(b,2))  :: alpha,theta,zeta,zetat,eta,tau,taut
        MTYPE(kind=Kdouble),dimension(size(b,2),size(b,2),3):: beta,Qa,Qb,Qc,Qd,omega
        MTYPE(kind=Kdouble),dimension(this%n,size(b,2))  :: vt
        MTYPE(kind=Kdouble),dimension(this%n,size(b,2),3)  :: v,p
#if MTYPEID == MTYPEID_COMPLEX
        real(kind=Kdouble),dimension(this%n,size(b,2),2) :: rtmp
#endif

        this%nrhs=size(b,2)
        m=this%nrhs

        if(this%state==0) call BLQMRPrep(this, Ap, Ai, Ax, nnz)

        v=0.0_Kdouble
        vt=0.0_Kdouble
        alpha=0.0_Kdouble
        theta=0.0_Kdouble
        zeta=0.0_Kdouble
        zetat=0.0_Kdouble
        eta=0.0_Kdouble
        tau=0.0_Kdouble
        taut=0.0_Kdouble
        beta=0.0_Kdouble
        Qa=0.0_Kdouble
        Qb=0.0_Kdouble
        Qc=0.0_Kdouble
        Qd=0.0_Kdouble
        p=0.0_Kdouble
        omega=0.0_Kdouble

        t3=modulo(0,3)+1;t3n=modulo(-1,3)+1
        call eye(this%nrhs,Qa(:,:,t3))
        Qd(:,:,t3n)=Qa(:,:,t3)
        Qd(:,:,t3) =Qa(:,:,t3)

        this%relres=1._Kdouble
        Qres1=-1._Kdouble
        call spmulmat(Ap,Ai,Ax,x,tmp)
        vt=b-tmp

        if(this%dopcond>0) then
                tmp=vt
#if MTYPEID == MTYPEID_REAL
                call ILUPcondSolve(this%ilu,Ap,Ai,Ax,this%n,this%nrhs,vt,tmp)
#else
                call ILUPcondSolve(this%ilu,Ap,Ai,real(Ax),this%n,this%nrhs,&
                        rtmp(:,:,1),real(tmp),aimag(Ax),rtmp(:,:,2),aimag(tmp))
                vt=cmplx(rtmp(:,:,1),rtmp(:,:,2))
#endif
                if(isnan(real(sum(vt-vt)))) then
                        this%flag=2
                        return
                endif
        endif

        t3p=modulo(1,3)+1

        call qqr(vt,v(:,:,t3p),beta(:,:,t3p))

#if MTYPEID == MTYPEID_REAL
        forall (i=1:this%nrhs) omega(i,i,t3p)=sqrt(sum(v(:,i,t3p)*v(:,i,t3p)))
#else
        forall (i=1:this%nrhs) omega(i,i,t3p)=sqrt(sum(conjg(v(:,i,t3p))*v(:,i,t3p)))
#endif
        taut=matmul(omega(:,:,t3p),beta(:,:,t3p))

        if(this%isquasires>0) then
#if MTYPEID == MTYPEID_REAL
            Qres0=maxval(sqrt(sum(taut*taut,1)))
#else
            Qres0=maxval(sqrt(sum(abs(conjg(taut)*taut),1)))
#endif
        else
            do i=1, m
                omegat(:,i)=v(:,i,t3p)*(1._Kdouble/omega(i,i,t3p))
            enddo
#if MTYPEID == MTYPEID_REAL
            Qres0=maxval(sqrt(sum(vt*vt,1)))
#else
            Qres0=maxval(sqrt(sum(abs(conjg(vt)*vt),1)))
#endif
        endif
        this%flag=1
        do k=1,this%maxit
                t3=modulo(k,3)+1;t3p=modulo(k+1,3)+1;t3n=modulo(k-1,3)+1;t3nn=modulo(k-2,3)+1

                call spmulmat(Ap,Ai,Ax,v(:,:,t3),tmp)

                if(this%dopcond > 0) then
#if MTYPEID == MTYPEID_REAL
                        call ILUPcondSolve(this%ilu,Ap,Ai,Ax,this%n,this%nrhs,vt,tmp)
#else
                        call ILUPcondSolve(this%ilu,Ap,Ai,real(Ax),this%n,this%nrhs,rtmp(:,:,1),real(tmp),aimag(Ax),&
                              rtmp(:,:,2),aimag(tmp))
                        vt=cmplx(rtmp(:,:,1),rtmp(:,:,2))
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
                tau=matmul(Qa(:,:,t3),taut)
                x=x+matmul(p(:,:,t3),tau)
                taut=matmul(Qc(:,:,t3),taut)

                if(this%isquasires>0) then
#if MTYPEID == MTYPEID_REAL
                    Qres=maxval(sqrt(sum(taut*taut,1)))
#else
                    Qres=maxval(sqrt(sum(abs(conjg(taut)*taut),1)))
#endif
                else
                    do i=1, m
                       tmp0(i,:)=Qd(:,i,t3)*(1._Kdouble/omega(i,i,t3p))
                    enddo
#if MTYPEID == MTYPEID_REAL
                    omegat=matmul(omegat,transpose(Qc(:,:,t3)))+matmul(v(:,:,t3p),tmp0)
                    tmp=matmul(omegat,taut)
                    Qres=maxval(sqrt(sum(tmp*tmp,1)))
#else
                    omegat=matmul(omegat,transpose(conjg(Qc(:,:,t3))))+matmul(v(:,:,t3p),conjg(tmp0))
                    tmp=conjg(matmul(omegat,taut))
                    Qres=maxval(sqrt(sum(abs(conjg(tmp)*tmp),1)))
#endif                
                endif
                this%res=Qres

                if(iand(this%debug,DEBUG_RES)>0) &
                    write(*,'(A,I4,A,E16.8,A,E16.8)') 'Iteration [',k,'] MaxResidual=', Qres, ', Relative=', Qres/Qres0

                if(k>1 .and. Qres==Qres1) then
                     this%flag=3
                     exit
                endif

                Qres1=Qres
                this%relres=Qres/Qres0
                this%iter=k
                if(this%relres < this%qtol) then
                        this%flag=0
                        exit
                endif
        enddo

        end subroutine BLQMRSolve
