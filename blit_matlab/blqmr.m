function [x,flag,relres,iter,resv]=blqmr(A,B,qtol,maxit,M1,M2,x0,varargin)
%
% [x,flag,relres,iter,resv] = blqmr(A,B,qtol,maxit,M1,M2,x0,opt)
%
%   Block Quasi-Minimal-Residual (BL-QMR) iterative solver for sparse 
%   complex symmetric systems (suited for FEM modeling) in the form of
%                  A*[x1, x2, ...] = [b1, b2, ...]
%
% Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
%
% Address: 149 13th St, Charlestown, MA 02148
%          Martinos Center for Biomedical Imaging
%          Massachusetts General Hospital, Harvard Medical School
%
% Input:
%      A: a full/sparse symmetric NxN matrix, can be either real or complex
%      B: an N x Nrhs array, representing Nrhs right-hand-side vectors
%      qtol: quasi-residual convergence tolerance, a number in [0,1]; if
%            qtol=[] or ignored, blqmr sets qtol=1e-6
%      maxit: maximum iteration number, if maxit=[] or ignored, blqmr uses
%            min(N,20)
%      M1, M2: preconditioning matrix M=M1*M2. If only M1 is given, M=M1;
%            if both M1 and M2 are empty, preconditioning is disabled. With
%            M, blqmr solves for the precond. system inv(M)*Ax=inv(M)*B;
%            the inversion is applied by (M1*M2)\(B-Ax) -> (M2\(M1\(B-Ax)))
%            M is the left preconditioning matrix where inv(M)*A approx. I
%      x0: the initial guess of x, dimension N x Nrhs; if x0=[] or ignored,
%            blqmr sets x0 to zeros(size(B))
%      opt: an optional input for additional parameters. 
%            if opt.residual=1, blqmr uses the true residual B-A*x instead
%            of the quasi-residual
%
% Output: 
%      x: the solution, dimension N x Nrhs
%      flag: if flag=0, blqmr converges below qtol within maxit iterations;
%            if flag=1, blqmr fails to converge below qtol within maxit;
%            if flag=2, M1 and/or M2 are rank-deficient
%            if flag=3, blqmr stalls, residual no longer reduce
%      relres: the relative residual of the last iteration
%      iter: the last iteration number
%      resv: the vector of relective residuals for all iterations
%
% Examples:
%      
%      n=100; 
%      di=ones(n,1); 
%      A=spdiags([-di 2*di -di],-1:1,n,n);
%      b=rand(n,4);
%      x = blqmr(A,b,[],100);
%
%      [x2,flag,res,iter,resv]=blqmr(A,b,1e-5,1000);
%      semilogy(resv);
%
%      A=A+i*A/2;
%      R=cholinc(A,0.8);
%      [x3,flag,res,iter,resv]=blqmr(A,b,1e-5,1000,R,R');
%
% Reference:
%      Boyse, W. E., Seidl, A. A. (1996) A block QMR method for computing 
%      multiple simultaneous solutions to complex symmetric systems. 
%      SIAM J. Sci. Comput. 17, 263–274.
% 
%  -- this file is part of Blit sparse solver library 
%     URL: http://blit.sf.net
%
%  License:
%      BSD or LGPL or GPL, see License.txt for more details
%

%
% k-index range for all variables:
%      v:101,vt:1,beta:01,alpha:0,omega101,theta:0,Qa:10,Qb:210
%      Qc:10,Qd:210,zeta:0,zetat:0,eta:0,taot:01,tau:0,p:210
%

%% initialization of BLQMR

[n,m]=size(B);

znm1=zeros(n,m);
znm3=zeros(n,m,3);
zmm1=zeros(m,m);
zmm3=zeros(m,m,3);

[v,   vt,  beta,alpha,omega,theta,Qa,  Qb,  Qc,  Qd]=deal(...
 znm3,znm1,zmm3,zmm1, zmm3, zmm1, zmm3,zmm3,zmm3,zmm3);

[zeta,zetat,eta,tau,taot,p]=deal(zmm1,zmm1,zmm1,zmm1,zmm1,znm3);

[t3,t3n,t3p]=updateidx(0);

Qa(:,:,t3)=eye(m);   %eq (35) in [Boyse1996], same below
Qd(:,:,t3n)=eye(m);
Qd(:,:,t3)=eye(m);

%% check input options

isprecond=0;
if(nargin<3 || (nargin>=3 && isempty(qtol)))
    qtol=1e-6;
end

if(nargin<4 || (nargin>=4 && isempty(maxit)))
    maxit=min(n,20);
end

% initialize preconditioner
if(nargin>=5 && ~isempty(M1))
    if(nargin>=6 && ~isempty(M2))
        isprecond=2;
    else
        M=M1;
        isprecond=1;
    end
else
    if(nargin>=6 && ~isempty(M2))
        fprintf(1,'M2 is ignored as M1 is not given');
    end
end

% initialize initial guess
if(~exist('x0','var') || isempty(x0))
    x0=zeros(size(B));
end
x=x0;
iter=0;
relres=1;

isquasires=1;
if(nargin>7 && ~isempty(varargin{:}))
   opt=varargin{1};
   if(~isstruct(opt))
       error('opt must be a structure'); 
   end
   if(isfield(opt,'residual'))
       isquasires=~opt.residual;
   end
end

% vt0 contains the initial true residual
if(isprecond)
    if(isprecond==2)
        vt=M1\(B-A*x0);
        vt=M2\vt;
    else
        vt=M\(B-A*x0);
    end
    if(any(isnan(vt)))
        flag=2;
        if(nargout<=1)
          fprintf('the preconditioner is rank-deficient, blqmr stops\n');
        end
        return;
    end
else
    vt=B-A*x0;                                %eq (36)
end


[Q,R]=qqr(vt,0); % quasi-qr decomposition,    %eq (37)
v(:,:,t3p)=Q;
beta(:,:,t3p)=R;

omega(:,:,t3p)=diag(sqrt(sum(conj(Q).*Q,1))); %eq (38)
taot=omega(:,:,t3p)*beta(:,:,t3p);            %eq (39)

% calculate the initial residual
if(isquasires)
   Qres0=max(sqrt(sum(conj(taot).*taot,1)));  %eq (31)
else
   omegat=Q*diag(1./diag(omega(:,:,t3p)));
   % R=(omegat*taot);
   % proof R==vt: omegat*taot=Q*inv(omega)*omega*beta=Q*R=vt
   Qres0=max(sqrt(sum(conj(vt).*vt,1)));      %eq (32)
end

%% start iterative solving process

flag=1;
resv=zeros(maxit,1);
for k=1:maxit
    [t3,t3n,t3p,t3nn]=updateidx(k);
    if(isprecond)
       if(isprecond==2)
           tmp=M1\(A*v(:,:,t3));
           tmp=M2\tmp;
           vt=tmp-v(:,:,t3n)*beta(:,:,t3).';
       else
           vt=M\(A*v(:,:,t3))-v(:,:,t3n)*beta(:,:,t3).';
       end
    else
       vt=(A*v(:,:,t3))-v(:,:,t3n)*beta(:,:,t3).'; % eq (40)
    end
    alpha=v(:,:,t3).'*vt;  % eq (41)
    vt=vt-v(:,:,t3)*alpha; % eq (42)
    [Q,R]=qqr(vt,0);       % eq (43), quasi-qr decomposition
    v(:,:,t3p)=Q;
    beta(:,:,t3p)=R;
    omega(:,:,t3p)=diag(sqrt(sum(conj(Q).*Q,1))); % eq (44)
    tmp0=omega(:,:,t3n)*beta(:,:,t3).';
    theta=Qb(:,:,t3nn)*tmp0; % eq (45)
    tmp1= Qd(:,:,t3nn)*tmp0;
    tmp2= omega(:,:,t3)*alpha;
    eta=  Qa(:,:,t3n)*tmp1+Qb(:,:,t3n)*tmp2; % eq (46)
    zetat=Qc(:,:,t3n)*tmp1+Qd(:,:,t3n)*tmp2; % eq (47)

    [QQ,zeta]=qr([zetat ; omega(:,:,t3p)*beta(:,:,t3p)]);  % eq (48)
    zeta=zeta(1:m,1:m);
    QQ=QQ'; % QQ is an orthogonal matrix, thus, Q'=inv(Q), here is 1 of the 2
            % places need congugate transpose

    Qa(:,:,t3)=QQ(1:m,1:m);
    Qb(:,:,t3)=QQ(1:m,m+1:2*m);
    Qc(:,:,t3)=QQ(m+1:2*m,1:m);
    Qd(:,:,t3)=QQ(m+1:2*m,m+1:2*m);

    p(:,:,t3)=(v(:,:,t3)-p(:,:,t3n)*eta-p(:,:,t3nn)*theta)*inv(zeta); %(49)
    tau=Qa(:,:,t3)*taot;  % eq (50)
    x=x+p(:,:,t3)*tau;    % eq (51)
    taot=Qc(:,:,t3)*taot; % eq (52)

    % calculate residual and terminate if converge
    if(isquasires)
       Qres=max(sqrt(sum(conj(taot).*taot,1))); %eq (31)
    else
       omegat=omegat*Qc(:,:,t3)'+v(:,:,t3p)*(Qd(:,:,t3)*diag(1./diag(omega(:,:,t3p))))';
       R=omegat*taot; %R is the residual, R==B-A*x;
       Qres=max(sqrt(sum(conj(R).*R,1))); % error is the max column norm
    end
    resv(k)=Qres;
    if(k>1 && Qres==Qres1)
       flag=3; % stagnated
       break;
    end
    Qres1=Qres;
    relres=Qres/Qres0;
    iter=k;
    if(relres <= qtol)
       flag=0;
       if(nargout<=1)
         fprintf(['blqmr converged at iteration %d with a relative ' ...
                 'quasi-residual %e\n'],iter,relres);
       end
       break;
    end
end
resv=resv(1:k);

if((flag==1 || flag==3) && nargout<=1)
    resname='quasi-';
    if(isquasires==0); resname=''; end;
    warning(['blqmr failed to converge within %d iterations,\nthe final'...
            ' %sresidual was %e\n'],iter,resname,relres);
end

%% cylic array index
function tnew=cycidx(t,dim)
tnew=mod(t,dim)+1;

%% compute the cylic indices for all internal arrays
function [t3,t3n,t3p,t3nn]=updateidx(t)
t3=cycidx(t,3);    % step k
t3n=cycidx(t-1,3); % step k-1
t3p=cycidx(t+1,3); % step k+1
t3nn=cycidx(t-2,3);% step k-2
