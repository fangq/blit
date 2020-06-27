function [q,r]=qqr(A,B)
%
% [q,r]=qqr(A,B)
%
%   Quasi-QR decomposition of a general rectangular matrix using modified 
%   Gram-Schmidt
%
% Author: Qianqian Fang <q.fang at neu.edu>
%
% Address: 360 Huntington Ave, ISEC 206
%          Boston, MA 02115, USA
%          Dept. of Bioengineering, Northeastern University, USA
%
% Input:
%      A: a general N x M matrix, can be either real or complex
%      B: if B=0, qqr returns the economic Quasi-QR decomposition
%
% Output: 
%      q: a quasi-orthogonal matrix, q.'*q=I (in contrast, the q matrix
%         output from qr() satisfies q'*q=I). When B=0 and N>M, q has a 
%         dimension of N x M; otherwise, q is N x N
%      r: an upper triangular matrix and satisfies q*r=A
%         When B=0 and N>M, r only contains the top M x M submatrix; 
%         otherwise, r is an N x M matrix with the bottom N-M rows of 0s
%
% Examples:
%      
%      a=magic(5);
%      [Q,R]=qqr(a,0);
%
% Reference:
%   Golub&Van Loan, "Matrix Computations," Johns Hopkins, 1996, pp. 231-232
% 
% License:
%      BSD or LGPL or GPL, see License.txt for more details
%
%  -- this file is part of Blit sparse solver library 
%     URL: http://blit.sf.net
%

dim=size(A);
m=dim(1);
n=dim(2);

q=A;
r=zeros(m,n);

iseco=0;
if(nargin>=2)
    if(length(B)==1 && B==0 && m>n)
        r=zeros(n,n);
        q=A;
        iseco=1;
    end
end

if(iseco==0 && m>n)
    q=[A zeros(m,m-n)];
end

for k=1:n
    r(k,k)=sqrt(q(:,k).'*q(:,k));
    q(:,k)=q(:,k)*(1/r(k,k));
    for j=k+1:n
        r(k,j)=q(:,k).'*q(:,j);
        q(:,j)=q(:,j)-r(k,j)*q(:,k);
    end
end
