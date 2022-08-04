function [H,S] = BCA_reconstruct(Y,n)
%Y：signal mixture
%n: number of sourses
%H: separated channel with ambiguity
%S: signals obtained by L:east Square Method corresponding to H
MaxIter=1500; % the maximum iterations 
checkpoint=40; % the number of iterations, after that the program begins to check whether the convergence 
alpha=1.8; % penalty factor %the                                                              
[W,Rank]=whitening(Y,n); % whitening 
m=Rank;
z=W*Y; 
q=15*(2*m*n);                     
E=randn(m,q)+1i*randn(m,q);      
B= E./repmat(sqrt(sum(abs(W'*E).^2)),m,1); %the matrix consists of one-dimension projection vector
L=perimeter(B,z); %obtain the vector of convex perimeter after projection
G=(1/sqrt(2))*(randn(q,n)+1i*randn(q,n)); %initialization of H after projection
P=eye(270)-B'*inv(B*B')*B; %the matrix in the constrain condition
[Cost(1),penalty(1)]=EvalCost_penalty(G,L,P,alpha);
i=1;
Continue=1;%one of the flags of the loop
while Continue&&(i<=MaxIter)
i=i+1;
for j=1:q
GradG(j,:)=-8*(L(j,1)-4*norm(G(j,:),1))*[(G(j,1)^(1/2)*(G(j,1)')^(-1/2))/2 (G(j,2)^(1/2)*(G(j,2)')^(-1/2))/2 (G(j,3)^(1/2)*(G(j,3)')^(-1/2))/2];
end
GradG_p=alpha*P'*P*G; %part of gradient
NGrad=GradG+GradG_p; % gradient of whole function
mu=0.7*Cost(i-1)/(trace(NGrad'*NGrad));% step-size
G=G-mu*NGrad;
[Cost(i),penalty(i)]=EvalCost_penalty(G,L,P,alpha);
%Check whrether the function reach its convergence
if i>checkpoint
    Dist=20;
    Continue=std(Cost(i-Dist:i))>10;
end
end
H=pinv(W)*pinv(B')*G;
S=pinv(H)*Y;
end
%%----------------------------------------------------------------------
function [W,Rank]=whitening(Y,n)
% Estimates the whitening matrix W and its rank m
M   =size(Y,1);
[Q,D]=eig(cov(Y'));
[v,i]=sort(diag(D)); vy=flipud([v,i]);
if M>n
    vn=mean(vy(n+1:M,1)); % Estimated noise power
else    
    vn=0;                 % Estimated noise power
end
Rank=min(M,n); % m =rank{W}.
D=diag(sqrt(max(vy(1:Rank,1)-vn,eps)));
W=pinv(D)*Q(:,vy(1:Rank,2))'; % whitening matrix
end
%%-------------------------------------------------------------------
function L=perimeter(B,y)
% Returns L, the vector of perimeters of projections of the observations
q=size(B,2);
for i=1:q
    y_=B(:,i)'*y;
    k = convhull(real(y_),imag(y_));
    L(i,1)=sum(abs(y_(k(2:length(k)))-y_(k(1:(length(k)-1)))));
end
end
%%--------------------------------------------------------------------
function [Cost,penalty]=EvalCost_penalty(G,L,P,alpha)
e=L-4*sum(abs(G),2); %4==>for BPSK
penalty=norm(P*G)^2;
Cost=norm(e)^2+alpha*penalty;
end
