function [H,S] = BCA_reconstruct_new(Y,n,beta)
%Y：signal mixture
%n: number of sourses
%H: separated channel with ambiguity
%S: signals obtained by L:east Square Method corresponding to H
MaxIter=2000; % the maximum iterations 
checkpoint=100; % the number of iterations, after that the program begins to check whether the convergence 
alpha=15; % penalty factor %the  
[W,Rank]=whitening(Y,n); % whitening 
m=Rank;
z=W*Y; 
q=15*(2*m*n);                     
H=randn(m,q)+1i*randn(m,q);      
B= H./repmat(sqrt(sum(abs(W'*H).^2)),m,1); %the matrix consists of one-dimension projection vector
M   =size(Y,1);
[~,D]=eig(cov(Y'));
[v,j]=sort(diag(D)); 
vy=flipud([v,j]);
Lr=0.9*sqrt(sum(vy(4:size(Y,1),1)));
L=perimeter(B,z)-Lr; %obtain the vector of convex perimeter after projection
G=(1/sqrt(2))*(randn(q,n)+1i*randn(q,n)); %initialization of H after projection
R=eye(270)-B'*inv(B*B')*B; %the matrix in the constrain condition
[Cost(1),penalty(1)]=EvalCost_penalty(G,L,R,alpha,beta);
i=1;
Continue=1;%one of the flags of the loop
while Continue&&(i<=MaxIter)
i=i+1;
for j=1:q
GradG(j,:)=-2*beta*(L(j,1)-beta*norm(G(j,:),1))*[(G(j,1)^(1/2)*(G(j,1)')^(-1/2))/2 (G(j,2)^(1/2)*(G(j,2)')^(-1/2))/2 (G(j,3)^(1/2)*(G(j,3)')^(-1/2))/2];
end
GradG_p=alpha*3*R'*R*G; %part of gradient
NGrad=GradG+GradG_p; % gradient of whole function
mu=0.65*Cost(i-1)/(trace(NGrad'*NGrad));% step-size
flag(1,i)=trace(NGrad'*NGrad);
if i==2
    c=1;
else
    c=c/(0.8^2);
end
Cost(i)=Cost(i-1)+5;%initial the value of Cost function 
while c>1e-3&&(Cost(i)>Cost(i-1)- 0.7*mu*c*(trace(NGrad'*NGrad)))
    G1=G-c*mu*NGrad;
    [Cost(i),penalty(i)]=EvalCost_penalty(G1,L,R,alpha,beta);
    c=c*0.8;
    if c<1e-3&&(Cost(i)>Cost(i-1)- 0.7*mu*c*(trace(NGrad'*NGrad)))
        c=1;
        G1=G-c*mu*NGrad;
        [Cost(i),penalty(i)]=EvalCost_penalty(G1,L,R,alpha,beta);
        break;
    end
end
G=G1;
%Check whrether the function reach its convergence
if i>checkpoint
    if trace(NGrad'*NGrad)<1 %the shreshold of the gradient
    Dist=3;
    Continue=std(Cost(i-Dist:i)/(size(Y,2)))>1e-3;% Standard Deviation verification
    end   
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
function [Cost,penalty]=EvalCost_penalty(G,L,R,alpha,beta)
e=L-beta*sum(abs(G),2); %4==>for BPSK
% penalty=norm(R*G)^2;
penalty=norm(R*G*[1;1;1])^2;
Cost=norm(e)^2+alpha*penalty;
end