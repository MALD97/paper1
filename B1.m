function [A,S,Cost] =B1(x,n,H)

% Function that identifies the mixing matrix in Bounded Component Analysis

% except for the permutation and scaling indeterminacies of its columns.

% This implementation of the BCA-PM algorithm considers complex signals

% and works for both cases of undetermined and (over)determined mixtures.

%

% (c) Sergio Cruces 2014, sergio(at)us.es

%

%-OUTPUTS-----------------------------------------------------------------

% A    = estimated mixing matrix up to permutation-scaling indeterminacy.

% Cost = evaluation of the cost function through the iterations.

%

%-INPUTS------------------------------------------------------------------

% x    = matrix of observations with dimension (n. sensors, n. samples)

% n    = number of sources that are mixed to give the observations,

%        when this information is not available set it to [].

% H    = matrix whose q columns (of dim m'=min(n,m)) fix the proyections.

%    It can be obtained with:

%    d=min(n,m);                       % dim. of signal subspace

%    q = k*(2*n*d);                    % n. of projections

%    H=SolveTammesProblem(d,q,0);      % random projections  (for large q) 

%    H=SolveTammesProblem(d,q,MaxIter);% homogeneous project.(for small q)

%

%-EXAMPLES OF USE---------------------------------------------------------

%  A      = BCA-PM(x,n);

% [A,Cost]= BCA-PM(x,n,H);

% 

%-BIBLIOGRAPHY------------------------------------------------------------

%

% This code can be only reused for academic or research purposes, 

% provided that the authors made explicit reference to the manuscript:

%

% [1] S. Cruces, "Bounded component analysis of noisy underdetermined and

%     overdetermined mixtures", IEEE Trans. on Signal Processing, 2015.

%     http://www.personal.us.es/sergio/PAPERS/2015-TSP-draft.pdf

%

%-Copyright (c) Sergio Cruces 2014----------------------------------------

%

% All rights reserved. 

% Redistribution and use in source and binary forms, with or without

% modification, are permitted for academic or research purposes

% provided that the following conditions are met:

% 1. Redistributions of source code must retain the above copyright

%    notice, this list of conditions and the following disclaimer.

% 2. Redistributions in binary form must reproduce the above copyright

%    notice, this list of conditions and the following disclaimer in the

%    documentation and/or other materials provided with the distribution.

% 

% THIS SOFTWARE IS PROVIDED BY Sergio Cruces ''AS IS'' AND ANY EXPRESS 

% OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED

% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE

% DISCLAIMED. IN NO EVENT SHALL Sergio Cruces BE LIABLE FOR ANY

% DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 

% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 

% OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 

% HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 

% STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 

% IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 

% POSSIBILITY OF SUCH DAMAGE.

%-------------------------------------------------------------------------

 

%% Setting the initial parameters

[M,T]=size(x); 

if ~exist('n')          n=M;            end; % Number of columns of A

if ~exist('MaxIter')    MaxIter=500;   end; % Max. mumber of iterations

check=min(20,MaxIter);                       % Min. number of iterations

stop=1e-7;                                   % Stopping value

one=ones(n,1);                               % Vector of ones

 

%% Whitening of the observations

[W,Rank]=whitening(x,n); m=Rank;

z=W*x; 

 

%% Initialization of the whitened mixing system A

A=1/sqrt(2)*(randn(m,n)+1i*randn(m,n)); % Initializes A. 

 

%% Determine the set of homogenously spaced projections

if ~exist('H')                        

    q=15*(2*m*n);                     % Number of projections

    H=randn(m,q)+1i*randn(m,q);       % Initializes H randomly

else

    q=size(H,2);

end

B= H./repmat(sqrt(sum(abs(W'*H).^2)),m,1); 

B_=pinv(B);

 

L=perimeter(B,z);                     % Evaluation of the perimeters

G=A'*B;                               % Global transfer matrix G

[Cost(1),Laverage(1),e]=EvalCost(G,L);% Evaluates the initial cost & error

 

%% Main iteration

i=1;

Continue=1; 

while Continue && (i<=MaxIter)

    i=i+1;        

    SignG=sign(G);                      % Complex-sign of G (global matrix)

     

    %% Exact solution when SignG is accurate.

    for ind=1:q

        MH(:,ind)=kron(conj(SignG(:,ind)),B(:,ind));

    end    

    if Laverage(i-1)>=0 

        PMH=MH-repmat(mean(MH,2),1,q);  % Removes the average mismatch.

    end

    Aest1=reshape(pinv(PMH')*L,m,n);    % First candidate solution

    G1=Aest1'*B;

    [Cost1(i),Laverage1(i),e1]=EvalCost(G1,L);

    

    %% Evaluation of the Gradients

    GradG=-2*(SignG'.*(e*one.'));       % Gradient with respect to G^T      

    NGradA=B_'*GradG;                   % Natural gradient descent wrt A^* 

    PGrad=B'*NGradA;                    % Proyected gradient: P_B'*GradG   

    

    %% Backtracking & Natural Gradient Descent  

    factor=.7; c=.1; 

    NormPGrad2=norm(PGrad,'fro')^2;

    if i==2, eta=1;

    else     eta=eta/(factor^2);

    end

    mu=Cost(i-1)/(1e-7+2*NormPGrad2);   % Normalized stepsize

    Cost2(i)=Cost(i-1)+1e5;             % Initializes Cost2(i)>Cost(i-1)

    while (Cost2(i)> Cost(i-1) - c*mu*eta*NormPGrad2) && eta>1e-3       

        Aest2=A-eta*mu*NGradA;          % Second candidate solution  

        G2=Aest2'*B;

        [Cost2(i),Laverage2(i),e2]=EvalCost(G2,L);

        eta=eta*factor;

    end

    

    %% Choosing the best update

    if Cost1(i)<Cost2(i)

        A=Aest1; G=G1; e=e1;

        Cost(i)=Cost1(i);

        Laverage(i)=Laverage1(i);

    else

        A=Aest2; G=G2; e=e2;

        Cost(i)=Cost2(i);

        Laverage(i)=Laverage2(i);

    end

    

    %% Checking for the convergence

    if i>check

        Dist=min(check,40);

        Continue=std(Cost(i-Dist:i)/T)>stop;

    end    

end

 

A=pinv(W)*A; % Estimate of the mixing matrix, 

             % up to an arbitray scaling & permutation of the columns.
S=pinv(A)*x;
 

 

%%------------------------------------------------------------------------

 

function [W,m]=whitening(x,n);  

% Estimates the whitening matrix W and its rank m.

 

M    =size(x,1);

[Q,D]=eig(cov(x'));

[v,i]=sort(diag(D)); vx=flipud([v,i]);

if M>n, vn=mean(vx(n+1:M,1)); % Estimated noise power

else    vn=0;                 % Estimated noise power

end

m=min(M,n); % m =rank{W}.

D=diag(sqrt(max(vx(1:m,1)-vn,eps)));

W=pinv(D)*Q(:,vx(1:m,2))'; % whitening matrix

 

%%------------------------------------------------------------------------

 

function [Cost,Laverage,e]=EvalCost(G,L)

% Evaluates the cost function.

% Returned variables:

% Cost      = cost function

% Laverage  = average mismatch in the perimeter.

% e         = centered mismatch.

 

e=L-8*sum(abs(G),1)';

Laverage=max(mean(e),0);  
% Laverage=0; 
e=e-Laverage;

Cost=norm(e)^2;

 

%%------------------------------------------------------------------------

 

function L=perimeter(B,x)

% Returns L, the vector of perimeters of projections of the observations

 

q=size(B,2);

for i=1:q

    y=B(:,i)'*x;

    k = convhull(real(y),imag(y));

    L(i,1)=sum(abs(y(k(2:length(k)))-y(k(1:(length(k)-1)))));

end

 

%%-END OF BCAPM-----------------------------------------------------------

 

 

function H=SolveTammesProblem(m,q,MaxIter);

% Creates a matrix H whose q-columns homogeneously sample the complex

% hypersphere S^{m-1}.

%

% (c) Sergio Cruces 2014, sergio@us.es

%

%-OUTPUTS-----------------------------------------------------------------

% H     = (m x q) matrix which almost homogeneously samples S^{m-1}(C).

%

%-INPUTS------------------------------------------------------------------

% m     = 1st dimension of H 

% q     = 2nd dimension of H (number of directions)

% MaxIter = number of iterations (depends on problem dimensionality).

%

%-EXAMPLE OF USE---------------------------------------------------------

%  H = SolveTammesProblem(3,100); % H will be 3x100.

%  H = SolveTammesProblem(2,50);  % H will be 2x50.

%-------------------------------------------------------------------------

 

complex=1;                % 0-real case, 1-complex case

show=~complex;            % can display the configuration in the real case

if ~exist('MaxIter') MaxIter=50; end;          % number of iterations

H=(randn(m,q)+complex*1i*randn(m,q));          % initialization

H=H./repmat(sqrt(sum(abs(H).^2,1)),m,1);       % normalization

H(:,q+(1:q))=-H;                               % include opposite vectors

Hini=H;

 

% Main Loop

for i=1:MaxIter

    for k=1:q

        iprod=real(H'*H(:,k));                 % inner product

        iprod(k)=0;                            % excludes the ith vector

        [maxiprod,j]=max(iprod);               % finds the nearest neighbor

        Hnn=H(:,j);                            % nearest neighbor

      

        mu=.1*10^(-2*i/MaxIter);

        grad=-(Hnn-H(:,k)'*Hnn*H(:,k));

        vector=H(:,k)+mu*grad/norm(grad);      % gradient descent

        SmallPerturbation=mu/100*(randn(m,1)+complex*1i*randn(m,1));

        vector=vector+SmallPerturbation;       % adds stoch. perturbation

        H(:,k)=vector/norm(vector);            % renormalizes

        H(:,k+q)=-H(:,k);                      % includes the opposite        

    end

end

 

if show

    clf

    show_sampled_sphere(Hini,221);title('RANDOM CONFIGURATION')

    show_sampled_sphere(H,223);title('FINAL CONFIGURATION')

    drawnow

end

 

H=H(:,1:q);     % Preserves only one vector for each direction.

 

 

function show_sampled_sphere(H,subplotind)

% Plots the configuration of directions which sample the hypersphere.

% With complex data it plots only the real part so the equispacing cannot

% be appreciated in this case.

 

Hr=real(H);

if size(Hr,1)<3 || size(Hr,2)<3

    subplot(subplotind);

    hold on

    for in=1:size(Hr,2);

        plot([0, Hr(1,in)],[0, Hr(2,in)],'-k');

    end

    hold off

    axis equal;

else % m=3

    subplot(subplotind);

    hold on;

    for in=1:size(Hr,2);

        plot3([0,Hr(1,in)],[0,Hr(2,in)],[0,Hr(3,in)],'k-');

    end;

    view (135,30); grid

    axis equal;

    hold off;

    subplot(subplotind+1);

    [K,vx]=convhull(Hr(1:3,:)');

    trisurf(K,Hr(1,:)',Hr(2,:)',Hr(3,:)',1);

    axis equal;

    view (135,30);

end

 

% END of SolveTammesProblem-----------------------------------------------