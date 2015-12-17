function [Alphas, pre_Fs]= RegressionEnsemble(EnsF1,trY,teY,lambda)
N=size(trY,1);
Nt=size(teY,1);
numC=size(trY,2);
[EnsK]=size(EnsF1,1);
index=find(sum(trY,2)>0);%trY include the labeled and unlabeled
numL=length(index);
reY=trY(index,:);
tol=0.01;
Alphas=zeros(EnsK,numC);
if lambda<0.001
    lambda=tol;
end
for cc=1:numC
    tempY=reY(:,cc);
    tempF=EnsF1(:,index,cc);
%     tempF=repmat(EnsF1(:,index,cc),EnsK,numL);
    alpha=(tempF*tempF'+lambda*eye(EnsK,EnsK))\(tempF*tempY);
    Alphas(:,cc)=alpha;
end

pre_Fs=zeros(N+Nt,numC);
for cc=1:numC
    alpha=Alphas(:,cc);
    tempF=EnsF1(:,:,cc);
%     tempF=repmat(EnsF2(:,:,cc),Nt,EnsK);
    tempF=alpha'*tempF;
    pre_Fs(:,cc)=tempF;
end
