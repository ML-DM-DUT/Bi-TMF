function [ EnsZ ] = DecTempEnsemble(Zs,Y,lab_idx)
%function [ EnsZ ] = DecTempEnsemble(Zs,Y,lab_idx)
%combine basic classifiers with decision template
% See Paper: Decision Templates for multiple classifiers fusion: an experimental comparison, Pattern Recognition, Vol. 34, 2001.
[Ncls,Ndata,Nfun]=size(Zs);
EnsZ=zeros(Ndata,Nfun);
lab_num=length(lab_idx);
test_idx=setdiff(1:Ndata,lab_idx);
tempY=Y;
tempY(test_idx,:)=0;

%decision profile
Dp=zeros(Ndata,Ncls,Nfun);
for c_idx=1:Ncls
    for d_idx=1:lab_num
        Dp(lab_idx(d_idx),c_idx,:)=tempY(lab_idx(d_idx),:);
    end
    for d_idx=1:length(test_idx)
        Dp(test_idx(d_idx),c_idx,:)=Zs(c_idx,test_idx(d_idx),:);
    end
end

%decision template
Dt=zeros(Nfun,Ncls,Nfun);
for f_idx=1:Nfun
    index=find(tempY(:,f_idx)==1);%find training samples belong to f_idx class
    tempDp=Dp(index,:,:);
    Dt(f_idx,:,:)=sum(tempDp,1)/length(index);
end
S=zeros(length(test_idx),Nfun);

for f_idx = 1: Nfun
    for d_idx=1:length(test_idx)
        temp1=reshape(Zs(:,test_idx(d_idx),:),Ncls,Nfun);
        temp2=reshape(Dt(f_idx,:,:),Ncls,Nfun);
        temp=temp1-temp2;
        div=sum(sum(temp.^2));
        S(d_idx,f_idx)=1-div/(Nfun*Ncls);
    end
end
EnsZ=S;







