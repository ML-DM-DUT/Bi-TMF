function [ Corrs ] = CosLCorr( Y )
% CosLCorr: calculate the label correlation using Cosine distance.
% Input Y: the multi-label matrix (NxC), C is the number of labels, in 1
% and -1 form, 1 means labeled and -1 donates unlabeled.
% Output: Corrs, C x C correlation matrix
%   written by Guoxian Yu (guoxian85@gmail.com), School of Computer Science and Engineering,
%   South China University of Technology.
%   version 1.0 date:2011-12-11
[N,C]=size(Y);
% Corrs=ones(C,C);
Corrs=zeros(C,C);
for ii=1:C
    Yii=Y(:,ii);
    index=find(Yii<=0);
    Yii(index)=0;
    for jj=ii+1:C
        Yjj=Y(:,jj);
        index=find(Yjj<=0);
        Yjj(index)=0;
        temp=sqrt(sum(Yii)) * sqrt(sum(Yjj));
        if(temp==0)
            temp=1;
        end
        Corrs(ii,jj)=((Yii')*Yjj)/temp;
    end
end
% Corrs=Corrs+eye(C,C);%self correlation is 1;
% Corrs=max(Corrs,Corrs);
Corrs=Corrs+eye(C,C);%self correlation is 1;
Corrs=max(Corrs,Corrs');


