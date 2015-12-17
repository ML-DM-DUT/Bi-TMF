function [ Corrs ] = JaccorbLCorr( Y )
% JaccorbLCorr: calculate the label correlation using Jaccorbe distance.
% Input Y: the multi-label matrix (NxC), C is the number of labels, in 1
% and -1 form, 1 means labeled and -1 donates unlabeled.
% Output: Corrs, C x C correlation matrix
%   written by Guoxian Yu (guoxian85@gmail.com), School of Computer Science and Engineering,
%   South China University of Technology.
%   version 1.0 date:2011-12-11
[N,C]=size(Y);
Corrs=zeros(C,C);
for ii=1:C
    Yii=Y(:,ii);
    index=find(Yii<=0);
    Yii(index)=0;
    for jj=ii+1:C
        Yjj=Y(:,jj);
        index=find(Yjj<=0);
        Yjj(index)=0;
        part1=(Yii')*Yjj;
        part2=Yii+Yjj;
        part2(find(part2)==2)=1;
        Corrs(ii,jj)=part1/sum(part2);
    end
end
Corrs=Corrs+eye(C,C);
Corrs=max(Corrs,Corrs');

