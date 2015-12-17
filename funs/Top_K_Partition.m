function [newZ]=Top_K_Partition(Z,topK)
%partite the largest k elements as relevant and the left as irrelavant
%input  Z(Ndata x Nfun): predicted probability 
%       topK: the top K element relevant labeled as 1
%output newZ(Ndata x Nfun): partitioned relevant and irrelevant labels (1) for relevant and (-1) for irrelevant
%   written by Guoxian Yu (guoxian85@gmail.com), School of Computer Science and Engineering,
%   South China University of Technology.
%   version 1.0 date:2012-01-08
[Ndata,Nfun]=size(Z);
newZ=-ones(Ndata,Nfun);
for ii=1:Ndata
    [sorted, index] = sort(Z(ii,:),'descend');
    newZ(ii,index(1:topK))=1;
end