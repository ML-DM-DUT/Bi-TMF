function [newLabels]=transform_pos_neg(oldLabels)
%Transform the label 0 1 1 0 into -1 1 1 -1 forms
%oldLabels: the output of the ith instance for the jth class is stored in oldLabels(j,i)
%newLabels: if the ith instance belong to the jth class, newLabels(j,i)=1, otherwise newLabels(j,i)=-1)
%   written by Guoxian Yu (guoxian85@gmail.com), School of Computer Science and Engineering,
%   South China University of Technology.
%   version 1.0 date:2011-11-15
[L,N]=size(oldLabels);
newLabels=2*oldLabels-ones(L,N);