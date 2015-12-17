function [tr_idx,te_idx] = gen_train_test(gnd,ratio)
%gen_train_test divide training and testing set
%Input: gnd: NxC label matrix
%       ratio:the pecentage of training set in X
%Ouput: tr_idx are training index
%       te_idx are testing index
%Note: We make sure each label have at least one instances
%   written by Guoxian Yu (guoxian85@gmail.com), School of Computer Science and Engineering,
%   South China University of Technology.
%   version 1.0 date:2011-11-12

minval=min(min(gnd));
if(minval<0)
   gnd=(gnd+abs(gnd))/2;%transform to 1 0 form
end

if ratio>=1 || ratio<=0
    error('wrong split of training and testing samples!!!');
end
[N,C]=size(gnd);
lab_total=length(find(sum(gnd,2)>0));
N1=fix(lab_total*ratio);
if N1<C
    disp('ratio is too small to ensure each label have one instance!!!');
    N1=C;
end
id1=zeros(C,1);
for i=1:C%ensure each label have at least one instance
    index=find(gnd(:,i)==1);
    rand_idx=randperm(length(index));
    j=1;
    id=index(rand_idx(j));
    while (length(find(id1==id))==1) && (j<length(rand_idx)) 
        j=j+1;
        id=index(rand_idx(j));
    end
    id1(i)=id; 
end

id2=find(sum(gnd,2)==0);%some instances may not have labels
index=setdiff(1:N,id1);
index=setdiff(index,id2);%left the unlabeled instances in testing set
if N1>C
    if length(index)>N1-C
        rand_idx=randperm(length(index));
        id3=index(rand_idx(1:N1-C));
        tr_idx=[id1;id3'];
    else
        tr_idx=[id1;index'];
    end
else
    tr_idx=id1;
end
tr_idx=sort(tr_idx);
te_idx=setdiff(1:N,tr_idx)';


