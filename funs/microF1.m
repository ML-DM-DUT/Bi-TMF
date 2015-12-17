function [ F1 ] = microF1(X,Y)
X(X>0) = 1;X(X<=0) = 0;
Y(Y>0) = 1;Y(Y<=0) = 0;
XandY = X&Y;
Precision=sum(XandY(:))/sum(X(:));
Recall=sum(XandY(:))/sum(Y(:));
F1=2*Precision*Recall/(Precision+Recall);
