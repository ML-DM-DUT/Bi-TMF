
clear;
datapath=[pwd,filesep,'data',filesep];%pwd is the current work directory
addpath(datapath);

funspath=[pwd,filesep,'funs',filesep];
addpath(funspath);

dataname='yeast';
dataset=[dataname,'_multigraph.mat'];
fprintf('%s start %s\n',datestr(now),dataset);
load(dataset);
Y=2*yMat-ones(size(yMat));
Ndata3=length(find(sum(yMat,2)>0));

[Ndata,Nfun]=size(Y);
K=length(W);
Kernels=cell(length(W),1);
for ii=1:K
    Kernel=W{ii};
    Kernel=Kernel-diag(diag(Kernel));
    Kernels{ii}=Kernel;
end
clear yMat Shuffle_Index L W; 

LabRatios=0.8;
[m,n]=size(Y);
alpha=0.01
round=10;
tol=0.000001;
beta=0.8

Confidence=zeros(length(LabRatios),round);
AveragePrecisions=zeros(length(LabRatios),round);
Coverages=zeros(length(LabRatios),round);
% MicroF1=zeros(length(LabRatios),round);

for lab_idx=1:length(LabRatios)
    lab_ratio=LabRatios(lab_idx);
    lab_num=fix(lab_ratio*Ndata3);
    for run=1:round
        [tr_idx,te_idx]=gen_train_test(Y,lab_ratio);
         random_idx=[tr_idx;te_idx];
        Random_Y=Y(random_idx,:);
          
        tempY=(abs(Random_Y)+Random_Y)/2;
        fun_idx1=find(sum(tempY(1:lab_num,:),1)>0);
        fun_idx2=find(sum(tempY(lab_num+1:Ndata,:),1)>0);
        fun_idx=intersect(fun_idx1,fun_idx2);
        Nfun2=length(fun_idx);
        n=Nfun2;
        tempY=tempY(:,fun_idx);
        %tempY=[tempY;eye(n,n)];Function matrix becomes£¨m+n£©*n
        
        Temp_F=(abs(Random_Y)+Random_Y)/2;%transform 1,-1 to 1,0
        Temp_F=Temp_F(:,fun_idx);
        Temp_F(lab_num+1:Ndata,:)=0;
        
        %Obtain the correlation matrix between function
        Corr=CosLCorr(Temp_F);
        D_f=diag(sum(Corr,2));
        P_f=(D_f+tol*eye(size(D_f)))\Corr;
        D_pf=diag(sum(Temp_F,2));
        D_fp=diag(sum(Temp_F,1));
        P_pf=(sqrt(D_pf)+tol*eye(size(D_pf)))\Temp_F/(sqrt(D_fp)+tol*eye(size(D_fp)));
        
        train_label=tempY(1:lab_num,:)';
        train_num=lab_num;
        test_num=m-lab_num;
        numlab=sum(train_label,1);
        train_label=train_label*spdiags(1./sum(train_label,1)',0,train_num,train_num);
        WZ=zeros(m+n,m+n);
        P=zeros(m+n,m+n);%Set up(M+N)dimension
        b=zeros(m-train_num,Nfun);
        bt=0;
        ANN=zeros(m-train_num,m-train_num);
        for k_idx=1:length(Kernels)
         Kernel=Kernels{k_idx};
         Random_W=Kernel(random_idx,random_idx);
         Random_W=(Random_W+Random_W')/2;
         P=[Random_W,tempY;zeros(n,m),P_f];
         A = speye(m+n,m+n)-P; 
         ANN0 = A(train_num+1:m,train_num+1:m);
         ANL = A(train_num+1:m,1:train_num);
         b1=-ANL*train_label';
         b=b+b1;
         bt1=-ANL*numlab';
         bt=bt+bt1;
         ANN=ANN+ANN0;
         end
        b=b/length(Kernels);
        bt=bt/length(Kernels);
        ANN=ANN/length(Kernels);
        
        
        
        
%             for k_idx=1:length(Kernels)
%             Kernel=Kernels{k_idx};
%             Random_W=Kernel(random_idx,random_idx);
%             Random_W=(Random_W+Random_W')/2;
%             P=[Random_W,tempY;zeros(n,m),P_f];
%             %P=[Random_W,tempY];
%             WZ=WZ+P;
%             end           
%             
%             W=WZ/length(Kernels);
%         A = speye(m+n,m+n)-W; 
%        ANN = A(train_num+1:m,train_num+1:m);
%        ANL = A(train_num+1:m,1:train_num);
       
%        b=-ANL*train_label';
%        bt=-ANL*numlab';

Confidence=lscov(ANN,b)';
card=lscov(ANN, bt)';

card=floor(card+0.5);
[~,IX]=sort(Confidence,1,'descend');
Pre_Labels=-ones(size(Confidence));
            for x=1:test_num
                 Pre_Labels(IX(1:card(x),x),x)=1;
            end
       te_Y=tempY(lab_num+1:Ndata,:);
index=find(sum(te_Y,2)==0);
te_Y(index,:)=[];
test_target=(2*te_Y-ones(size(te_Y)))';
Outputs=Confidence;
% ranking loss
    RankingLosses(lab_idx,run)=1-ranking_loss(Outputs,test_target);
% average precision
    AveragePrecisions(lab_idx,run)=average_precision(Outputs,test_target);
    Coverages(lab_idx,run)=coverage(Outputs,test_target);
%     MicroF1(lab_idx,run)=microF1(Pre_Labels,test_target);
    end  %end for
end  % end for lab_idx=1:length(LabRatios)


precision=cell(6,1);

precision{1}=sum(RankingLosses,2)/round;
precision{2}=sum(AveragePrecisions,2)/round;
precision{3}=sum(Coverages,2)/round;
precision{4}=RankingLosses;
precision{5}=AveragePrecisions;
precision{6}=Coverages;


stds=cell(3,1);
stds{1}=std(RankingLosses,0,2);
stds{2}=std(AveragePrecisions,0,2);
stds{3}=std(Coverages,0,2);   
% stds{4}=std(MicroF1,0,2);  

evalstr=['save D:\TMEC\TMEC_Web\results',filesep,dataname, '_BiTMF.mat LabRatios precision stds'];
eval(evalstr);   
    % hamming loss result.H=hamming_loss(Pre_Labels,test_target);
% micro-F1
    %[result.F]=microF1(Pre_Labels,test_target);
%     [ result ] = mlevaluation( Outputs, Pre_Labels,test_target); 




