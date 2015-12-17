% A framework for incorporating functional inter-relationships into Protein
% Function Prediction Algorithms, IEEE/ACM Trans on Computational Biology
% and Bioinformatics, 2012. Xiaofei Zhang and Daoqing Dai
%   written by Guoxian Yu (guoxian85@gmail.com), School of Computer Science and Engineering,
%   South China University of Technology.
%   version 1.0 date:2012-02-13

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
Kernels=cell(K,1);
CKernel=zeros(Ndata,Ndata);
for ii=1:K
    Kernel=W{ii};
    Kernel=Kernel-diag(diag(Kernel));
    Kernels{ii}=Kernel;
    CKernel=CKernel+Kernel/K;
end
clear yMat Shuffle_Index L W;

LabRatios=0.2:0.1:0.8;
tol=0.000001;
g_I=0.001;
g_K=0.000001;
round=10;

RankingLosses=zeros(length(LabRatios),round);
AveragePrecisions=zeros(length(LabRatios),round);
Coverages=zeros(length(LabRatios),round);
AUCs=zeros(length(LabRatios),round);

for lab_idx=1:length(LabRatios)
    lab_ratio=LabRatios(lab_idx);
    lab_num=fix(lab_ratio*Ndata3);
    for run=1:round
        [tr_idx,te_idx]=gen_train_test(Y,lab_ratio); 
        random_idx=[tr_idx;te_idx];
        Random_W=CKernel(random_idx,random_idx);
        Random_W=(Random_W+Random_W')/2;
        Random_Y=Y(random_idx,:);
        
        W=Random_W;
       
        %validate on the functions that exist in both training and testing dataset
        tempY=(abs(Random_Y)+Random_Y)/2;
        fun_idx1=find(sum(tempY(1:lab_num,:),1)>0);
        fun_idx2=find(sum(tempY(lab_num+1:Ndata,:),1)>0);
        fun_idx=intersect(fun_idx1,fun_idx2);
        Nfun2=length(fun_idx);
        tempY=tempY(:,fun_idx);
        
        Temp_F=(abs(Random_Y)+Random_Y)/2;%transform 1,-1 to 1,0
        Temp_F=Temp_F(:,fun_idx);
        Temp_F(lab_num+1:Ndata,:)=0;
        %get the correlation matrix
        Corr=JaccorbLCorr(Temp_F);
       
        %normalize correlation matrix
        for ii=1:Nfun2
            Corr(ii,:)=Corr(ii,:)/sum(Corr(ii,:));
        end
        
        U=zeros(Ndata,1);
        U(1:lab_num)=1;
        U=diag(U);%indicative matrix for annotated proteins
       
        %using manifold regularization
        I=eye(Ndata,Ndata);%identity matrix
        Lap=I-(diag(sum(W,2))+tol*eye(size(W)))\W;%regularized laplacian matrix
        
        Z=(U+g_K*I+g_I*Lap)\(U*Temp_F*Corr);%predicting functions
       
        Z=diag(sum(Z,2)+tol*ones(Ndata,1))\Z;
       
        rocZ=Z;

        te_Y=tempY(lab_num+1:Ndata,:);
        rocZ=rocZ(lab_num+1:Ndata,:);
        
        index=find(sum(te_Y,2)==0);
        te_Y(index,:)=[];
        rocZ(index,:)=[];
        newY=2*te_Y-ones(size(te_Y));
        
        fprintf('==GRF_CK Run=%d,lab_ratio=%-10.4f, Nfun2=%d, time:%s\n',run,lab_ratio,Nfun2,datestr(now));
        RankingLosses(lab_idx,run) = 1-Ranking_loss(rocZ',newY');
        AveragePrecisions(lab_idx,run) = Average_precision(rocZ',newY');
        Coverages(lab_idx,run)=coverage(rocZ',newY');
        [tpr,fpr] = mlr_roc(rocZ, newY);
        [AUC, area2] = mlr_auc(fpr,tpr);
        AUCs(lab_idx,run)=AUC;
        fprintf('Multi Label Metric: %s\n',datestr(now));
        fprintf('1-RankingLoss=%-10.4f, AveragePrecision=%-10.4f, Coverage=%-10.4f, AUC=%-10.4f\n\n',...
            RankingLosses(lab_idx,run), AveragePrecisions(lab_idx,run), Coverages(lab_idx,run),  AUCs(lab_idx,run));
    end %end of for run=1:round
end%end for lab_idx=1:length(LabRatios)

prec_seq='RankingLoss, AveragePrecision, Coverage, AUC';
precision=cell(8,1);

precision{1}=sum(RankingLosses,2)/round;
precision{2}=sum(AveragePrecisions,2)/round;
precision{3}=sum(Coverages,2)/round;
precision{4}=sum(AUCs,2)/round;
precision{5}=RankingLosses;
precision{6}=AveragePrecisions;
precision{7}=Coverages;
precision{8}=AUCs;
   
stds=cell(4,1);
stds{1}=std(RankingLosses,0,2);
stds{2}=std(AveragePrecisions,0,2);
stds{3}=std(Coverages,0,2);
stds{4}=std(AUCs,0,2);

evalstr=['save  results',filesep,dataname, '_GRF_CK_Ls.mat g_I g_K LabRatios precision stds prec_seq'];
eval(evalstr);

fprintf('\n =====%s finish GRF_CK_Ls time=%s\n',dataset,datestr(now));



