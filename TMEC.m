%   Transductive Multi-label Ensemble Classication on Multiple Kernels
%  by Guoxian Yu (guoxian85@gmail.com), School of Computer and Information Science,
%  Southwest University, Chongqing, China.
%   version 2.0 date:2012-12-11

clear;
datapath=[pwd,filesep,'data',filesep];%pwd is the current work directory
addpath(datapath);

funspath=[pwd,filesep,'funs',filesep];
addpath(funspath);

dataname='human';
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

LabRatios=0.2:0.1:0.8;
alpha=0.01;
beta=0.8;
gamma=0.3;
tol=0.000001;
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
        Random_Y=Y(random_idx,:);
        
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
        Corr=CosLCorr(Temp_F);
        
        EnsZ=zeros(Ndata,Nfun2);
        for k_idx=1:length(Kernels)
            Kernel=Kernels{k_idx};
            Random_W=Kernel(random_idx,random_idx);
            Random_W=(Random_W+Random_W')/2;
            
            D_p=diag(sum(Random_W,2));
            P_p=(D_p+tol*eye(size(D_p)))\Random_W;
            D_f=diag(sum(Corr,2));
            P_f=(D_f+tol*eye(size(D_f)))\Corr;
            D_pf=diag(sum(Temp_F,2));
            D_fp=diag(sum(Temp_F,1));
            %         P_pf=(sqrt(D_pf))\Temp_F/(sqrt(D_fp));
            P_pf=(sqrt(D_pf)+tol*eye(size(D_pf)))\Temp_F/(sqrt(D_fp)+tol*eye(size(D_fp)));
            P_np=[(1-beta)*P_p, beta*P_pf;zeros(Nfun2,Ndata),(1-beta)*P_f];%directed Bi-Graph
            
            Temp_Y=zeros(Ndata+Nfun2,Nfun2);
            for ii=1:Nfun2
                temp=sum(Temp_F(1:Ndata,ii));
                if(temp==0)
                    temp=1;
                end
                Temp_Y(1:Ndata,ii)=gamma*Temp_F(1:Ndata,ii)/temp;
                Temp_Y(Ndata+ii,ii)=1-gamma;
            end
            
            temp=eye(Ndata+Nfun2,Ndata+Nfun2)-(1-alpha)*P_np;
            Z=temp\Temp_Y;%predict the functions
            Z(Ndata+1:Ndata+Nfun2,:)=[];
            Z=diag(sum(Z,2)+tol*ones(Ndata,1))\Z;
            rocZ=Z;
    
            EnsZ=EnsZ+Z;
          
            te_Y=tempY(lab_num+1:Ndata,:);
            rocZ=rocZ(lab_num+1:Ndata,:);
            
            index=find(sum(te_Y,2)==0);
            te_Y(index,:)=[];
            rocZ(index,:)=[];
            newY=2*te_Y-ones(size(te_Y));
 
            fprintf('== TMEC Single Run=%d,lab_ratio=%-10.4f, Kernel={%d}, Nfun2=%d, time:%s\n',run,lab_ratio,k_idx, Nfun2,datestr(now));
            RankingLoss = 1-Ranking_loss(rocZ',newY');
            AveragePrecision = Average_precision(rocZ',newY');
            Coverage= coverage(rocZ',newY');
            [tpr,fpr] = mlr_roc(rocZ, newY);
            [AUC, area2] = mlr_auc(fpr,tpr);
            fprintf('Multi Label Metric: %s\n',datestr(now));
            fprintf('1-RankingLoss=%-10.4f, AveragePrecision=%-10.4f, Coverage=%-10.4f, AUC=%-10.4f\n',...
                RankingLoss, AveragePrecision,Coverage, AUC);
        end %for k_idx=1:length(Kernels)
        
        %Ensemble Kernel PERFORMANCE EVALUATION
        EnsZ=EnsZ/length(Kernels);
        EnsZ=diag(sum(EnsZ,2)+tol*ones(Ndata,1))\EnsZ;
        rocZ=EnsZ;
        
       
        te_Y=tempY(lab_num+1:Ndata,:);
        rocZ=rocZ(lab_num+1:Ndata,:);
        
        index=find(sum(te_Y,2)==0);
        te_Y(index,:)=[];
        rocZ(index,:)=[];
        newY=2*te_Y-ones(size(te_Y));
       
       
        fprintf('\n== TMEC Ensemble Run=%d,lab_ratio=%-10.4f, Nfun2=%d, time:%s\n',run,lab_ratio,Nfun2,datestr(now));
        RankingLosses(lab_idx,run) = 1-Ranking_loss(rocZ',newY');
        AveragePrecisions(lab_idx,run) = Average_precision(rocZ',newY');
        Coverages(lab_idx,run)=coverage(rocZ',newY');
        [tpr,fpr] = mlr_roc(rocZ, newY);
        [AUC, area2] = mlr_auc(fpr,tpr);
        AUCs(lab_idx,run)=AUC;
        fprintf('Multi Label Metric: %s\n',datestr(now));
        fprintf('1-RankingLoss=%-10.4f, AveragePrecision=%-10.4f, Coverage=%-10.4f, AUC=%-10.4f\n\n',...
            RankingLosses(lab_idx,run), AveragePrecisions(lab_idx,run), Coverages(lab_idx,run),  AUCs(lab_idx,run));
    end %end for run=1:round
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

evalstr=['save D:\TMEC\TMEC_Web\results',filesep,dataname, '_TMEC_MK_1211.mat gamma beta LabRatios precision stds prec_seq'];
eval(evalstr);

fprintf('\n =====%s finish TMEC_MK_Ls time=%s\n',dataset,datestr(now));

