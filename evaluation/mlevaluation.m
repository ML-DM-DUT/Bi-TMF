function [ result ] = slevaluation( Outputs, Pre_Labels,test_target)
%SLEVALUATION Calculate the evaluation values under various criteria
% These multi-label evaluation criteria include: 
%       . ranking loss;
%       . average precision;
%       . hamming loss
%       . micro-F1
% $ Arguments $
%   - Outputs:       The confidence output for testing samples: cnum x |U| matrix, each sample takes a column.
%   - Pre_Labels:    The label set predicted for testing samples: cnum x |U| matrix (+1/-1)
%   - test_target:   The ground truth label set for testing samples: cnum x |U| matrix (+1/-1)
%   
% $ Syntax $
%   - [ result ] = slevaluation( Outputs, Pre_Labels,test_target...)

% ranking loss
    result.RL=ranking_loss(Outputs,test_target);
% average precision
    result.A=average_precision(Outputs,test_target);
% hamming loss
    result.H=hamming_loss(Pre_Labels,test_target);
% micro-F1
    [result.F]=microF1(Pre_Labels,test_target);

% result summary
 result.text =[ ...
        '  -RL:' num2str(result.RL) ...
        '  -A:' num2str(result.A)...
        '  -H:' num2str(result.H) ...
        '  -F:' num2str(result.F)...
        ];
    disp(['result:' result.text]);
