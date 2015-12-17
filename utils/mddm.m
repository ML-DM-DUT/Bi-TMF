classdef mddm
   properties
       P=[];
       eigvals=[];
   end

   methods
      function s = train(s, X,labels)
            [s.P s.eigvals] =s.maxhsic(X,labels);
      end
      function Y = apply(s, X)
            k = sldim_by_eigval(s.eigvals,'rank');
            Y = s.P(:, 1:k)' * X;
      end
   end
   methods (Static)
       function [P evals] = maxhsic(X,L)
            [~,N] = size(X);
            tS=X*L'-X*repmat(ones(1,N)*L',N,1)/N;
            S=tS*tS';
            [tmp_P tmp_evals] = eig(S);
            c_evals = diag(tmp_evals);
            [evals order] = sort(real(c_evals), 'descend');
            P = tmp_P(:,order);
       end
   end
end
