function [result, Loss,nmi] = MMGC(data, labels, gamma, phi, k, normData, max_iter)
% Metric Multi-view Subspace Clustering with Graph Filtering
% Par.gamma     :Trade-off parameter in LASC for penalty term ||S^v||_F^2
% Par.phi       :Trade-off parameter in Subspace Clustering for penalty term ||Z^v||_F^2
% Par.beta      :Trade-off parameter for the term of spectral clustering Tr(F^T(D-E)F)
% Par.k         : Order of the low-pass filter based on normalized Laplacian Fourier base 
% labels        : groundtruth of the data, num_samp by 1
% num_clus      : number of clusters
% num_view      : number of views
% num_samp      : number of samples
% Wv{v}         : Trade-off parameter for the term of Consensus graph learning ||Z-Z^v||_F^2
    if nargin < 3
        gamma = 1;
    end
    if nargin < 4
        phi = 1;
    end
    if nargin < 5
        k = 2;
    end
    if nargin < 6
        normData = 2;
    end
    if nargin < 7
        max_iter = 5;
    end
    num_view = size(data,1);
    num_samp = size(labels,1);
    num_clus = length(unique(labels));
    % beta = randperm(10,1);
%     beta = 100;
    beta = 0;
    zr = 10e-10;
    %
    % === Normalization1 ===
    if normData == 1
        for i = 1:num_view
            dist = max(max(data{i})) - min(min(data{i}));
            m01 = (data{i} - min(min(data{i})))/dist;
            data{i} = 2 * m01 - 1;
        end
    end
    % === Normalization2 ===
    if normData == 2
        for iter = 1:num_view
            for  j = 1:num_samp
                normItem = std(data{iter}(j,:));
                if (0 == normItem)
                    normItem = eps;
                end
                data{iter}(j,:) = (data{iter}(j,:) - mean(data{iter}(j,:)))/normItem;
            end
        end
    end
    %
    % === Initialization ===
    ZV = cell(num_view, 1);
    SV = cell(num_view, 1);
    DV2 = cell(num_view, 1);
    XV_bar = cell(num_view, 1);
    Wv = ones(num_view, 1) / num_view;   % initial Wv
    C = zeros(num_samp, num_samp);
    for v = 1:num_view 
        % === initialize Zv ===
        X = data{v};
        Zv = (X*X' + phi*eye(num_samp))\(X*X'); % initial Zv
        Zv = max(Zv,0);
        Zv = (Zv + Zv')/2;
        Zv = Zv - diag(diag(Zv));
        ZV{v} = Zv;
        C = C + Zv;
%         XV_bar{v} = X_bar;  % initial Xv_bar
    end
    C = C/num_view;
    Lc = diag(sum(C)) - C;
    % initialize F
    [F, ~, evs] = eig1(Lc, num_clus, 0);
    
    % ================== iteration ==================
    %fprintf('begin updating ......\n')
    iter = 0;
    bstop = 0;
    Loss = [];
%     nmi = [];
%     index = 1; % denote tsne index
%     W_tsne = cell(6,1);
    % for iter = 1: Iter
    while ~bstop
        iter = iter + 1;
        %fprintf('the %d -th iteration ......\n', iter);
        %
        % === update Sv ===
        for v = 1:num_view
            Zv = ZV{v};
            Dv = 1 - corr(Zv, 'type', 'Pearson');
            dv2 = Dv.*Dv;
            DV2{v} = dv2;
            hv = -(dv2-2*Wv(v)*C)/(2*(gamma+Wv(v)));
            hv = (hv + hv')/2;
           % Sv = Sv - diag(diag(Sv));
            Sv = zeros(num_samp, num_samp);
            for ic = 1:num_samp
                Sv(ic, :) = EProjSimplex_new(hv(ic, :));
            end
            Sv = (Sv + Sv')/2;
            SV{v} = Sv; 
        end
        %
        % === update Xv_bar ===
        for v = 1:num_view
            Xv_bar = data{v};
            if k > 0
               Sv = SV{v};
               Ls = diag(sum(Sv)) - Sv;
                for i = 1:k
                    Xv_bar = (eye(num_samp)-Ls/2)*Xv_bar;
                end
            end
            XV_bar{v} = Xv_bar;
        end
        %
        % === update C ===
        C = updateC(SV, F, Wv, beta, num_samp, num_view); 
        C = (C + C')/2;
        %
        % === update Wv ===
        for v = 1:num_view
            Sv = SV{v};
            temp = norm(C - Sv, 'fro');
            Wv(v) = 1/(2*temp + 1);
        end
        %
        % === update Zv ===
        for v = 1:num_view
            Xv_bar = XV_bar{v};
            Zv = (Xv_bar*Xv_bar' + phi*eye(num_samp))\(Xv_bar*Xv_bar');
            Zv = max(Zv, 0);
            Zv = (Zv + Zv')/2;
            Zv = Zv - diag(diag(Zv));
            ZV{v} = Zv;
        end
        %
        % === update F ===
        Lc = diag(sum(C)) - C;
        [F, ~, ev] = eig1(Lc, num_clus, 0);
        L1_loss = 0; L2_loss = 0; L3_loss = 0; 
        for v=1:num_view
            temp = DV2{v}.*SV{v};
            L1_loss = L1_loss + norm(XV_bar{v}' - (XV_bar{v}'*ZV{v}),'fro')^2 + phi*(norm(ZV{v},'fro')^2);
            L2_loss = L2_loss + gamma*(norm(SV{v},'fro')^2) + sum(temp(:));
            L3_loss = L3_loss + Wv(v)*(norm(C - SV{v},'fro')^2);
        end
        Loss(iter) = L1_loss + L2_loss + L3_loss; % + L4_loss;
       
        if((iter >= max_iter))
            bstop =1;
        end
% --Different criteria for ending loop.--
%         if (iter > 1) && ((iter > max_iter)||(abs(Loss(iter-1)-Loss(iter))/Loss(iter-1) <= 1e-6))
%             bstop = 1;
%         end
%         if (iter==1 || iter == 10 || iter ==20 || iter == 30 || iter == 40 || iter ==50 || bstop ==1)
%             W_tsne{index} = C;
%             index = index + 1; 
%         end
        W = abs(C) + abs((C)');
        [D] = SpectralClustering(W, num_clus);
        result_temp = EvaluationMetrics(labels, D);
        nmi(iter) = result_temp(2);
    end

    W = abs(C) + abs((C)');
    [D] = SpectralClustering(W, num_clus);
    result = EvaluationMetrics(labels, D);
    S_final = SV{num_view};
end

    function C = updateC(SV, F, Wv, beta, num_samp, num_view)
        dist = L2_distance_1(F', F');
        C = zeros(num_samp);
        for i = 1:num_samp
            zv0 = zeros(1, num_samp);
            for v = 1:num_view
                temp = SV{v};
                zv0 = zv0 + Wv(v)*temp(i,:);
            end
            idxa0 = find(zv0>0);
            zi = zv0(idxa0);
            ui = dist(i, idxa0);
            cu = (zi - 0.5*beta*ui)/sum(Wv);
            C(i,idxa0) = EProjSimplex_new(cu);
        end
    end