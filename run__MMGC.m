clear all;
clc;
close all;
load("HW2.mat")

% Better parameter setting can be found via grid-search.
parfor k = 1:10
    result_loop(k,:) = MMGC(data, labels, 10, 10, 3, 2, 50);
end
mean_acc = mean(result_loop(:,1));
mean_nmi = mean(result_loop(:,2));
std_acc = std(result_loop(:,1));
std_nmi = std(result_loop(:,2));

RES = [mean_nmi, mean_acc, std_nmi, std_acc];

save MMGC_result.mat
