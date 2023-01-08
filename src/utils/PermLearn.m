function [Perm align_gnd accL]= PermLearn(VS1,VS2,VS_gnd,VS2_gnd,t)
option = [];
option.KernelType = 'Gaussian';
% option.t = 1; %{'Gaussian','Polynomial','PolyPlus','Linear'}
option.t=t;
G = hungarian_cost(VS1,VS2,option);
[c,t] = hungarian(-G);
Perm = permut(c);
align_gnd = Perm*VS2_gnd;
accL = length(find(align_gnd==VS_gnd))/length(VS_gnd);