% MIT License
% 
% Copyright (c) 2022 Ao Li, Cong Feng
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function [alignX,unalignX,align_gnd,unalign_gnd,align_sum] = unaligned(data,gnd,ratio) 
nnClus_min = min(unique(gnd));
nnClus_max = max(unique(gnd));
X = cell(1,length(data));
alignX = cell(1,length(data));
unalignX = cell(1,length(data));
align_sum = 0;
align_gnd = [];
unalign_gnd = [];
idx_rand = cell(1,length(data)-1);
for k = nnClus_min : nnClus_max
    idx = find(gnd==k);
    align_num = round(length(idx) * ratio);
    for v = 1:length(data)
        alignX{v}   =  [alignX{v}   data{v}(:,idx(1:align_num))];        
        unalignX{v} =  [unalignX{v} data{v}(:,idx(align_num+1:end))];
    end
    align_gnd   = [align_gnd;gnd(idx(1:align_num))];
    unalign_gnd = [unalign_gnd;gnd(idx(align_num+1:end))];
    align_sum = align_sum + align_num;
end
% %% Unaligned Operation for view 2 to end %%
% % With this method, we take the first view as the
% % baseline, which means that the truth lables(gnd) 
% % are corresponding to first view .
% for v = 2:length(data)
%     uda = unalignX{v};
%     idx_rand{v} = randperm(size(uda,2));
%     unalignX{v} = uda(:,idx_rand{v});
% end
%% Unaligned Data Generation %%
% for v = 1:length(data)
%     X{v} = [alignX{v} unalignX{v}];
% end
% gnd_new = [align_gnd;unalign_gnd];
