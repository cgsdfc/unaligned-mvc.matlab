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