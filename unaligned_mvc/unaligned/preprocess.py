# MIT License

# Copyright (c) 2022 Ao Li, Cong Feng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from pathlib import Path as P

from unaligned_mvc.data.unaligned import get_unaligned_permutations
from unaligned_mvc.data.dataset import MultiviewDataset
from unaligned_mvc.utils.metrics import MaxMetrics
from unaligned_mvc.utils.torch_utils import convert_tensor, nn, torch
from .config import args


class Preprocess(nn.Module):
    def forward(self):
        data = MultiviewDataset(
            datapath=P(args.datapath),
            view_ids=args.views,
            normalize="minmax",  # Note the alignment
        )
        logging.info("Loaded dataset {}".format(data.name))
        
        permutation_list = get_unaligned_permutations(
            n=data.sampleNum,
            V=data.viewNum,
            aligned_ratio=args.aligned_ratio,
        )['permutation_list']

        X = []
        for v in range(data.viewNum):
            P_gt = permutation_list[v]
            x = data.X[v].t()
            x = x[:, P_gt]
            X.append(x)

        res = dict(
            data=data,
            mm=MaxMetrics(ACC=True, NMI=True, PUR=True, F1=True),
            X=convert_tensor(X, torch.float, args.device),
            V=data.viewNum,
            n=data.sampleNum,
        )
        return res
