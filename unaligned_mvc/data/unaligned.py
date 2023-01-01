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

from pprint import pprint
import numpy as np
from sklearn.model_selection import train_test_split


def get_unaligned_permutations(n: int, V: int, aligned_ratio: float):
    """
    Args
        n: The number of samples.
        V: The number of views.
        aligned_ratio: in (0, 1), the ratio of aligned samples.
    Return
        permutation_list: A list of generated permutations for each view.
        idx_a: The indexes of the aligned samples.
        idx_u: The indexes of the unaligned samples.
    """
    assert 0 <= aligned_ratio <= 1
    idx_a, idx_u = train_test_split(
        list(range(n)), train_size=aligned_ratio, shuffle=True
    )
    idx_a.sort()
    idx_u.sort()
    permutation_list = [np.arange(n) for _ in range(V)]
    for v in range(V):
        idx_shuffle = np.copy(idx_u)
        np.random.shuffle(idx_shuffle)
        permutation_list[v][idx_u] = idx_shuffle

    return dict(
        permutation_list=permutation_list,
        idx_a=idx_a,
        idx_u=idx_u,
    )


if __name__=="__main__":
    res=get_unaligned_permutations(n=10, V=3, aligned_ratio=0.5)
    pprint(res)
