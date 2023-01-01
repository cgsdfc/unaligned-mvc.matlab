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
