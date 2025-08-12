import random
import pytest

# Replace `your_module` with the actual module name where random_words is defined.
from language import random_words


def test_empty_alphabet_raises():
    with pytest.raises(ValueError, match="Alphabet A must be non-empty."):
        random_words([], n=3, k=2)


@pytest.mark.parametrize("n,k", [(-1, 1), (1, -1), (-5, -2)])
def test_negative_n_or_k_raises(n, k):
    with pytest.raises(ValueError, match="n and k must be non-negative integers."):
        random_words([0, 1], n=n, k=k)


def test_zero_k_returns_empty_list():
    out = random_words([0, 1, 2], n=4, k=0)
    assert out == []


def test_zero_length_words_are_empty_lists():
    out = random_words(['a', 'b'], n=0, k=3)
    # Expect k empty lists
    assert len(out) == 3
    assert all(isinstance(w, list) for w in out)
    assert all(len(w) == 0 for w in out)


@pytest.mark.parametrize("alphabet,n,k", [
    (['a', 'b', 'c'], 5, 7),
    ((0, 1), 3, 4),
    (range(5), 10, 2),
])
def test_lengths_and_membership(alphabet, n, k):
    out = random_words(alphabet, n=n, k=k)
    # Correct number of words
    assert len(out) == k
    # Each word is a list of length n and elements from A
    A = set(alphabet)
    for w in out:
        assert isinstance(w, list)
        assert len(w) == n
        assert set(w).issubset(A)


def test_single_symbol_alphabet_degenerates_to_constant_words():
    random.seed(123)
    out = random_words(['x'], n=5, k=3)
    assert out == [['x'] * 5, ['x'] * 5, ['x'] * 5]


def test_reproducibility_with_seed():
    # Since random.choices uses the global RNG, seeding ensures determinism
    alphabet = ['a', 'b', 'c']
    n, k = 6, 4

    random.seed(42)
    out1 = random_words(alphabet, n=n, k=k)

    random.seed(42)
    out2 = random_words(alphabet, n=n, k=k)

    assert out1 == out2