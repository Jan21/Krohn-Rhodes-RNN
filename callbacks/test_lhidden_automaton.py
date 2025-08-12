import pytest
import random
from hidden_automaton import (
    FiniteLanguage,
    finite_language_from_sequences,
    random_words,
)

# ---------- Tests for FiniteLanguage ----------

def test_valid_finite_language_construction():
    levels = [
        {""},             # L0
        {"a", "b"},       # L1
        {"aa", "ab", "ba", "bb"}  # L2
    ]
    alphabet = {"a", "b"}
    fl = FiniteLanguage(levels, alphabet)
    assert fl.levels == levels
    assert fl.alphabet == alphabet
    assert repr(fl) == f"FiniteLanguage(levels={levels})"


def test_invalid_level0_multiple_words():
    levels = [
        {"", "a"},  # invalid: more than one word in level 0
        {"a"}
    ]
    alphabet = {"a"}
    with pytest.raises(ValueError, match="Level 0 must contain only the empty string"):
        FiniteLanguage(levels, alphabet)


def test_invalid_forward_extension():
    levels = [
        {""},        # L0
        {"a","b"},
        {"ab"}        # L1 (missing extension for "b")
    ]
    alphabet = {"a","b"}
    with pytest.raises(ValueError, match="does not extend"):
        FiniteLanguage(levels, alphabet)


def test_invalid_backward_compatibility():
    levels = [
        {""},              # L0
        {"a"},             # L1
        {"aa", "bb"}       # L2 ("bb" has no prefix in L1)
    ]
    alphabet = {"a", "b"}
    with pytest.raises(ValueError, match="has prefix"):
        FiniteLanguage(levels, alphabet)


# ---------- Tests for finite_language_from_sequences ----------

def test_finite_language_from_sequences_valid():
    strings = ["ab", "aa"]
    fl = finite_language_from_sequences(strings)

    # All strings same length
    assert all(len(s) == 2 for s in strings)
    # Alphabet is {'a', 'b'}
    assert fl.alphabet == {"a", "b"}
    # Level 0 contains only ""
    assert fl.levels[0] == {""}
    # Level 1 contains 'a'
    assert "a" in fl.levels[1]
    # Level 2 contains the input strings
    assert set(strings).issubset(fl.levels[2])


def test_finite_language_from_sequences_empty_list():
    with pytest.raises(AssertionError, match="Input list is empty"):
        finite_language_from_sequences([])


def test_finite_language_from_sequences_unequal_length():
    with pytest.raises(AssertionError, match="must have the same length"):
        finite_language_from_sequences(["a", "bb"])


def test_finite_language_from_sequences_levels_correctness():
    strings = ["ab", "ac"]
    fl = finite_language_from_sequences(strings)

    # Expected levels
    expected_levels = [
        {""},
        {"a"},
        {"ab", "ac"}
    ]
    for lvl in expected_levels:
        for word in lvl:
            assert word in fl.levels[len(word)]
    assert fl.alphabet == {"a", "b", "c"}


# ---------- Tests for random_words ----------

def test_random_words_empty_alphabet_raises():
    with pytest.raises(ValueError, match="Alphabet A must be non-empty"):
        random_words([], n=3, k=2)


@pytest.mark.parametrize("n,k", [(-1, 1), (1, -1), (-5, -2)])
def test_random_words_negative_n_or_k_raises(n, k):
    with pytest.raises(ValueError, match="n and k must be non-negative integers"):
        random_words([0, 1], n=n, k=k)


def test_random_words_zero_k_returns_empty_list():
    out = random_words([0, 1, 2], n=4, k=0)
    assert out == []


def test_random_words_zero_length_words_are_empty_strings():
    out = random_words(['a', 'b'], n=0, k=3)
    assert len(out) == 3
    assert all(isinstance(w, str) and w == "" for w in out)


@pytest.mark.parametrize("alphabet,n,k", [
    (['a', 'b', 'c'], 5, 7),
    ((0, 1), 3, 4),
    (range(5), 10, 2),
])
def test_random_words_lengths_and_membership(alphabet, n, k):
    out = random_words(alphabet, n=n, k=k)
    # Correct number of words
    assert len(out) == k
    # Each word is of length n and made of characters from A (converted to str)
    A_str = set(map(str, alphabet))
    for w in out:
        assert isinstance(w, str)
        assert len(w) == n
        assert set(w).issubset(A_str)


def test_random_words_single_symbol_alphabet_degenerates_to_constant_words():
    random.seed(123)
    out = random_words(['x'], n=5, k=3)
    assert out == ['xxxxx', 'xxxxx', 'xxxxx']


def test_random_words_reproducibility_with_seed():
    alphabet = ['a', 'b', 'c']
    n, k = 6, 4

    random.seed(42)
    out1 = random_words(alphabet, n=n, k=k)

    random.seed(42)
    out2 = random_words(alphabet, n=n, k=k)

    assert out1 == out2