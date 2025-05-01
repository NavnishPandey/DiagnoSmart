import pytest
from data_generation.generate_data import introduce_typos

def test_typo_injection_probability():
    original = "This is a simple sentence"
    typo_version = introduce_typos(original, prob=1.0)
    assert original != typo_version  # Should have typo

def test_typo_preserves_word_count():
    sentence = "This is a test"
    modified = introduce_typos(sentence, prob=0.5)
    assert len(sentence.split()) == len(modified.split())  # Should keep same word count