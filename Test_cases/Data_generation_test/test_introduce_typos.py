import pytest
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_generation.generate_data import introduce_typos

#This tests checks if the function actually works, so if it introduces typos
def test_typo_injection_probability():
    original = "This is a simple sentence"
    typo_version = introduce_typos(original, prob=1.0)
    assert original != typo_version  # Should have typo

#This tests checks if the behaviour of the function is correct, in particular it
#should not change the number of words in the sentence
def test_typo_preserves_word_count():
    sentence = "This is a test"
    modified = introduce_typos(sentence, prob=0.5)
    assert len(sentence.split()) == len(modified.split())  

#This tests checks that if prob = 0.0, introduce_typos returns the same sentence of input
def test_no_typo_when_prob_zero():
    text = "Clean sentence with no typos"
    modified = introduce_typos(text, prob=0.0)
    assert text == modified

#This tests checks what happens with a sentence of all words that have at most 3 chars
def test_no_typo_when_all_words_short():
    text = "The dog ran off"
    modified = introduce_typos(text,prob=1.0)
    assert text == modified

def test_strange_text_inputs():
    text = 3
    with pytest.raises(TypeError, match="Input 'text' must be a string."):
        introduce_typos(text, prob=1.0)
    text="" #empty string
    modified = introduce_typos(text,prob=1.0)
    assert text == modified
    text = "I paid $400 for that therapy" #special chars as $
    modified = introduce_typos(text,prob=1.0) 
    assert text != modified
    assert isinstance(modified, str)
    assert len(modified.split()) == len(text.split())
    text = "121 bones!" #special chars as ! and numbers
    modified = introduce_typos(text,prob=1.0) 
    assert text != modified
    assert isinstance(modified, str)
    assert len(modified.split()) == len(text.split())
    text = "èèèèèèèèèè" #unicode chars
    modified = introduce_typos(text,prob=1.0) 
    assert text != modified
    assert isinstance(modified, str)
    assert len(modified.split()) == len(text.split())
    text = "pneumonoultramicroscopicsilicovolcanoconiosis" #very long word
    modified = introduce_typos(text,prob=1.0) 
    assert len(modified) == len(text) + 1  # One char added
    assert text != modified
    assert isinstance(modified, str)
    assert len(modified.split()) == len(text.split())
    text = "this  has   multiple spaces" #double spaces
    modified = introduce_typos(text,prob=1.0) 
    assert text != modified
    assert isinstance(modified, str)
    assert len(modified.split()) == len(text.split())
    # Whitespace will be normalized by .split()
    assert "  " not in modified
    text = "this is a test sentence with multiple words"
    result = introduce_typos(text, prob=1.0)
    assert result != text  # At least some words should change
    # All words with len > 3 should be typoed
    original_words = text.split()
    modified_words = result.split()
    for o, m in zip(original_words, modified_words):
        if len(o) > 3:
            assert o != m

#If the input is a single word it should modify it duplicating some of its chars
def test_single_word_modification():
    result = introduce_typos("encyclopedia", prob=0.5)
    assert "encyclopedia" in result or len(result) == len("encyclopedia") + 1


def test_introduce_typos_invalid_prob():
    text = "I have severe chest pain that has been persistent for last week."
    
    # Test for invalid probability: 2.0
    with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
        introduce_typos(text, prob=2.0)
    
    # Test for invalid probability: 100
    with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
        introduce_typos(text, prob=100)
    
    # Test for invalid probability: -2.0
    with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
        introduce_typos(text, prob=-2.0)
    
    # Test for valid probability (this is just an example, you can define valid ranges in your logic)
    typo_text = introduce_typos(text, prob=0.3)  # Assuming 0.3 is a valid probability
    assert isinstance(typo_text, str)  # Check if the output is a string
    assert text != typo_text  # Ensure typos were introduced
