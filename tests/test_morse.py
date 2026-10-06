from eyemorse.morse import (CHAR_TO_MORSE, CURSOR, MORSE_TO_CHAR, TYPE, WORD_GAP, Action,
                            MorseDecoder, interpret, preview)


def test_tables_are_unique():
    assert len(CHAR_TO_MORSE) == len(MORSE_TO_CHAR)


def test_decoder_emits_letter_after_gap():
    d = MorseDecoder(letter_gap=0.9)
    for i, s in enumerate("..."):
        d.push(s, i * 0.3)
    assert d.update(1.0) == []          # only 0.4 s since last symbol
    assert d.update(1.5) == ["..."]
    assert d.buffer == ""


def test_decoder_word_gap():
    d = MorseDecoder(letter_gap=0.5, word_gap=2.0)
    d.push(".", 0.0)
    assert d.update(0.6) == ["."]
    assert d.update(1.0) == []
    assert d.update(2.1) == [WORD_GAP]
    assert d.update(5.0) == []           # only once


def test_type_mode():
    assert interpret(".-", TYPE) == Action("type", "a")
    assert interpret("..--", TYPE) == Action("type", " ")
    assert interpret("----", TYPE) == Action("key", "backspace")
    assert interpret(".-.-", TYPE) == Action("key", "enter")
    assert interpret(".-.-.-", TYPE) == Action("type", ".")
    assert interpret("......", TYPE).kind == "unknown"


def test_cursor_mode_ignores_natural_blinks():
    assert interpret(".", CURSOR) is None
    assert interpret("..", CURSOR) is None
    assert interpret("-", CURSOR) == Action("mouse", "left_click")
    assert interpret(".-", CURSOR) == Action("mouse", "left_click")  # stray blink first
    assert interpret("-.", CURSOR) == Action("mouse", "right_click")
    assert interpret(WORD_GAP, CURSOR) is None


def test_preview():
    assert preview("-.-.", TYPE) == "c"
    assert preview("..--", TYPE) == "space"
    assert preview("--", CURSOR) == "double_click"
