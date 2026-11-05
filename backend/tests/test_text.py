import pytest

from app.core.text import split_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two.", ["One.", "Two."]),
        ("One. [Evidence 1] Two. [Evidence 2]", ["One. [Evidence 1]", "Two. [Evidence 2]"]),
        ("One [Evidence 1]. Two [Evidence 2].", ["One [Evidence 1].", "Two [Evidence 2]."]),
        ("One. [Evidence 1] [Evidence 2] Two.", ["One. [Evidence 1] [Evidence 2]", "Two."]),
        ("Use a tool, e.g. curl, here. Next.", ["Use a tool, e.g. curl, here.", "Next."]),
        ("TLS 1.3 removes it. Next.", ["TLS 1.3 removes it.", "Next."]),
        ("- First\n- Second", ["- First", "- Second"]),
        ("First. [Evidence 1]\nSecond. [Evidence 2]", ["First. [Evidence 1]", "Second. [Evidence 2]"]),
        ("No end mark", ["No end mark"]),
        ("", []),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected
