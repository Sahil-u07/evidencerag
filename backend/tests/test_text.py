import pytest

from app.core.text import split_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two.", ["One.", "Two."]),
        (
            "One. [Evidence 1] Two. [Evidence 2]",
            ["One. [Evidence 1]", "Two. [Evidence 2]"],
        ),
        (
            "One [Evidence 1]. Two [Evidence 2].",
            ["One [Evidence 1].", "Two [Evidence 2]."],
        ),
        (
            "One. [Evidence 1] [Evidence 2] Two.",
            ["One. [Evidence 1] [Evidence 2]", "Two."],
        ),
        (
            "Use a tool, e.g. curl, here. Next.",
            ["Use a tool, e.g. curl, here.", "Next."],
        ),
        ("TLS 1.3 removes it. Next.", ["TLS 1.3 removes it.", "Next."]),
        ("- First\n- Second", ["- First", "- Second"]),
        (
            "First. [Evidence 1]\nSecond. [Evidence 2]",
            ["First. [Evidence 1]", "Second. [Evidence 2]"],
        ),
        ("3.1. Header Format", ["3.1. Header Format"]),
        ("1. First item. Second item.", ["1. First item.", "Second item."]),
        ("The limit is 512. Next.", ["The limit is 512.", "Next."]),
        ("The U.S. government agreed. Next.", ["The U.S. government agreed.", "Next."]),
        ("Dr. Smith agreed. Next.", ["Dr. Smith agreed.", "Next."]),
        ("Smith et al. found it. Next.", ["Smith et al. found it.", "Next."]),
        ("Acme Inc. makes it. Next.", ["Acme Inc. makes it.", "Next."]),
        ("It is approx. 5 long. Next.", ["It is approx. 5 long.", "Next."]),
        ("Do not ask for no. Next.", ["Do not ask for no.", "Next."]),
        (
            "Use a token (e.g. a JWT) here. Next.",
            ["Use a token (e.g. a JWT) here.", "Next."],
        ),
        ('Say "i.e. this" now. Next.', ['Say "i.e. this" now.', "Next."]),
        ("No end mark", ["No end mark"]),
        ("", []),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected
