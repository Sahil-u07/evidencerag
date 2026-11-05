import re


_CITATION = r"\[\s*Evidence\s*\d+\s*\]"
_BOUNDARY = re.compile(rf"[.!?]+(?:\s*{_CITATION})*(?=\s|$)", re.IGNORECASE)
_ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "vs.", "cf.", "fig.", "sec."}


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []

    for line in text.splitlines():
        start = 0

        for match in _BOUNDARY.finditer(line):
            words = line[start : match.start() + 1].split()

            if words and words[-1].lower() in _ABBREVIATIONS:
                continue

            sentences.append(line[start : match.end()].strip())
            start = match.end()

        remainder = line[start:].strip()

        if remainder:
            sentences.append(remainder)

    return sentences
