import re


_CITATION = r"\[\s*Evidence\s*\d+\s*\]"
_BOUNDARY = re.compile(rf"[.!?]+(?:\s*{_CITATION})*(?=\s|$)", re.IGNORECASE)
_ABBREVIATIONS = {"etc.", "vs.", "cf.", "fig.", "sec.", "dr.", "mr.", "mrs.", "ms.", "al.", "inc.", "approx."}
_INITIALISM = re.compile(r"(?:[a-z]\.){2,}")
_LIST_NUMBER = re.compile(r"\d+(?:\.\d+)*\.")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []

    for line in text.splitlines():
        start = 0

        for match in _BOUNDARY.finditer(line):
            words = line[start : match.start() + 1].split()

            last = words[-1].lower().lstrip("([\"'")

            if last in _ABBREVIATIONS or _INITIALISM.fullmatch(last):
                continue

            # A bare "3.1." is a section or list number, not the end of a sentence.
            if len(words) == 1 and _LIST_NUMBER.fullmatch(last):
                continue

            sentences.append(line[start : match.end()].strip())
            start = match.end()

        remainder = line[start:].strip()

        if remainder:
            sentences.append(remainder)

    return sentences
