"""Check citation markers while an answer streams.

A marker can arrive split across chunks ("[" then "2]"), so text that might be the
start of one is held back until it can be judged. Markers pointing at sources that
were never supplied are removed, so a user can never click a made-up citation.
"""

import re

_MARKER = re.compile(
    r"\[\s*(?:source\s*)?(\d+(?:\s*[,;]\s*(?:source\s*)?\d+)*)\s*\]", re.IGNORECASE
)
_PARTIAL = re.compile(r"\[[\s\d,;a-z]*\Z", re.IGNORECASE)
_NUMBER = re.compile(r"\d+")
_MAX_HELD_CHARS = 40


class CitationFilter:
    def __init__(self, source_count: int) -> None:
        self._source_count = source_count
        self._buffer = ""
        self.cited: list[int] = []  # valid source numbers, in order of first use

    def _replace(self, match: re.Match[str]) -> str:
        valid = [
            number
            for number in (int(raw) for raw in _NUMBER.findall(match.group(1)))
            if 1 <= number <= self._source_count
        ]
        for number in valid:
            if number not in self.cited:
                self.cited.append(number)
        return "".join(f"[{number}]" for number in valid)

    def feed(self, text: str) -> str:
        """Add streamed text and return the part that is safe to show."""
        self._buffer += text
        hold_from = self._buffer.rfind("[")
        if (
            hold_from == -1
            or len(self._buffer) - hold_from > _MAX_HELD_CHARS
            or _PARTIAL.match(self._buffer, hold_from) is None
        ):
            ready, self._buffer = self._buffer, ""
        else:
            ready, self._buffer = self._buffer[:hold_from], self._buffer[hold_from:]
        return _MARKER.sub(self._replace, ready)

    def finish(self) -> str:
        """Return whatever is left when the stream ends."""
        ready, self._buffer = self._buffer, ""
        return _MARKER.sub(self._replace, ready)
