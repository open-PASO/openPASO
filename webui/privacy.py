"""Nothing that identifies the person or the machine leaves the server.

Transcripts, file views and run records are things people screenshot, share
and attach to papers. Tool output routinely contains the home directory and the
user name (`pwd`, `ls -l`, tracebacks). Everything sent to a browser or
downloaded passes through `scrub`; the files on disk stay exactly as written.
"""
from __future__ import annotations

import getpass
import re
from pathlib import Path

_HOME = str(Path.home())
try:
    _USER = getpass.getuser()
except Exception:
    _USER = ""

# A path is recognised only where one can begin. Base64 (a field file's frames
# are megabytes of it) uses A-Z a-z 0-9 + / =, so "/home/<name>" occurs inside
# it by chance; replacing that changed the numbers a run had computed. Requiring
# a boundary in front leaves encoded data alone and still catches every path in
# prose, JSON, logs and tracebacks.
#
# This machine's own home directory is the exception, from five characters on ("/root"): it is
# removed wherever it stands as a path, glued to letters included ('A' * 40 + '<home>/x' kept its home
# behind the boundary rule). Base64 holds the home and the "/" after it by chance about once in
# 64**6 positions at five characters, about once in five thousand 12 MB fields, and practically
# never for a /home/<name> of nine or more; a shorter home names nobody.
_BOUNDARY = r"(?<![A-Za-z0-9+/])"   # "=" stays a boundary: --prefix=/home/... is a path
_HOMELY = len(_HOME) >= 5
_HOME_RE = re.compile(("" if _HOMELY else _BOUNDARY) + re.escape(_HOME) + r"(?=[/\s'\"\\:,)\]}]|$)")
_ANY_HOME_RE = re.compile(_BOUNDARY + r"/(?:home|Users|media)/[A-Za-z0-9._-]+")
_USER_RE = re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(_USER) + r"(?![A-Za-z0-9_-])") if len(_USER) >= 3 else None


# The encoded payloads a run writes: a field series keeps its frames and its
# mask as base64, and those must reach the browser byte for byte or the picture
# shows numbers nobody computed.
#
# Named rather than guessed. "any long run of base64 characters" also matches a
# long enough home path — "/home/<name>/" followed by forty characters is one
# such run — and the guard would then hide from the scrubber exactly what the
# scrubber exists to remove. So only the values of the keys that carry encoded
# data are protected, and everything else is scrubbed as prose.
_KEYED_VALUE = re.compile(r'"(?:frames|mask|data|image)"\s*:\s*"([^"\\]*)"')
_DATA_URI = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,([A-Za-z0-9+/=]+)")

# AND DECIDED ON THE WHOLE VALUE, by one rule wherever a value is met: in a parsed object, in a
# document, in a download streamed in pieces, in a data: URI's payload. Encoded data is base64 as an
# encoder writes it: groups of four of A-Z a-z 0-9 + /, only the last group of a block padded ("xx=="
# or "xxx="), blocks possibly joined, and possibly an unpadded group of two or three at the end. It is
# at least 40 characters long, it does not begin where a home directory begins, and it never holds
# this machine's home directory as a path (see _HOMELY). Anything else is text and is scrubbed as
# text.
# Each weaker form let a home path out, measured on all three paths: judged by the first 64
# characters, or with spaces allowed, 'A' * 64 + ' <home>/x' and a long path a run wrote under
# "data"; with "=" allowed anywhere, 'A' * 40 + '=<home>/x' (Copilot on the org PR, three rounds).
# The form alone cannot settle it: 'A' * 40 + '<home>/x' IS valid base64, so the home check decides.
_BASE64_TEXT = re.compile(r"[A-Za-z0-9+/=]{40,}")
_PADDING = re.compile(r"=+")


def _is_base64(value: str) -> bool:
    if _BASE64_TEXT.fullmatch(value) is None or len(value) % 4 == 1:
        return False
    whole = len(value) - len(value) % 4           # an unpadded last group of 2 or 3 starts here
    return all(len(m.group()) <= 2 and m.end() % 4 == 0 and m.end() <= whole
               for m in _PADDING.finditer(value))


def is_encoded(value: str) -> bool:
    """Whether `value`, met under one of the encoded keys, is encoded data to pass on untouched."""
    return (_is_base64(value) and _ANY_HOME_RE.match(value) is None
            and (_HOME_RE.search(value) if _HOMELY else _HOME_RE.match(value)) is None)


def _scrub_prose(t: str) -> str:
    """Every substitution, on a span that is not encoded data."""
    t = _HOME_RE.sub("~", t)
    t = _ANY_HOME_RE.sub(lambda m: "~" if m.group(0).startswith(("/home/", "/Users/")) else "/media/…", t)
    if _USER_RE is not None:
        t = _USER_RE.sub("user", t)
    return t


def scrub_text(s: str) -> str:
    """Scrub the prose and leave encoded data exactly as the run wrote it.

    The split comes first and covers all three substitutions, rather than
    guarding whichever one was fixed last. The boundary in the patterns above
    is not enough on its own: "=" has to stay a boundary so that
    --prefix=/home/... is caught, and "=" is also base64's padding, so
    an equals sign followed by a home path inside concatenated frames matched
    and was rewritten."""
    if not s:
        return s
    # the two kinds cannot overlap: encoded data holds no ":" and a data: URI no quote
    spans = sorted([m.span(1) for m in _KEYED_VALUE.finditer(s) if is_encoded(m.group(1))]
                   + [m.span(1) for m in _DATA_URI.finditer(s) if is_encoded(m.group(1))])
    out, last = [], 0
    for start, end in spans:
        out.append(_scrub_prose(s[last:start]))
        out.append(s[start:end])            # untouched
        last = end
    out.append(_scrub_prose(s[last:]))
    return "".join(out)


# The keys whose values are encoded data. scrub_text() recognises them written
# out ("frames": "..."), which is how a whole document reaches it — but walking
# a parsed object hands over the bare value with its key already stripped, so
# the guard could not fire there at all and a field's frames were scrubbed as
# prose. This is the same defect a third time in this file, so the walk names
# the keys itself rather than relying on the text form.
_ENCODED_KEYS = {"frames", "mask", "data", "image"}


def scrub(obj, _key: str | None = None):
    if isinstance(obj, str):
        # the whole value decides, by the one rule above
        if _key in _ENCODED_KEYS and is_encoded(obj):
            return obj                      # the numbers a solver produced
        return scrub_text(obj)
    if isinstance(obj, list):
        return [scrub(x, _key) for x in obj]
    if isinstance(obj, dict):
        # keys as well: a run's own JSON can be keyed by a path it wrote
        return {(scrub_text(k) if isinstance(k, str) else k): scrub(v, k if isinstance(k, str) else None)
                for k, v in obj.items()}
    return obj
