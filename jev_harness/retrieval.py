"""Normalize public task and API terms for retrieval."""

from functools import lru_cache
import re


STOP = set("""a an the and or of to in on by at as is are be for from with my me i all this that
              it which what who whose how please user current available account task requested request
              read get show list use data output record records public id ids operation step prepare""".split())


@lru_cache(maxsize=4096)
def terms(text):
    words = re.findall(r"[^\W_]+", text.casefold())
    def singular(word):
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
            return word[:-1]
        return word
    return frozenset(singular(w) for w in words if w not in STOP)
