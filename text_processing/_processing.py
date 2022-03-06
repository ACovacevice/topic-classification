import re
import unicodedata
from typing import List

from text_processing import pos_tagger


class DeepSub:

    """
    Class for recursive string replacement using multiple regex patterns.
    Example:
        >> text = 'O rato roeu a roupa do rei de Roma.'
        >> cls = DeepSub(pattern1=r'(roupa)', pattern2=r'([aeiou])', repl=r"-")
        >> result = cls.sub(text)
        >> print(result)
        Output:
            'O rato roeu a r--p- do rei de Roma.'
        The first pattern found a match for the word 'roupa'.
        The second pattern found vowels inside the word 'roupa'.
        Replacement for the final matches took place, replacing vowels in 'roupa' by '-'.
    """

    def __init__(self, repl: str, flags: int = 0, **kwargs) -> None:
        """
        Args:
            pattern1, pattern2, ... (str):
                regex patterns to lookout for (first is mandatory, rest is optional).
            repl (str):
                replacement value for matches.
            flags (int, optional):
                flags to be passed to the regex engine. Defaults to 0.
        Obs.: One important thing is to encapsulate patterns within parentheses; otherwise, it may not work.
        """
        kwargs.update(
            {
                k: re.compile(v, flags)
                for k, v in kwargs.items()
                if k.startswith("pattern")
            }
        )
        self.__dict__ = kwargs
        self.__pats = [
            kwargs[k] for k in sorted(kwargs.keys()) if k.startswith("pattern")
        ]
        self.__pos = 0
        self.repl = repl

    def __sub(self, match: re.Match) -> str:
        self.__pos += 1
        if self.__pos == len(self.__pats) - 1:
            self.__pos = 0
            return self.__pats[-1].sub(self.repl, match.groups()[0])
        return self.__pats[self.__pos].sub(self.__sub, match.groups()[0])

    def sub(self, text: str) -> str:
        """Replace all patterns found in `text`.
        Args:
            text (str): text to be processed.
        Returns:
            str: processed version of `text`.
        """
        if len(self.__pats) == 1:
            return self.__pats[0].sub(self.repl, text)
        return self.__pats[self.__pos].sub(self.__sub, text)


def remove_bad_chars(text: str) -> str:
    """Remove chars alien to Brazilian Portuguese.
    Args:
        text (str): text to be processed.
    Returns:
        str: processed version of `text`.
    """
    chars = re.compile(
        r"([^a-z\s0-9&#@=><\+\/\*\^,\.;:!?_\-\)\(\]\[\}\{áâãàéêíóôõúç]+)", flags=2
    )

    def _repl(match):
        return chars.sub(
            " ",
            unicodedata.normalize("NFKD", match.groups()[0])
            .encode("ascii", errors="ignore")
            .decode("utf-8", errors="ignore"),
        )

    return chars.sub(_repl, text)


def reformat_abbreviations(text: str) -> str:
    """Replace abbreviations such as U.S.A. by USA.
    Args:
        text (str): text to be processed.
    Returns:
        str: processed version of `text`.
    """
    pattern = DeepSub(pattern1=r"((:?[A-Z]+\.)+)", pattern2=r"(\.)", repl=r"", flags=0)
    return pattern.sub(text)


def reformat_float(text: str) -> str:
    """Replace float numbers such as 27.345,90 by 2734590.
    Args:
        text (str): text to be processed.
    Returns:
        str: processed version of `text`.
    """
    pattern = DeepSub(
        pattern1=r"([0-9]*[,\.]*[0-9]+)", pattern2=r"([,\.]+)", repl="", flags=2
    )
    return pattern.sub(text)


def replace_email(text: str, by: str = " ") -> str:
    """Replace emails by `by`.
    Args:
        text (str): text to be processed.
        by (str, optional): replacement value. Defaults to " ".
    Returns:
        str: processed version of `text`.
    """
    pattern = re.compile(r"([A-Z0-9_.+-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+)", flags=2)
    return pattern.sub(by, text)


def replace_url(text: str, by: str = " ") -> str:
    """Replace URLs by `by`.
    Args:
        text (str): text to be processed.
        by (str, optional): replacement value. Defaults to " ".
    Returns:
        str: processed version of `text`.
    """
    pattern = re.compile(
        r"((https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))",
        flags=2,
    )
    return pattern.sub(by, text)


def replace_date(text: str, by: str = " ") -> str:
    """Replace dates by `by`.
    Args:
        text (str): text to be processed.
        by (str, optional): replacement value. Defaults to " ".
    Returns:
        str: processed version of `text`.
    """
    pattern = re.compile(r"(\d{2}\s?[\/\-]\s?\d{2}\s?[\/\-]\s?\d{2,4})", flags=2)
    return pattern.sub(by, text)


def remove_repetitions(text: str) -> str:
    """Remove repeating tokens in succession.
    Args:
        text (str): text to be processed.
    Returns:
        str: processed version of `text`.
    """
    # Removes repetitions of non-alphanumerical chars.
    ptext = DeepSub(pattern1=r"([^a-záâãàéêíóôõúç0-9\s])\1+", repl=r"\1", flags=2).sub(
        text
    )
    # Removes repetitions of words separated by a space.
    ptext = DeepSub(pattern1=r"\b(\w+)( \1\b)+", repl=r"\1", flags=2).sub(ptext)
    return ptext


def remove_ntr(text: str) -> str:
    """Remove \n, \r and \t from `text`.
    Args:
        text (str): text to be processed.
    Returns:
        str: processed version of `text`.
    """
    sub = DeepSub(pattern1=r"(:?(\-+[\n\r]+)+)", pattern2=r"([\-\n\r]+)", repl=r"")
    return sub.sub(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")


def adjust_spacing(text: str) -> str:
    """Correct spacing issues in `text`.
    Args:
        text (str): text to be processed.
    Returns:
        str: processed version of `text`.
    """
    # Create spacing around punctuation.
    create_spaces = DeepSub(pattern1=r"([^\w\d\s]+)", repl=r" \1 ", flags=2)
    # Collapse multiple spaces into only one.
    collapse_spaces = DeepSub(pattern1=r"(\s+)", repl=r" ", flags=2)
    return collapse_spaces.sub(create_spaces.sub(text)).strip()


def clean_text(text: str) -> str:
    """Apply series of transformations to `text`.
    Args:
        text (str): text to be processed.
    Returns:
        str: processed version of `text`.
    """
    ptext = remove_bad_chars(text)
    ptext = replace_email(ptext)
    ptext = replace_url(ptext)
    ptext = replace_date(ptext)
    ptext = reformat_abbreviations(ptext)
    ptext = reformat_float(ptext)
    ptext = remove_repetitions(ptext)
    ptext = remove_ntr(ptext)
    return adjust_spacing(ptext)


def replace_accented_chars(text: str) -> str:
    """In order to reduce vocabulary, normalize characters from `text`
    to not being accented.
    Args:
        text (str): text to be processed.
    Returns:
        str: processed version of `text`.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def get_keywords_from_text(text: str) -> List:
    """Return keywords from `text`.
    Args:
        text (str): text to be processed.
    Returns:
        list: extracted keywords.
    """
    tags, text = [], clean_text(text)

    if isinstance(text, str):
        # Remove symbols that are not letters or spaces.
        text = re.sub(r"[^a-z áâãàéêíóôõúüç-]+", "", text.lower())

    for sentence in pos_tagger.tag(text):
        tokens, pos = zip(*sentence)
        for idx, expr in enumerate(pos):
            if expr in ("N", "ADJ") and len(tokens[idx]) > 2:
                tag = tokens[idx]
                tags.append(replace_accented_chars(tag))

    return tags