"""Unicode sanitization for API-safe text."""


def sanitize_surrogates(text: str) -> str:
    """Replace lone Unicode surrogates with the replacement character (U+FFFD).

    Valid emoji and paired surrogates are preserved. Lone surrogates
    (which break JSON serialization in many API providers) are replaced.
    """
    return text.encode("utf-8", errors="surrogatepass").decode(
        "utf-8", errors="replace"
    )
