"""Tests for Unicode surrogate handling."""

from pi_ai import sanitize_surrogates


class TestSanitizeSurrogates:
    def test_normal_ascii(self):
        assert sanitize_surrogates("hello world") == "hello world"

    def test_empty_string(self):
        assert sanitize_surrogates("") == ""

    def test_preserves_emoji(self):
        text = "Hello \U0001f600 World"  # grinning face
        assert sanitize_surrogates(text) == text

    def test_preserves_monkey_emoji(self):
        text = "\U0001f435"  # monkey face
        assert sanitize_surrogates(text) == text

    def test_preserves_thumbs_up(self):
        text = "\U0001f44d"
        assert sanitize_surrogates(text) == text

    def test_preserves_rocket(self):
        text = "\U0001f680"
        assert sanitize_surrogates(text) == text

    def test_preserves_japanese(self):
        text = "\u3053\u3093\u306b\u3061\u306f"  # konnichiwa
        assert sanitize_surrogates(text) == text

    def test_preserves_chinese(self):
        text = "\u4f60\u597d"  # ni hao
        assert sanitize_surrogates(text) == text

    def test_preserves_mathematical_symbols(self):
        text = "\u222b\u2211\u221a"  # integral, sum, sqrt
        assert sanitize_surrogates(text) == text

    def test_preserves_curly_quotes(self):
        text = "\u201c\u201d"  # left/right double quotes
        assert sanitize_surrogates(text) == text

    def test_preserves_mixed_content(self):
        text = "Hello \U0001f600 World \U0001f44d \u3053\u3093\u306b\u3061\u306f"
        assert sanitize_surrogates(text) == text

    def test_real_world_linkedin_data(self):
        # German text with monkey emoji (real-world pattern)
        text = "Senior Software Engineer bei ACME GmbH \U0001f435"
        assert sanitize_surrogates(text) == text

    def test_multiple_emoji_in_sequence(self):
        text = "\U0001f600\U0001f601\U0001f602\U0001f603"
        assert sanitize_surrogates(text) == text

    def test_emoji_at_boundaries(self):
        text = "\U0001f600 start and end \U0001f600"
        assert sanitize_surrogates(text) == text
