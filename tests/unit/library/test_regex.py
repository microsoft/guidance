import pytest
from functools import partial
import unicodedata

from guidance.library._gen import regex

from ...utils import check_match_failure, generate_and_check


def byte_range(byterange: bytes):
    start, end = byterange
    return {bytes([i]) for i in range(start, end + 1)}


ASCII_START_BYTES = byte_range(b"\x00\x7f")
UNICODE_SPECIAL_START_BYTES = byte_range(b"\xc2\xf4")
UNICODE_START_BYTES = ASCII_START_BYTES | UNICODE_SPECIAL_START_BYTES

# Equivalent to the following (in python 3.12)
# {
#     char.encode()[:1]
#     for codepoint in range(0x110000)
#     if unicodedata.category(char := chr(codepoint)) == "Nd"
# }
UNICODE_DIGIT_START_BYTES = byte_range(b"09") | {
    b"\xd9",
    b"\xdb",
    b"\xdf",
    b"\xe0",
    b"\xe1",
    b"\xea",
    b"\xef",
    b"\xf0",
}

# Equivalent to the following (in python 3.12)
# {
#     char.encode()[:1]
#     for codepoint in range(0x110000)
#     if unicodedata.category(char := chr(codepoint))[:1] in {"L", "M", "N"}
# } | {b"_"}
UNICODE_WORD_START_BYTES = {
    b"\xe5",
    b"9",
    b"J",
    b"x",
    b"\xd7",
    b"\xc2",
    b"\xcc",
    b"1",
    b"N",
    b"\xe9",
    b"c",
    b"C",
    b"i",
    b"7",
    b"\xec",
    b"k",
    b"\xe3",
    b"\xd9",
    b"4",
    b"\xde",
    b"S",
    b"\xcd",
    b"G",
    b"\xd2",
    b"b",
    b"\xc7",
    b"\xd1",
    b"\xdc",
    b"e",
    b"\xd3",
    b"\xe4",
    b"\xd0",
    b"\xc6",
    b"r",
    b"t",
    b"8",
    b"0",
    b"T",
    b"Z",
    b"\xc3",
    b"\xe1",
    b"\xe7",
    b"2",
    b"\xef",
    b"\xc8",
    b"R",
    b"W",
    b"\xca",
    b"\xce",
    b"\xcb",
    b"\xea",
    b"O",
    b"a",
    b"\xdd",
    b"5",
    b"n",
    b"L",
    b"v",
    b"\xda",
    b"Y",
    b"j",
    b"V",
    b"\xe2",
    b"d",
    b"f",
    b"\xf3",
    b"s",
    b"\xd4",
    b"y",
    b"u",
    b"\xdb",
    b"h",
    b"Q",
    b"I",
    b"A",
    b"\xc5",
    b"\xc9",
    b"\xd5",
    b"q",
    b"z",
    b"3",
    b"F",
    b"p",
    b"K",
    b"m",
    b"M",
    b"\xd6",
    b"\xf0",
    b"w",
    b"\xed",
    b"H",
    b"\xc4",
    b"X",
    b"_",
    b"\xe6",
    b"U",
    b"\xd8",
    b"6",
    b"g",
    b"E",
    b"o",
    b"\xdf",
    b"P",
    b"B",
    b"\xe0",
    b"\xcf",
    b"\xeb",
    b"D",
    b"\xe8",
    b"l",
}

# Equivalent to the following (in python 3.12)
# {
#     char.encode("utf-8", "surrogatepass")[:1]
#     for codepoint in range(0x110000)
#     if unicodedata.category(char := chr(codepoint))[:1] not in {"L", "M", "N"}
# } - {b"_"}
UNICODE_NON_WORD_START_BYTES = {
    b"@",
    b"\xf4",
    b"&",
    b"\xd7",
    b"\x1e",
    b"=",
    b"\xc2",
    b",",
    b"\x14",
    b"\x18",
    b"/",
    b"\x0e",
    b"\xe3",
    b"\x13",
    b"\x16",
    b"\xd9",
    b")",
    b"\xde",
    b"\xcd",
    b"\xf1",
    b"\xd2",
    b"\x10",
    b"\x1a",
    b"\xdc",
    b"\n",
    b"\x06",
    b"\xe4",
    b"\x17",
    b"$",
    b":",
    b";",
    b"\xc3",
    b"\x11",
    b"\x1d",
    b"^",
    b"|",
    b"\xe1",
    b"\x1c",
    b'"',
    b"`",
    b"(",
    b"\xef",
    b"'",
    b"\x07",
    b"\x05",
    b"\x08",
    b"*",
    b"\xce",
    b"\xcb",
    b"\xea",
    b"\xdd",
    b"}",
    b"\x12",
    b"\xee",
    b" ",
    b"\x0c",
    b"\xe2",
    b"\x01",
    b"\x04",
    b"\xf3",
    b"#",
    b"\x0b",
    b"+",
    b"\t",
    b"\xd4",
    b"\xdb",
    b"\x15",
    b"\x1f",
    b"\x1b",
    b"{",
    b"~",
    b"\xd5",
    b"\\",
    b"\x19",
    b"?",
    b"\x03",
    b"%",
    b"<",
    b">",
    b"\xd6",
    b"\xf0",
    b"\xed",
    b"\x0f",
    b"!",
    b"]",
    b"\xf2",
    b"\x02",
    b"-",
    b"[",
    b"\x00",
    b"\xe0",
    b"\xcf",
    b"\xdf",
    b"\xd8",
    b"\x7f",
    b"\r",
    b".",
}


class TestCharacterClasses:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"[abc]+", "cbabbaccabc"),
            (r"[a-z]+", "thequickbrownfoxjumpsoverthelazydog"),
            (r"[0-9]+", "9876543210"),
            (r"[b-y]+", "by"),  # range is left and right inclusive
            (r"[a-f0-9]+", "abcdef0123456789"),
            (r"[abcA-Z]+", "abcABCXYZ"),
            (r"[a-z\d]+", "abc123"),
            (r"[^abc]+", "ABCxyz8743-!@#$%^&*()_+"),
            (r"[^\d]+", "abcXYZ-!@#$%^&*()_+"),
            (r"[^B-Z]+", "qwertyA"),
            (r"[^a-z\d]+", "ABCDEF-!@#$%^&*()_+"),
            (r"[^\n]+", "ABCxyz8743-!@#$%^&*()_+"),
        ],
    )
    def test_good(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r"[abc]+",
                "cbabbaccabcx",
                b"cbabbaccabc",
                b"x",
                {b"a", b"b", b"c"},
            ),
            (
                r"[a-z]+",
                "thequickbrownfoxjumpsoverthelazydogX",
                b"thequickbrownfoxjumpsoverthelazydog",
                b"X",
                byte_range(b"az"),
            ),
            (
                r"[0-9]+",
                "9876543210x",
                b"9876543210",
                b"x",
                byte_range(b"09"),
            ),
            (
                r"[b-y]+",
                "bya",
                b"by",
                b"a",
                byte_range(b"by"),
            ),  # range doesn't overflow left
            (
                r"[b-y]+",
                "byz",
                b"by",
                b"z",
                byte_range(b"by"),
            ),  # range doesn't overflow right
            (
                r"[a-f0-9]+",
                "abcdef0123456789x",
                b"abcdef0123456789",
                b"x",
                {*byte_range(b"af"), *byte_range(b"09")},
            ),
            (
                r"[abcA-Z]+",
                "abcABCXYZx",
                b"abcABCXYZ",
                b"x",
                {b"a", b"b", b"c", *byte_range(b"AZ")},
            ),
            (
                r"[a-z\d]+",
                "abc123@",
                b"abc123",
                b"@",
                byte_range(b"az") | UNICODE_DIGIT_START_BYTES,
            ),
            (
                r"[^abc]+",
                "ABCxyz8743-!@#$%^&*()_+a",
                b"ABCxyz8743-!@#$%^&*()_+",
                b"a",
                UNICODE_START_BYTES - {b"a", b"b", b"c"},
            ),
            (
                r"[^\d]+",
                "abcXYZ-!@#$%^&*()_+6",
                b"abcXYZ-!@#$%^&*()_+",
                b"6",
                UNICODE_START_BYTES - byte_range(b"09"),
            ),
            (
                r"[^B-Z]+",
                "qwertyAB",
                b"qwertyA",
                b"B",
                UNICODE_START_BYTES - byte_range(b"BZ"),
            ),
            (
                r"[^a-z\d]+",
                "ABCDEF-!@#$%^&*()_+x",
                b"ABCDEF-!@#$%^&*()_+",
                b"x",
                UNICODE_START_BYTES - (byte_range(b"az") | byte_range(b"09")),
            ),
            (
                r"[^\n]+",
                "ABCxyz8743-!@#$%^&*()_+\n",
                b"ABCxyz8743-!@#$%^&*()_+",
                b"\n",
                UNICODE_START_BYTES - {b"\n"},
            ),
        ],
    )
    def test_bad(self, pattern, string, good_bytes, failure_byte, allowed_bytes):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )


class TestQuantifiers:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"a*b", "b"),
            (r"a*b", "ab"),
            (r"a*b", "aaaaab"),
            (r"a?b", "b"),
            (r"a?b", "ab"),
            (r"a+b", "ab"),
            (r"a+b", "aaaaab"),
        ],
    )
    def test_kleene_star_optional_one_or_more(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"a{3}", "aaa"),
            (r"(ab){2}", "abab"),
        ],
    )
    def test_exact_repetitions(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"(a{2,3}b){2}", "aabaaab"),
            (r"(a{2,3}){2}", "aaaaaa"),
            (r"a{2}b{2,3}", "aabbb"),
            (r"(ab){1,3}c{2,4}", "ababccc"),
            (r"(ab){1,3}c{2,4}", "abababcccc"),
        ],
    )
    def test_nested_quantifiers(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r"a*b",
                "axb",
                b"a",
                b"x",
                {b"a", b"b"},
            ),  # 'x' disrupts the match
            (
                r"a+b",
                "b",
                b"",
                b"b",
                {b"a"},
            ),  # 'a+' requires at least one 'a' before 'b'
            (
                r"a?b",
                "x",
                b"",
                b"x",
                {b"a", b"b"},
            ),  # 'a?' requires zero or one 'a' before 'b'
            (
                r"a?b",
                "axb",
                b"a",
                b"x",
                {b"b"},
            ),  # 'x' disrupts the match
            (
                r"a?b",
                "aab",
                b"a",
                b"a",
                {b"b"},
            ),  # Second 'a' is too many
            (
                r"(xyz)?abc",
                "xyabc",
                b"xy",
                b"a",
                {b"z"},
            ),  # Expected 'z'
            (
                r"(xyz)?abc",
                "abcx",
                b"abc",
                b"x",
                set(),
            ),  # Extra character after 'abc'
            (
                r"a{2,4}",
                "aaaaa",
                b"aaaa",
                b"a",
                set(),
            ),  # Too many 'a's
            (
                r"(ab){2,3}",
                "abababab",
                b"ababab",
                b"a",
                set(),
            ),  # 'ab' four times, more than the maximum
            (
                r"a{3,5}b",
                "aab",
                b"aa",
                b"b",
                {b"a"},
            ),  # Less than the minimum 'a's before 'b'
            (
                r"a{3,5}b",
                "aaaaaab",
                b"aaaaa",
                b"a",
                {b"b"},
            ),  # More than the maximum 'a's before 'b'
        ],
    )
    def test_quantifiers_failure(self, pattern, string, good_bytes, failure_byte, allowed_bytes):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )


class TestAlternations:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"a|b", "a"),
            (r"a|b", "b"),
            (r"cat|dog", "cat"),
            (r"cat|dog", "dog"),
        ],
    )
    def test_simple_alternations(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"apple|orange", "apple"),
            (r"apple|orange", "orange"),
            (r"100|200", "100"),
            (r"100|200", "200"),
        ],
    )
    def test_alternations_multiple_characters(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"(a|b)c|d", "ac"),
            (r"(a|b)c|d", "bc"),
            (r"(a|b)c|d", "d"),
        ],
    )
    def test_nested_alternations(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"(a|b)+", "aaa"),
            (r"(a|b)+", "bbb"),
            (r"(a|b)+", "abab"),
            (r"(dog|cat)s?", "dog"),
            (r"(dog|cat)s?", "cats"),
        ],
    )
    def test_alternations_with_quantifiers(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r"a|b",
                "c",
                b"",
                b"c",
                {b"a", b"b"},
            ),  # Neither 'a' nor 'b'
            (
                r"apple|orange",
                "banana",
                b"",
                b"b",
                {b"a", b"o"},
            ),  # Neither 'apple' nor 'orange'
            (
                r"100|200",
                "300",
                b"",
                b"3",
                {b"1", b"2"},
            ),  # Neither '100' nor '200'
            (
                r"(a|b)c|d",
                "ae",
                b"a",
                b"e",
                {b"c", b"c"},
            ),  # Neither 'ac' nor 'bc' nor 'd'
            (
                r"(a|b)+",
                "abbaabbabc",
                b"abbaabbab",
                b"c",
                {b"a", b"b"},
            ),  # 'c' does not match pattern '(a|b)+'
            (
                r"cat|dog",
                "car",
                b"ca",
                b"r",
                {b"t"},
            ),  # 't' should be forced
            (
                r"(dog|cat)s?",
                "cars",
                b"ca",
                b"r",
                {b"t"},
            ),  # 't' should be forced
        ],
    )
    def test_alternations_failures(self, pattern, string, good_bytes, failure_byte, allowed_bytes):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )


class TestDot:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r".+", "ABCxyz8743-!@#$%^&*()_+ \t"),
        ],
    )
    def test_dot(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r".+",
                "ABCxyz8743-!@#$%^&*()_+\n",
                b"ABCxyz8743-!@#$%^&*()_+",
                b"\n",
                UNICODE_START_BYTES - {b"\n"},
            ),
        ],
    )
    def test_dot_failures(self, pattern, string, good_bytes, failure_byte, allowed_bytes):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )


class TestSpecialCharacters:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"\d+", "1234567890"),
            (r"[^\D]+", "1234567890"),
            (r"\D+", "ABCxyz-!@#$%^&*()_+"),
            (r"[^\d]+", "ABCxyz-!@#$%^&*()_+"),
            (r"\w+", "abcABC123_"),
            (r"[^\W]+", "abcABC123_"),
            (r"\W+", " -!@#$%^&*()+"),
            (r"[^\w]+", "-!@#$%^&*()+"),
            (r"\s+", " \t\n\r\f\v"),
            (r"[^\S]+", " \t\n\r\f\v"),
            (r"\S+", "ABCxyz8743-!@#$%^&*()_+"),
            (r"[^\s]+", "ABCxyz8743-!@#$%^&*()_+"),
        ],
    )
    def test_good(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r"\d+",
                "0123456789x",
                b"0123456789",
                b"x",
                UNICODE_DIGIT_START_BYTES,
            ),
            (
                r"\D+",
                "ABCxyz-!@#$%^&*()_+1",
                b"ABCxyz-!@#$%^&*()_+",
                b"1",
                UNICODE_START_BYTES - byte_range(b"09"),
            ),
            (r"\w+", "abcABC123_@", b"abcABC123_", b"@", UNICODE_WORD_START_BYTES),
            (r"\W+", " -!@#$%^&*()+a", b" -!@#$%^&*()+", b"a", UNICODE_NON_WORD_START_BYTES),
            (
                r"\s+",
                " \t\n\r\f\v8",
                b" \t\n\r\f\v",
                b"8",
                {b" ", b"\t", b"\n", b"\r", b"\f", b"\v"}
                | {b"\xc2", b"\xe1", b"\xe2", b"\xe3"},  # include unicode whitespace starts
            ),
            (
                r"\S+",
                "abcABC123_ ",
                b"abcABC123_",
                b" ",
                UNICODE_START_BYTES
                - {
                    b" ",
                    b"\t",
                    b"\n",
                    b"\r",
                    b"\f",
                    b"\v",
                },
            ),
        ],
    )
    def test_bad(self, pattern, string, good_bytes, failure_byte, allowed_bytes):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )
