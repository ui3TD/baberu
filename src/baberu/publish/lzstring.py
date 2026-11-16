"""lz-string's compressToBase64, matching the JavaScript library byte for byte.

NekoCap stores raw caption files compressed with lz-string (pieroxy/lz-string,
1.4-1.5). This port works on UTF-16 code units, like the original.
"""
_BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="


def _utf16_units(text: str) -> str:
    """The text as one Python character per UTF-16 code unit (surrogates kept apart)."""
    data = text.encode("utf-16-le", "surrogatepass")
    return "".join(chr(data[i] | (data[i + 1] << 8)) for i in range(0, len(data), 2))


class _BitWriter:
    def __init__(self, bits_per_char: int, alphabet: str):
        self.bits_per_char = bits_per_char
        self.alphabet = alphabet
        self.out: list[str] = []
        self.value = 0
        self.position = 0

    def write(self, value: int, bits: int) -> None:
        """Writes `bits` bits of `value`, lowest bit first."""
        for _ in range(bits):
            self.value = (self.value << 1) | (value & 1)
            if self.position == self.bits_per_char - 1:
                self.position = 0
                self.out.append(self.alphabet[self.value])
                self.value = 0
            else:
                self.position += 1
            value >>= 1

    def flush(self) -> str:
        while True:
            self.value <<= 1
            if self.position == self.bits_per_char - 1:
                self.out.append(self.alphabet[self.value])
                break
            self.position += 1
        return "".join(self.out)


def _compress(text: str, bits_per_char: int, alphabet: str) -> str:
    units = _utf16_units(text)
    dictionary: dict[str, int] = {}
    to_create: set[str] = set()
    w = ""
    enlarge_in = 2  # the first entry isn't counted
    dict_size = 3
    num_bits = 2
    out = _BitWriter(bits_per_char, alphabet)

    def emit(word: str) -> None:
        nonlocal enlarge_in, num_bits
        if word in to_create:
            code = ord(word[0])
            if code < 256:
                out.write(0, num_bits)
                out.write(code, 8)
            else:
                out.write(1, num_bits)
                out.write(code, 16)
            enlarge_in -= 1
            if enlarge_in == 0:
                enlarge_in = 2 ** num_bits
                num_bits += 1
            to_create.discard(word)
        else:
            out.write(dictionary[word], num_bits)
        enlarge_in -= 1
        if enlarge_in == 0:
            enlarge_in = 2 ** num_bits
            num_bits += 1

    for c in units:
        if c not in dictionary:
            dictionary[c] = dict_size
            dict_size += 1
            to_create.add(c)
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            emit(w)
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    if w:
        emit(w)
    out.write(2, num_bits)  # end of stream
    return out.flush()


def compress_to_base64(text: str) -> str:
    """Same output as JavaScript's LZString.compressToBase64(text)."""
    result = _compress(text, 6, _BASE64)
    return result + "=" * ((4 - len(result) % 4) % 4)
