import unicodedata

def normalize(text: str) -> str:
    """NFC, lowercase, punctuation -> space, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    chars = [
        " " if unicodedata.category(c).startswith("P") else c
        for c in text
    ]
    return " ".join("".join(chars).split())

def edit_distance(ref_words: list[str], hyp_words: list[str]) -> int:
    """Levenshtein distance between two word lists (dynamic programming)."""
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1

    
    d = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        d[i][0] = i                      # delete everything
    for j in range(cols):
        d[0][j] = j                      # insert everything

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,         
                d[i][j - 1] + 1,         
                d[i - 1][j - 1] + cost,  
            )

    return d[rows - 1][cols - 1]

def wer(reference: str, hypothesis: str) -> float:
    """Word error rate = edits / number of reference words."""
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return edit_distance(ref_words, hyp_words) / len(ref_words)