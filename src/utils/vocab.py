"""Vocabulary utilities for slice generation"""

# Define vocabulary
vocab = [
    "<START>",
    "B",
    "Zn",
    "Er",
    "P",
    "N",
    "Y",
    "Bi",
    "Ho",
    "Ce",
    "Te",
    "Ca",
    "4",
    "Nb",
    "Xe",
    "Dy",
    "C",
    "Re",
    "Tb",
    "Tm",
    "K",
    "Ta",
    "Tl",
    "Al",
    "1",
    "O",
    "Eu",
    "H",
    "Os",
    "Cd",
    "Hf",
    "In",
    "Na",
    "Tc",
    "Pb",
    "3",
    "Sn",
    "Cs",
    "0",
    "Ru",
    "Mo",
    "Fe",
    "F",
    "Zr",
    "Br",
    "Mn",
    "Ag",
    "I",
    "Pd",
    "Cl",
    "Ne",
    "Hg",
    "W",
    "Sr",
    "Kr",
    "V",
    "Si",
    "Ge",
    "La",
    "o",
    "2",
    "Mg",
    "Au",
    "Rb",
    "Se",
    "+",
    "Li",
    "Pm",
    "Ar",
    "Nd",
    "S",
    "Co",
    "Lu",
    "Pr",
    "Sc",
    "Cu",
    "Sb",
    "Rh",
    "He",
    "-",
    "Pt",
    "As",
    "Sm",
    "Gd",
    "Be",
    "Ga",
    "Ti",
    "Ni",
    "Ir",
    "Cr",
    "Ba",
    "<END>",
]

# Create mappings
stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for i, s in enumerate(vocab)}


def encode_slice(slice_text):
    """Convert slice text to token IDs"""
    tokens = [stoi["<START>"]]
    i = 0
    while i < len(slice_text):
        if (
            i + 1 < len(slice_text) and slice_text[i : i + 2] in stoi
        ):  # Try 2-char token
            tokens.append(stoi[slice_text[i : i + 2]])
            i += 2
        elif slice_text[i] in stoi:  # Try 1-char token
            tokens.append(stoi[slice_text[i]])
            i += 1
        else:
            i += 1  # Skip unknown characters
    tokens.append(stoi["<END>"])
    return tokens


def decode_slice(tokens):
    """Convert token IDs back to atomic symbols, stopping at EOS token"""
    special_tokens = {stoi["<START>"], stoi["<END>"]}
    result = []
    for token in tokens:
        # Skip special tokens and handle out-of-vocabulary tokens
        if token in itos and token not in special_tokens:
            result.append(itos[token])
    return "".join(result)
