from collections import Counter


# --------------------------------------------------
# 1. Convert text to UTF-8 byte IDs
# --------------------------------------------------

def text_to_bytes(text):
    """
    Convert a Python string into a list of UTF-8 byte IDs.

    Each byte has a value from 0 to 255.
    """
    byte_array = bytearray(text, "utf-8")

    return list(byte_array)


# --------------------------------------------------
# 2. Find the most frequent neighboring pair
# --------------------------------------------------

def get_frequent_pair(tokens):
    """
    Find and return the most frequent neighboring pair.

    Example:

        [1, 2, 3, 2, 3]

    produces:

        (1, 2)
        (2, 3)
        (3, 2)
        (2, 3)

    The most frequent pair is:

        (2, 3)
    """

    if len(tokens) < 2:
        return None

    pairs = list(zip(tokens, tokens[1:]))

    pair_counts = Counter(pairs)

    return pair_counts.most_common(1)[0][0]


# --------------------------------------------------
# 3. Replace a pair with a new token ID
# --------------------------------------------------

def replace_pair(tokens, pair, new_id):
    """
    Replace every occurrence of 'pair' with 'new_id'.

    Example:

        tokens = [1, 2, 3, 2, 3, 4]
        pair = (2, 3)
        new_id = 256

    result:

        [1, 256, 256, 4]
    """

    result = []

    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and (tokens[i], tokens[i + 1]) == pair
        ):
            result.append(new_id)

            # We consumed two tokens.
            i += 2

        else:
            result.append(tokens[i])

            # We consumed one token.
            i += 1

    return result


# --------------------------------------------------
# 4. Train the BPE tokenizer
# --------------------------------------------------

def train_bpe(tokens, num_merges):
    """
    Learn BPE merge rules from training tokens.

    Returns:

        merges
        vocab
    """

    merges = {}

    # Initial vocabulary contains all 256 byte values.
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    # Work on a copy of the training tokens.
    tokens = tokens.copy()

    # New token IDs start at 256.
    for new_id in range(256, 256 + num_merges):

        # Find the most frequent pair.
        pair = get_frequent_pair(tokens)

        # Stop if there are no pairs.
        if pair is None:
            break

        # Replace the pair with the new token ID.
        tokens = replace_pair(
            tokens,
            pair,
            new_id
        )

        # Remember the merge rule.
        merges[pair] = new_id

        # Build the byte representation of the new token.
        vocab[new_id] = (
            vocab[pair[0]]
            + vocab[pair[1]]
        )

    return merges, vocab


# --------------------------------------------------
# 5. Encode text
# --------------------------------------------------

def encode(text, merges):
    """
    Convert text into token IDs using learned merges.
    """

    # Start with byte IDs.
    tokens = text_to_bytes(text)

    # Apply learned merges in order.
    for pair, new_id in merges.items():

        tokens = replace_pair(
            tokens,
            pair,
            new_id
        )

    return tokens


# --------------------------------------------------
# 6. Decode token IDs
# --------------------------------------------------

def decode(token_ids, vocab):
    """
    Convert token IDs back into a Python string.
    """

    result = b""

    for token_id in token_ids:

        result += vocab[token_id]

    return result.decode("utf-8")


# ==================================================
# EXECUTE THE COMPLETE PIPELINE
# ==================================================

# Training data
training_text = "the cat in the hat"

# Step 1: Convert training text to byte tokens
training_tokens = text_to_bytes(training_text)

print("Training tokens:")
print(training_tokens)
print()


# Step 2: Train BPE
merges, vocab = train_bpe(
    training_tokens,
    num_merges=10
)

print("Learned merges:")
for pair, new_id in merges.items():
    print(pair, "→", new_id)

print()


# Step 3: Encode text
ids = encode(
    training_text,
    merges
)

print("Encoded IDs:")
print(ids)
print()


# Step 4: Decode text
decoded = decode(
    ids,
    vocab
)

print("Decoded text:")
print(decoded)
print()


# Step 5: Verify correctness
assert decoded == training_text

print("Encoding and decoding successful!")