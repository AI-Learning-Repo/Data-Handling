# Assignment: Build a Simple BPE Tokenizer

### Objective

In this exercise, we will implement the main ideas behind Byte Pair Encoding (BPE) using Python functions.

We will use the same small training example:

```python
text = "the cat in the hat"
```

The implementation will be simplified:

* use functions instead of a class;
* focus on training, encoding, and decoding;
* use UTF-8 bytes as the initial tokens;
* use `Counter` to find frequent pairs;
* do not implement GPT-2 file loading;
* do not implement special-token handling;
* do not implement caching;
* do not worry about performance.

The purpose is to understand **how BPE works**, not to build a production tokenizer.

---

# Part 1: From text to bytes

We begin by converting text to UTF-8 bytes:

```python
text = "This is some text"

byte_array = bytearray(text, "utf-8")

print(byte_array)
```

This produces something like:

```text
bytearray(b'This is some text')
```

We can turn the bytes into integers:

```python
ids = list(byte_array)

print(ids)
```

The result is:

```text
[84, 104, 105, 115, 32, 105, 115, 32, ...]
```

Each number is between 0 and 255.

### Task 1

Write a function:

```python
def text_to_bytes(text):
    ...
```

It should return the UTF-8 bytes as a list of integers.

Test it:

```python
text = "the cat in the hat"

tokens = text_to_bytes(text)

print(tokens)
```

<details>
<summary><strong>View Answer for Task 1</strong></summary>

```python
def text_to_bytes(text):
    # Convert the string to a bytearray using utf-8 encoding
    byte_array = bytearray(text, "utf-8")
    # Convert the bytearray to a list of integers
    return list(byte_array)
```

</details>

### Concept

At this point, we have a very simple tokenizer:

```text
text
 ↓ UTF-8
bytes
 ↓
integer IDs
```

But there is a problem.

Every byte is currently a separate token.

For:

```text
"This is some text"
```

this gives 17 token IDs, whereas the GPT-2 tokenizer represents the same text with only four tokens. 

This motivates BPE.

---

# Part 2: Finding frequent pairs

Our next task is to find pairs of neighboring tokens.

For:

```python
tokens = [1, 2, 3, 2, 3]
```

the neighboring pairs are:

```text
(1, 2)
(2, 3)
(3, 2)
(2, 3)
```

Python's `zip` makes this easy:

```python
pairs = list(zip(tokens, tokens[1:]))

print(pairs)
```

Now count them:

```python
pair_counts = Counter(pairs)

print(pair_counts)
```

We can find the most frequent pair:

```python
pair = pair_counts.most_common(1)[0][0]

print(pair)
```

### Task 2

Turn this into a function:

```python
def get_frequent_pair(tokens):
    ...
```

It should return the most frequent adjacent pair.

Test it with:

```python
tokens = [1, 2, 3, 2, 3, 2, 3]

print(get_frequent_pair(tokens))
```

Expected:

```text
(2, 3)
```

<details>
<summary><strong>View Answer for Task 2</strong></summary>

```python
from collections import Counter

def get_frequent_pair(tokens):
    # If there are fewer than 2 tokens, no pairs can be formed
    if len(tokens) < 2:
        return None
        
    # Group neighboring tokens into pairs
    pairs = list(zip(tokens, tokens[1:]))
    
    # Count the occurrences of each pair
    pair_counts = Counter(pairs)
    
    # Return the most common pair
    return pair_counts.most_common(1)[0][0]
```

</details>

### Concept

This is the first important BPE operation:

```text
tokens
   ↓
find neighboring pairs
   ↓
count them
   ↓
choose the most frequent pair
```

This is the first step in the BPE outline. 

---

# Part 3: Replacing a pair

Suppose we have:

```text
[1, 2, 3, 2, 3, 4]
```

and the most frequent pair is:

```text
(2, 3)
```

We introduce a new token ID:

```text
256
```

and replace every occurrence:

```text
[1, 256, 256, 4]
```

Write:

```python
def replace_pair(tokens, pair, new_id):
    ...
```

A direct implementation is:

```python
def replace_pair(tokens, pair, new_id):
    result = []
    i = 0

    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return result
```

Test:

```python
tokens = [1, 2, 3, 2, 3, 4]

new_tokens = replace_pair(tokens, (2, 3), 256)

print(new_tokens)
```

Result:

```text
[1, 256, 256, 4]
```

<details>
<summary><strong>View Answer for Part 3 (Implementation)</strong></summary>

```python
def replace_pair(tokens, pair, new_id):
    result = []
    i = 0

    while i < len(tokens):
        # Check if we have a match for the pair
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            result.append(new_id)
            i += 2  # Skip the next token since it was part of the pair
        else:
            result.append(tokens[i])
            i += 1

    return result
```

</details>

### Concept

We have now implemented the second BPE operation:

```text
frequent pair
      ↓
new token ID
      ↓
replace pair
      ↓
record merge
```

The first new token ID being 256 because IDs `0–255` represent the possible byte values. 

---

# Part 4: Training BPE

Now we can combine the two functions.

Start with:

```python
text = "the cat in the hat"

tokens = text_to_bytes(text)

print(tokens)
```

Then repeatedly:

```text
1. Find the most frequent pair
2. Create a new token ID
3. Replace the pair
4. Record the merge
5. Repeat
```

We can write:

```python
def train_bpe(tokens, num_merges):

    merges = {}

    for new_id in range(256, 256 + num_merges):

        pair = get_frequent_pair(tokens)

        if pair is None:
            break

        tokens = replace_pair(tokens, pair, new_id)

        merges[pair] = new_id

    return tokens, merges
```

---

# Part 5: Connect the numbers back to text

The output might look like:

```text
(116, 104) → 256
(256, 101) → 257
(257, 32) → 258
...
```

The numbers initially look meaningless.

So we should make them human-readable.

Create a vocabulary containing the original bytes:

```python
vocab = {i: bytes([i]) for i in range(256)}
```

Now:

```python
vocab[116]
```

gives:

```text
b't'
```

and:

```python
vocab[104]
```

gives:

```text
b'h'
```

Therefore:

```text
(116, 104) → 256
```

means:

```text
t + h → 256
```

We can extend the vocabulary whenever we create a merge:

```python
vocab[256] = vocab[116] + vocab[104]
```

Now:

```python
print(vocab[256])
```

gives:

```text
b'th'
```

### Task 5

Modify `train_bpe()` so that it also builds the vocabulary.

A useful function interface would be:

```python
def train_bpe(tokens, num_merges):
    ...
    return merges, vocab
```

<details>
<summary><strong>View Answer for Task 5</strong></summary>

```python
def train_bpe(tokens, num_merges):
    merges = {}
    
    # Initialize vocabulary with single bytes (0-255)
    vocab = {i: bytes([i]) for i in range(256)}

    for new_id in range(256, 256 + num_merges):
        pair = get_frequent_pair(tokens)

        if pair is None:
            break

        tokens = replace_pair(tokens, pair, new_id)
        merges[pair] = new_id
        
        # Extend the vocabulary by concatenating the byte representations of the pair
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

    return merges, vocab
```

</details>

---

# Part 6: Making it Observable

Now we can make the training process visible.

For:

```text
the cat in the hat
```

the first important merge is:

```text
t + h → th
```

represented as:

```text
116 + 104 → 256
```

The text conceptually becomes:

```text
<256>e cat in <256>e hat
```

The next merge is:

```text
256 + e → 257
```

giving:

```text
<257> cat in <257> hat
```

Then:

```text
257 + space → 258
```

giving:

```text
<258>cat in <258>hat
```

This small example walks through the steps clearly. 

This is where students can print each step:

```python
print("Merge:", pair)
print("New ID:", new_id)
print("Tokens:", tokens)
```

That makes the algorithm observable rather than hiding it inside a function.

<details>
<summary><strong>View Answer for Part 6 (Observable Training)</strong></summary>

```python
def train_bpe_observable(tokens, num_merges):
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}

    for new_id in range(256, 256 + num_merges):
        pair = get_frequent_pair(tokens)
        if pair is None:
            break

        tokens = replace_pair(tokens, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        
        # Make it observable
        print(f"Merge: {pair} -> New ID: {new_id}")
        print(f"Tokens: {tokens}\n")

    return merges, vocab
```

</details>


---

# Part 7: Encoding new text

Now comes an important distinction.

We have **trained** our tokenizer.

The merge rules are stored in:

```python
merges
```

For example:

```python
{
    (116, 104): 256,
    (256, 101): 257,
    (257, 32): 258,
}
```

Suppose we now receive:

```python
text = "the hat"
```

We start with its bytes:

```python
tokens = text_to_bytes(text)
```

Then we apply the learned merges.

This is different from training.

During training:

> Find the most frequent pair.

During encoding:

> Apply the merges that we already learned.

`encode()` starts with individual bytes and repeatedly applies learned BPE merges according to their learned priority. 

### Task 7

Write:

```python
def encode(text, merges):
    ...
```

Start with:

```python
tokens = text_to_bytes(text)
```

Then apply the learned merges.

For a first simplified version, students can process the merge rules in their learned order.

<details>
<summary><strong>View Answer for Task 7</strong></summary>

```python
def encode(text, merges):
    # Start by converting the text to individual byte integers
    tokens = text_to_bytes(text)
    
    # Iterate through the merges dictionary (which inherently respects the insertion/learned order in modern Python)
    for pair, new_id in merges.items():
        # Apply each learned rule to the tokens
        tokens = replace_pair(tokens, pair, new_id)
        
    return tokens
```

</details>

---

# Part 8: Decoding

Now reverse the process.

Suppose encoding produced:

```text
[258, 99, 97, 116]
```

We need to turn the IDs back into bytes.

The vocabulary contains:

```text
256 → b"th"
257 → b"the"
258 → b"the "
```

So decoding is simply a vocabulary lookup followed by concatenation.

```python
def decode(token_ids, vocab):
    result = b""

    for token_id in token_ids:
        result += vocab[token_id]

    return result.decode("utf-8")
```

Test:

```python
ids = encode("the cat in the hat", merges)

print(ids)

text = decode(ids, vocab)

print(text)
```

We should get:

```text
the cat in the hat
```

<details>
<summary><strong>View Answer for Task 8 (Decoding Test Run)</strong></summary>

```python
def decode(token_ids, vocab):
    result = b""

    for token_id in token_ids:
        result += vocab[token_id]

    # Decode bytes back to a UTF-8 string
    return result.decode("utf-8")

# Assuming `merges` and `vocab` were generated from train_bpe
# ids = encode("the cat in the hat", merges)
# print("Encoded IDs:", ids)
# text = decode(ids, vocab)
# print("Decoded Text:", text)
```

</details>

### Important conceptual point

For this simplified implementation, we don't actually need to manually reverse:

```text
258 → 257 + space
257 → 256 + e
256 → t + h
```

because the vocabulary already tells us that:

```text
258 → b"the "
```

So decoding can simply concatenate the byte strings.

This is a nice opportunity to show **two ways of thinking about decoding**:

```text
Option 1: reverse the merges

258
↓
257 + space
↓
256 + e
↓
t + h + e + space


Option 2: use the vocabulary

258
↓
b"the "
```

Both express the same underlying idea.

The explicit reverse-merge process is the conceptual example, while this implementation's `decode()` uses the vocabulary representation. 

---

# Part 9: The complete pipeline

Finally, put everything together:

```python
text = "the cat in the hat"

# Training
tokens = text_to_bytes(text)

merges, vocab = train_bpe(
    tokens,
    num_merges=10
)

# Encoding
ids = encode(text, merges)

# Decoding
decoded = decode(ids, vocab)

print("Original :", text)
print("Token IDs:", ids)
print("Decoded  :", decoded)
```

And verify:

```python
assert decoded == text
```

<details>
<summary><strong>View the Complete Pipeline Script</strong></summary>

```python
from collections import Counter

def text_to_bytes(text):
    return list(bytearray(text, "utf-8"))

def get_frequent_pair(tokens):
    if len(tokens) < 2:
        return None
    pairs = list(zip(tokens, tokens[1:]))
    pair_counts = Counter(pairs)
    return pair_counts.most_common(1)[0][0]

def replace_pair(tokens, pair, new_id):
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

def train_bpe(tokens, num_merges):
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}
    
    for new_id in range(256, 256 + num_merges):
        pair = get_frequent_pair(tokens)
        if pair is None:
            break
        tokens = replace_pair(tokens, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        
    return merges, vocab

def encode(text, merges):
    tokens = text_to_bytes(text)
    for pair, new_id in merges.items():
        tokens = replace_pair(tokens, pair, new_id)
    return tokens

def decode(token_ids, vocab):
    result = b""
    for token_id in token_ids:
        result += vocab[token_id]
    return result.decode("utf-8")

# Execute Pipeline
text = "the cat in the hat"
tokens = text_to_bytes(text)
merges, vocab = train_bpe(tokens, num_merges=10)

ids = encode(text, merges)
decoded = decode(ids, vocab)

print("Original :", text)
print("Token IDs:", ids)
print("Decoded  :", decoded)
assert decoded == text, "Decoding failed to reproduce the original string!"
```

</details>

The complete process is:

```text
                 TRAINING
                    │
                    ↓
              training text
                    │
                    ↓
              UTF-8 bytes
                    │
                    ↓
          frequent-pair counting
                    │
                    ↓
               BPE merges
                    │
                    ↓
              vocabulary
                    │
                    │
                    ▼
NEW TEXT ───────→ ENCODE ───────→ TOKEN IDs
                                      │
                                      │
                                      ▼
                                   DECODE
                                      │
                                      ▼
                                    TEXT
```

---

## Then compare with `tiktoken`

This is a real-world example.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "This is some text"

ids = tokenizer.encode(text)

print(ids)
```

The GPT-2 tokenizer gives:

```text
[1212, 318, 617, 2420]
```

Then:

```python
print(tokenizer.decode(ids))
```

gives the original text.

Compare:

```text
Our educational tokenizer
        ↓
simple functions
        ↓
understand the algorithm


tiktoken
        ↓
optimized implementation
        ↓
used in real applications
```

`tiktoken` is recommended for practical use. The from-scratch implementation is intended as primarily educational.

---

[Raschka's full BPE tutorial](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

[Raschka's standalone notebook on GitHub](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe_from_scratch/bpe-from-scratch.ipynb)

