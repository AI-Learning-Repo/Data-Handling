# Lab: Build a Simple BPE Tokenizer from Scratch

## Learning Objectives

By the end of this lab, you will be able to:

* Explain how text can be represented as UTF-8 bytes.
* Find neighboring token pairs.
* Count how frequently pairs occur.
* Merge frequent pairs using the basic BPE algorithm.
* Build a vocabulary for newly created tokens.
* Train a simple BPE tokenizer.
* Use learned merge rules to encode new text.
* Decode token IDs back into the original text.
* Explain the difference between **training**, **encoding**, and **decoding**.
* Compare a simple educational tokenizer with `tiktoken`.

---

# Overview

Byte Pair Encoding (BPE) is a tokenization algorithm that starts with small units and repeatedly combines frequently occurring pairs.

In this lab, we will build a simplified BPE tokenizer using Python.

We will use this training text:

```python
text = "the cat in the hat"
```

Our implementation is intentionally small and educational.

We will:

```text
Text
  ↓
UTF-8 bytes
  ↓
Byte IDs
  ↓
Find frequent pairs
  ↓
Merge frequent pairs
  ↓
Learn merge rules
  ↓
Encode text into token IDs
  ↓
Decode token IDs back into text
```

## Simplifications

This is **not** a production tokenizer.

We will:

* use functions instead of a class;
* use UTF-8 bytes as the initial tokens;
* use `Counter` to count frequent pairs;
* use a small training dataset;
* ignore performance;
* ignore special tokens;
* ignore caching;
* ignore GPT-2 vocabulary files;
* focus on understanding the algorithm.

---

# Part 1: Text to UTF-8 Bytes

Before implementing BPE, we need to understand what our initial tokens are.

Python strings contain characters:

```python
text = "the cat"
```

Computers ultimately represent this text using bytes.

Python can convert a string into UTF-8 bytes:

```python
text = "the cat"

byte_array = bytearray(text, "utf-8")

print(byte_array)
```

You should see something similar to:

```text
bytearray(b'the cat')
```

The `b` indicates that we are looking at bytes.

We can convert the byte values into integers:

```python
ids = list(byte_array)

print(ids)
```

Output:

```text
[116, 104, 101, 32, 99, 97, 116]
```

For example:

```text
't' → 116
'h' → 104
'e' → 101
' ' → 32
'c' → 99
'a' → 97
't' → 116
```

So we can think of the text as:

```text
"the cat"

      ↓ UTF-8

[116, 104, 101, 32, 99, 97, 116]
```

Each number represents one byte.

---

## Why start with bytes?

A byte can have a value from:

```text
0 to 255
```

Therefore, our initial vocabulary can contain 256 possible byte tokens.

Later, BPE will create additional tokens.

For example:

```text
116 + 104
   ↓
"th"
```

could become a new token:

```text
256
```

---

# Task 1: Write `text_to_bytes()`

Create a function called `text_to_bytes()`.

The function should:

1. Receive a Python string.
2. Convert it to UTF-8 bytes.
3. Return those bytes as a list of integers.

Start with:

```python
def text_to_bytes(text):
    # Your code here
    ...
```

### Hint

You can use:

```python
bytearray(text, "utf-8")
```

and then:

```python
list(...)
```

---

<details>
<summary>Answer</summary>

```python
def text_to_bytes(text):
    # Convert the text into UTF-8 bytes.
    byte_array = bytearray(text, "utf-8")

    # Convert the byte values into a list of integers.
    return list(byte_array)
```

</details>



### Explanation

This line:

```python
byte_array = bytearray(text, "utf-8")
```

converts the string into UTF-8 bytes.

This line:

```python
return list(byte_array)
```

converts the bytes into ordinary Python integers.

---

# Calling the Function

Defining a function does not execute it.

This:

```python
def text_to_bytes(text):
    ...
```

only tells Python what the function should do.

We need to **call** it.

```python
text = "the cat in the hat"

tokens = text_to_bytes(text)

print(tokens)
```

You should see:

```text
[116, 104, 101, 32, 99, 97, 116, 32, 105, 110, 32, 116, 104, 101, 32, 104, 97, 116]
```

We now have our initial tokens.

---

## Checkpoint 1

Answer the following questions:

1. What does UTF-8 do?
2. What is the token ID for the character `t` in our example?
3. Why are the initial IDs between 0 and 255?
4. What does `text_to_bytes()` return?

---

# Part 2: Finding Neighboring Pairs

BPE works by looking at neighboring tokens.

Suppose we have:

```python
tokens = [1, 2, 3, 2, 3]
```

The neighboring pairs are:

```text
(1, 2)
(2, 3)
(3, 2)
(2, 3)
```

Notice that `(2, 3)` occurs twice.

Python's `zip()` makes this easy:

```python
pairs = list(zip(tokens, tokens[1:]))

print(pairs)
```

Output:

```text
[(1, 2), (2, 3), (3, 2), (2, 3)]
```

Let's understand this expression:

```python
tokens[1:]
```

returns everything except the first token:

```text
[2, 3, 2, 3]
```

The original list is:

```text
[1, 2, 3, 2, 3]
```

`zip()` then combines them:

```text
1 → 2
2 → 3
3 → 2
2 → 3
```

creating:

```text
(1, 2)
(2, 3)
(3, 2)
(2, 3)
```

---

# Counting Pairs

Now we want to count how often each pair occurs.

Python's `Counter` is useful for this.

```python
from collections import Counter

pair_counts = Counter(pairs)

print(pair_counts)
```

Output:

```text
Counter({
    (2, 3): 2,
    (1, 2): 1,
    (3, 2): 1
})
```

We can ask for the most common pair:

```python
most_common = pair_counts.most_common(1)

print(most_common)
```

Output:

```text
[((2, 3), 2)]
```

The first item is:

```python
most_common[0]
```

which gives:

```text
((2, 3), 2)
```

The pair itself is:

```python
most_common[0][0]
```

Therefore:

```python
pair = pair_counts.most_common(1)[0][0]

print(pair)
```

Output:

```text
(2, 3)
```

---

# Task 2: Write `get_frequent_pair()`

Create:

```python
def get_frequent_pair(tokens):
    ...
```

The function should:

1. Find neighboring pairs.
2. Count the pairs.
3. Return the most frequent pair.
4. Return `None` if there are fewer than two tokens.

---

<details>
<summary>Answer</summary>

```python
from collections import Counter

def get_frequent_pair(tokens):
    # We need at least two tokens to create a pair.
    if len(tokens) < 2:
        return None

    # Create neighboring pairs.
    pairs = list(zip(tokens, tokens[1:]))

    # Count how frequently each pair occurs.
    pair_counts = Counter(pairs)

    # Return the most frequent pair.
    return pair_counts.most_common(1)[0][0]
```
</details>



---

# Calling the Function

Now actually call it.

```python
tokens = [1, 2, 3, 2, 3, 2, 3]

pair = get_frequent_pair(tokens)

print("Most frequent pair:", pair)
```

Output:

```text
Most frequent pair: (2, 3)
```

We can also test the edge case:

```python
print(get_frequent_pair([10]))
```

Output:

```text
None
```

There is no pair because we only have one token.

---

## Checkpoint 2

Explain the following line:

```python
pairs = list(zip(tokens, tokens[1:]))
```

Then explain why we use:

```python
Counter(pairs)
```

---

# Part 3: Replacing a Pair

We now know how to find a frequent pair.

The next step is to **merge** it.

Suppose we have:

```text
[1, 2, 3, 2, 3, 4]
```

and we want to merge:

```text
(2, 3)
```

We create a new token ID:

```text
256
```

Then:

```text
[1, 2, 3, 2, 3, 4]
```

becomes:

```text
[1, 256, 256, 4]
```

Why 256?

Because:

```text
0–255
```

are already being used for the original byte values.

Therefore, the first newly created token can be:

```text
256
```

The next one will be:

```text
257
```

then:

```text
258
```

and so on.

---

# Understanding the Replacement Algorithm

We can implement the merge using a `while` loop.

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

Let's understand the important parts.

We start with an empty list:

```python
result = []
```

This will contain the new token sequence.

We use:

```python
i = 0
```

to keep track of our current position.

The loop:

```python
while i < len(tokens):
```

continues until we reach the end.

The condition:

```python
(tokens[i], tokens[i + 1]) == pair
```

checks whether the current token and the next token form the pair we want to merge.

If they do:

```python
result.append(new_id)
```

adds the new token.

Then:

```python
i += 2
```

moves forward by two positions because both tokens were consumed.

If there is no match, we copy the current token:

```python
result.append(tokens[i])
```

and move forward by one:

```python
i += 1
```

---

# Calling `replace_pair()`

Now let's call the function.

```python
tokens = [1, 2, 3, 2, 3, 4]

new_tokens = replace_pair(
    tokens,
    (2, 3),
    256
)

print("Before:", tokens)
print("After :", new_tokens)
```

Output:

```text
Before: [1, 2, 3, 2, 3, 4]
After : [1, 256, 256, 4]
```

Notice that both occurrences of `(2, 3)` were replaced.

---

## Checkpoint 3

Why do we use:

```python
i += 2
```

when a pair is found?

What would happen if we used:

```python
i += 1
```

instead?

---

# Part 4: Training BPE

We now have two important operations:

```text
get_frequent_pair()
        ↓
find the most frequent pair

replace_pair()
        ↓
merge that pair
```

Training BPE means repeating these operations.

The algorithm is:

```text
1. Start with byte tokens.
2. Find the most frequent pair.
3. Assign a new token ID.
4. Replace the pair.
5. Record the merge.
6. Repeat.
```

For example:

```text
(116, 104) → 256
```

means:

```text
t + h → 256
```

Then perhaps:

```text
(256, 101) → 257
```

means:

```text
th + e → 257
```

The exact sequence depends on the training data and the pair-selection rule.

---

# The `merges` Dictionary

We need to remember what we learned.

We can store the rules in a dictionary:

```python
merges = {
    (116, 104): 256,
    (256, 101): 257
}
```

This means:

```text
(116, 104) → 256
(256, 101) → 257
```

The merge rules are important because we will use them later when encoding new text.

---

# Task 4: Write `train_bpe()`

Create:

```python
def train_bpe(tokens, num_merges):
    ...
```

The function should:

1. Create an empty `merges` dictionary.
2. Find the most frequent pair.
3. Assign a new ID starting at 256.
4. Replace the pair.
5. Record the merge.
6. Repeat `num_merges` times.
7. Return the learned merge rules.

---

<details>
<summary>Answer</summary>

```python
def train_bpe(tokens, num_merges):
    # Store the learned merge rules.
    merges = {}

    # Make a copy so we do not modify the caller's list.
    tokens = tokens.copy()

    # New token IDs start after the 256 possible byte values.
    for new_id in range(256, 256 + num_merges):

        # Find the most frequent neighboring pair.
        pair = get_frequent_pair(tokens)

        # Stop if no pair can be formed.
        if pair is None:
            break

        # Replace the pair with the new token ID.
        tokens = replace_pair(tokens, pair, new_id)

        # Remember the merge rule.
        merges[pair] = new_id

    return merges
```

</details>



---

# Calling `train_bpe()`

First create the training tokens:

```python
training_text = "the cat in the hat"

training_tokens = text_to_bytes(training_text)

print("Training tokens:")
print(training_tokens)
```

Now train the tokenizer:

```python
merges = train_bpe(
    training_tokens,
    num_merges=10
)
```

Print the learned rules:

```python
print("Learned merges:")

for pair, new_id in merges.items():
    print(pair, "→", new_id)
```

The exact output depends on the training data and pair-selection behavior.

This is an important point:

> BPE training is data-dependent.

---

# Making Training Observable

A useful way to understand an algorithm is to see what happens at every iteration.

We can create another function:

```python
def train_bpe_observable(tokens, num_merges):
    merges = {}
    tokens = tokens.copy()

    for new_id in range(256, 256 + num_merges):

        pair = get_frequent_pair(tokens)

        if pair is None:
            break

        tokens = replace_pair(tokens, pair, new_id)

        merges[pair] = new_id

        print("Merge:", pair, "→", new_id)
        print("Tokens:", tokens)
        print()

    return merges
```

---

# Calling the Observable Version

```python
training_text = "the cat in the hat"

training_tokens = text_to_bytes(training_text)

merges = train_bpe_observable(
    training_tokens,
    num_merges=10
)
```

Now you can watch the tokenizer learn.

For example, you might see something conceptually similar to:

```text
Merge: (116, 104) → 256
Tokens: [...]

Merge: (256, 101) → 257
Tokens: [...]

...
```

Do not assume that these exact merges must appear first. The important thing is to inspect what your implementation actually learns.

---

## Checkpoint 4

Explain the difference between:

```python
get_frequent_pair(tokens)
```

and:

```python
replace_pair(tokens, pair, new_id)
```

Then explain what information is stored in:

```python
merges
```

---

# Part 5: Building a Vocabulary

Our token IDs are useful for a computer, but they are difficult for humans to understand.

For example:

```text
116
```

means:

```text
t
```

and:

```text
104
```

means:

```text
h
```

We can create a vocabulary that maps token IDs to their byte representation.

Start with:

```python
vocab = {i: bytes([i]) for i in range(256)}
```

This creates:

```text
0   → b'\x00'
1   → b'\x01'
...
104 → b'h'
...
116 → b't'
...
255 → b'\xff'
```

Let's inspect some entries:

```python
vocab = {i: bytes([i]) for i in range(256)}

print(vocab[116])
print(vocab[104])
print(vocab[101])
```

Output:

```text
b't'
b'h'
b'e'
```

---

# Creating Vocabulary Entries for Merges

Suppose we learn:

```text
(116, 104) → 256
```

We already know:

```text
116 → b't'
104 → b'h'
```

Therefore:

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

This is very useful.

Instead of thinking:

```text
256
```

we can understand that token 256 represents:

```text
"th"
```

---

# Task 5: Modify Training to Build the Vocabulary

Create a new version of `train_bpe()` that returns:

```python
merges, vocab
```

---

<details>
<summary>Answer</summary>

```python
def train_bpe(tokens, num_merges):
    # Store the learned merge rules.
    merges = {}

    # Create the initial vocabulary.
    # IDs 0-255 represent individual byte values.
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    # Make a copy of the input tokens.
    tokens = tokens.copy()

    # Create new token IDs starting from 256.
    for new_id in range(256, 256 + num_merges):

        # Find the most frequent pair.
        pair = get_frequent_pair(tokens)

        # Stop if no pair exists.
        if pair is None:
            break

        # Merge the pair.
        tokens = replace_pair(tokens, pair, new_id)

        # Remember the merge rule.
        merges[pair] = new_id

        # Build the byte representation of the new token.
        vocab[new_id] = (
            vocab[pair[0]]
            + vocab[pair[1]]
        )

    return merges, vocab
```

</details>



---

# Calling the New `train_bpe()`

Now call it:

```python
training_text = "the cat in the hat"

training_tokens = text_to_bytes(training_text)

merges, vocab = train_bpe(
    training_tokens,
    num_merges=10
)
```

Let's inspect the results:

```python
print("Number of merges:", len(merges))
print("Vocabulary size:", len(vocab))
```

We can also inspect the vocabulary:

```python
for token_id in sorted(vocab):
    if token_id >= 256:
        print(token_id, "→", vocab[token_id])
```

You should see the newly learned tokens.

---

## Checkpoint 5

Why does the vocabulary contain:

```text
0–255
```

from the beginning?

Why do new tokens start at:

```text
256
```

What does an entry such as:

```python
vocab[256]
```

represent?

---

# Part 6: Encoding Text

Training and encoding are different processes.

This distinction is extremely important.

## Training

During training, we ask:

> Which pair occurs most frequently?

Then we create a new merge.

```text
Find frequent pair
        ↓
Create token
        ↓
Record merge
```

## Encoding

During encoding, we already have the learned merge rules.

We ask:

> How should this new text be transformed using the rules we already learned?

We do **not** train again.

---

# Example

Suppose training learned:

```python
merges = {
    (116, 104): 256,
    (256, 101): 257
}
```

A new piece of text:

```python
"the"
```

starts as:

```text
[116, 104, 101]
```

The first rule says:

```text
116 + 104 → 256
```

so we get:

```text
[256, 101]
```

The second rule says:

```text
256 + 101 → 257
```

so we get:

```text
[257]
```

---

# Task 6: Write `encode()`

Create:

```python
def encode(text, merges):
    ...
```

The function should:

1. Convert text to byte IDs.
2. Apply the learned merge rules in their learned order.
3. Return the resulting token IDs.

---

<details>
<summary>Answer</summary>

```python
def encode(text, merges):
    # Start with individual UTF-8 byte tokens.
    tokens = text_to_bytes(text)

    # Apply the learned merge rules in order.
    for pair, new_id in merges.items():

        # Replace occurrences of the current pair.
        tokens = replace_pair(
            tokens,
            pair,
            new_id
        )

    return tokens
```

</details>



---

# Calling `encode()`

Now encode some text:

```python
text = "the cat in the hat"

ids = encode(
    text,
    merges
)

print("Text:", text)
print("Token IDs:", ids)
```

You can also encode new text:

```python
new_text = "the hat"

new_ids = encode(
    new_text,
    merges
)

print("Text:", new_text)
print("Token IDs:", new_ids)
```

This demonstrates an important idea:

> The tokenizer can apply what it learned during training to new text.

---

# Important Note About This Simplified Encoder

Our encoder applies the learned merge rules in the order they were learned:

```python
for pair, new_id in merges.items():
```

This is appropriate for this educational implementation.

Real BPE tokenizers can have more complicated details involving merge ranks, token boundaries, regular expressions, special tokens, and other implementation choices.

Our goal here is to understand the core idea, not reproduce GPT-2 exactly.

---

# Part 7: Decoding

We now have token IDs.

For example:

```text
[257, 32, 99, 97, 116]
```

How do we get the original text back?

Our vocabulary already knows what each token represents.

For example:

```text
116 → b"t"
104 → b"h"
256 → b"th"
257 → b"the"
```

Therefore, decoding can simply concatenate the byte strings.

---

# Example

Suppose:

```python
token_ids = [257, 32, 99, 97, 116]
```

We can look up:

```text
257 → b"the"
32  → b" "
99  → b"c"
97  → b"a"
116 → b"t"
```

Concatenating them gives:

```text
b"the cat"
```

Then we decode those bytes as UTF-8:

```text
"the cat"
```

---

# Task 7: Write `decode()`

Create:

```python
def decode(token_ids, vocab):
    ...
```

The function should:

1. Look up each token ID in the vocabulary.
2. Concatenate the corresponding bytes.
3. Decode the resulting bytes as UTF-8.
4. Return the Python string.

---

<details>
<summary>Answer</summary>

```python
def decode(token_ids, vocab):
    # Start with an empty byte string.
    result = b""

    # Look up every token.
    for token_id in token_ids:

        # Add its byte representation.
        result += vocab[token_id]

    # Convert the final bytes back into text.
    return result.decode("utf-8")
```

</details>



---

# Calling `decode()`

Now we can test it.

```python
ids = encode(
    "the cat in the hat",
    merges
)

decoded_text = decode(
    ids,
    vocab
)

print("Encoded:", ids)
print("Decoded:", decoded_text)
```

Expected:

```text
Decoded: the cat in the hat
```

We can verify it automatically:

```python
assert decoded_text == "the cat in the hat"
```

If nothing is printed, the assertion passed.

---

# Checkpoint 7

Explain why decoding does not need to manually perform:

```text
257
 ↓
256 + 101
 ↓
116 + 104 + 101
```

in our implementation.

What does the vocabulary allow us to do instead?

---

# Part 8: Seeing the Complete Training Process

Let's create an observable training function.

```python
def train_bpe_observable(tokens, num_merges):
    merges = {}

    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    tokens = tokens.copy()

    for new_id in range(256, 256 + num_merges):

        pair = get_frequent_pair(tokens)

        if pair is None:
            break

        tokens = replace_pair(
            tokens,
            pair,
            new_id
        )

        merges[pair] = new_id

        vocab[new_id] = (
            vocab[pair[0]]
            + vocab[pair[1]]
        )

        print(
            f"Merge: {pair} → {new_id}"
        )

        print(
            f"Token represents: {vocab[new_id]!r}"
        )

        print(
            f"Current tokens: {tokens}"
        )

        print()

    return merges, vocab
```

---

# Calling Observable Training

```python
training_text = "the cat in the hat"

training_tokens = text_to_bytes(
    training_text
)

merges, vocab = train_bpe_observable(
    training_tokens,
    num_merges=10
)
```

Now the algorithm is visible.

For each merge, you can see:

```text
Which pair was selected?
        ↓
Which ID was created?
        ↓
What bytes does that ID represent?
        ↓
What do the tokens look like now?
```

This is useful when debugging and when learning the algorithm.

---

# Part 9: Complete BPE Pipeline

We can now connect everything.

The complete process is:

```text
                 TRAINING
                    │
                    ↓
             Training text
                    │
                    ↓
              UTF-8 bytes
                    │
                    ↓
          Find frequent pairs
                    │
                    ↓
              Merge pairs
                    │
                    ↓
          Learn merge rules
                    │
                    ↓
              Build vocab
                    │
                    │
                    ▼
                  ENCODE
                    │
                    ↓
              New text
                    │
                    ↓
              UTF-8 bytes
                    │
                    ↓
           Apply learned merges
                    │
                    ↓
                Token IDs
                    │
                    ▼
                  DECODE
                    │
                    ↓
                 Bytes
                    │
                    ↓
                  Text
```

---

# Task 8: Run the Complete Pipeline

Use:

```python
text = "the cat in the hat"
```

First convert it to bytes:

```python
tokens = text_to_bytes(text)

print("Initial tokens:")
print(tokens)
```

Train the tokenizer:

```python
merges, vocab = train_bpe(
    tokens,
    num_merges=10
)
```

Encode the original text:

```python
ids = encode(
    text,
    merges
)

print("Token IDs:")
print(ids)
```

Decode the token IDs:

```python
decoded = decode(
    ids,
    vocab
)

print("Decoded text:")
print(decoded)
```

Finally:

```python
assert decoded == text
```

---

# Final Test

Let's print everything together:

```python
print("Original :", text)
print("Token IDs:", ids)
print("Decoded  :", decoded)
```

The important property is:

```python
decoded == text
```

We can verify this with:

```python
assert decoded == text
```

If the assertion does not raise an error, our tokenizer successfully encoded and decoded the text.

---

# Part 10: Experiment with the Tokenizer

Now that the basic implementation works, experiment with it.

## Experiment 1: Change the number of merges

Try:

```python
num_merges=1
```

Then:

```python
num_merges=5
```

Then:

```python
num_merges=10
```

Then:

```python
num_merges=20
```

Compare the resulting token IDs.

Ask yourself:

> What happens to the number of tokens as more merges are learned?

---

## Experiment 2: Change the training text

Try:

```python
training_text = "hello hello hello"
```

Then train again:

```python
training_tokens = text_to_bytes(
    training_text
)

merges, vocab = train_bpe(
    training_tokens,
    num_merges=10
)
```

Print the merges:

```python
for pair, new_id in merges.items():
    print(pair, "→", new_id, "=", vocab[new_id])
```

What patterns does BPE discover?

---

## Experiment 3: Encode new text

Train on:

```python
training_text = "the cat in the hat"
```

Then encode:

```python
new_text = "the hat"

ids = encode(
    new_text,
    merges
)

print(ids)
```

Then decode:

```python
decoded = decode(
    ids,
    vocab
)

print(decoded)
```

Ask:

> Does the tokenizer need to retrain when it sees new text?

Answer:

> No. Encoding uses the merge rules learned during training.

---

# Part 11: Understanding the Three Main Functions

At this point, the most important functions are:

## `text_to_bytes()`

Converts text into initial byte tokens.

```text
"the"
 ↓
[116, 104, 101]
```

---

## `train_bpe()`

Learns merge rules from training data.

```text
[116, 104, 101, ...]
        ↓
find frequent pair
        ↓
create new token
        ↓
record merge
        ↓
repeat
```

---

## `encode()`

Uses the learned rules on text.

```text
"the cat"
   ↓
byte IDs
   ↓
learned merges
   ↓
token IDs
```

---

## `decode()`

Converts token IDs back into text.

```text
token IDs
   ↓
vocabulary lookup
   ↓
bytes
   ↓
UTF-8
   ↓
text
```

---

# Part 12: Complete Script

Here is the complete educational implementation.

```python
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
```
 
---

# Part 13: Compare with `tiktoken`

Our tokenizer is useful for learning, but real tokenizers are much more sophisticated.

For example, OpenAI's `tiktoken` library provides optimized tokenizers.

If `tiktoken` is installed, you can try:

```python
import tiktoken
```

Then:

```python
tokenizer = tiktoken.get_encoding("gpt2")
```

Now encode some text:

```python
text = "This is some text"

ids = tokenizer.encode(text)

print(ids)
```

Decode it:

```python
decoded = tokenizer.decode(ids)

print(decoded)
```

You should get back:

```text
This is some text
```

---

# Comparing the Two Approaches

Our educational tokenizer:

```text
Simple Python functions
        ↓
UTF-8 bytes
        ↓
Count pairs
        ↓
Merge pairs
        ↓
Learn vocabulary
```

A production tokenizer:

```text
Optimized implementation
        ↓
Pre-trained vocabulary
        ↓
Pre-trained merge/ranking rules
        ↓
Additional tokenization rules
        ↓
Fast encoding/decoding
```

The purpose of our implementation is therefore **understanding**, not performance or compatibility with GPT-2.

---

# Final Questions

Answer these questions after completing the lab.

### Question 1

Why do we start with 256 possible byte token IDs?

### Question 2

Why does the first newly created token have ID 256?

### Question 3

What does the following merge mean?

```text
(116, 104) → 256
```

### Question 4

What is the difference between BPE training and encoding?

### Question 5

Why do we need to store the learned merges?

### Question 6

Why do we need a vocabulary?

### Question 7

What does `decode()` do?

### Question 8

What happens when we increase `num_merges`?

### Question 9

Why can the tokenizer trained on:

```text
"the cat in the hat"
```

also encode:

```text
"the hat"
```

?

### Question 10

What are some differences between this educational tokenizer and a production tokenizer such as `tiktoken`?

---

# Final Conceptual Summary

The core idea of BPE is simple:

```text
Start with small tokens
        ↓
Find frequent neighboring pairs
        ↓
Merge them into new tokens
        ↓
Repeat
```

For example:

```text
t + h
  ↓
th

th + e
  ↓
the

the + " "
  ↓
"the "
```

The tokenizer learns these patterns during **training**.

Later, when it receives new text, it uses those learned rules during **encoding**:

```text
new text
   ↓
bytes
   ↓
learned BPE merges
   ↓
token IDs
```

Finally, the token IDs can be converted back:

```text
token IDs
   ↓
vocabulary
   ↓
bytes
   ↓
UTF-8
   ↓
original text
```

The most important distinction to remember is:

```text
TRAINING
"What should I learn?"

ENCODING
"How do I apply what I learned?"

DECODING
"How do I turn the token IDs back into text?"
```

That is the basic idea behind the BPE tokenizer you have built in this lab.



---

[Raschka's full BPE tutorial](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

[Raschka's standalone notebook on GitHub](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe_from_scratch/bpe-from-scratch.ipynb)

