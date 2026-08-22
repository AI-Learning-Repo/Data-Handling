# Activity 2: Building the Data Processing Pipeline

In this activity, you will step into the shoes of a Data Engineer. You will take raw, messy datasets from the internet and process them using Python so they are perfectly formatted and ready to train a Large Language Model (LLM).

## Setup: Environment Preparation

Before processing data, the required programming libraries must be installed in your Google Colab environment.

1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Create a new code cell, paste the following command, and click the "Run" button (the play icon).

```python
# Install the necessary libraries
!pip install datasets pandas -q
```

> [!NOTE] 
> **Terminology:**
> *   `datasets`: A library by Hugging Face used to easily download public datasets.
> *   `pandas`: The industry-standard Python library used for analyzing and manipulating tabular data (think of it as coding your own Excel spreadsheet).
> *   `-q`: Stands for "quiet," which reduces the amount of installation text printed on the screen so your notebook stays clean.

### Group Discussion 1
**Question:** In the code above, we used an exclamation mark (`!`) before `pip install`. In Colab, you might also occasionally see commands starting with a percent sign (`%` or `%%`). What is the difference between `!` and `%`, and why do we use them?
<details>
<summary><strong>Click here to reveal the answer</strong></summary>
<ul>
<li><b><code>!</code> (Shell Command):</b> Google Colab is basically a Python notebook running on a Linux computer in the cloud. The <code>!</code> tells Colab, <i>"Do not run this as Python code. Instead, pass this directly to the underlying computer's terminal."</i> We use it to install software (<code>!pip</code>) or download files.</li>
<li><b><code>%</code> or <code>%%</code> (Magic Commands):</b> These are special commands built specifically into Jupyter/Colab notebooks to control how the notebook behaves. For example, <code>%time</code> will measure exactly how many seconds a line of Python code takes to run, which is highly useful when processing massive datasets!</li>
</ul>
</details>

---

## Part 1: Inspecting and Deduplicating Data

**Relates to reading material sections:** *Inspect Data* and *Deduplicate Data*.
**Dataset:** `fka/awesome-chatgpt-prompts` (A simple dataset containing system prompts).

**Context:** The reading material notes that manual inspection is critical for understanding data quality. It also states that duplicated data can introduce bias and cause test contamination, sometimes degrading a model's performance drastically.

**Your Task:**
Use Gemini to write Python code that accomplishes the following:
1. Loads the `fka/awesome-chatgpt-prompts` dataset from Hugging Face.
2. Converts the dataset into a Pandas DataFrame (a data table).
3. Prints the total number of rows.
4. Removes any duplicate rows based on the `prompt` column.
5. Prints the total number of rows again to see if any duplicates were removed.

**Suggested prompt for Gemini:**
> *"Write Python code for Google Colab. I need to load the Hugging Face dataset 'fka/awesome-chatgpt-prompts' (train split) and convert it to a pandas DataFrame. Print the number of rows. Then, drop any exact duplicates from the 'prompt' column, and print the new number of rows."*

<details>
<summary><strong>Click here for the working code and Python explanations</strong></summary>

```python
from datasets import load_dataset
import pandas as pd

# 1. Load the dataset using the Hugging Face library
dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train")

# 2. Convert to a Pandas DataFrame
# A DataFrame (often named 'df') is a 2-dimensional table with rows and columns.
df_prompts = dataset.to_pandas()

# 3. Print the original length using the len() function
print("Original row count:", len(df_prompts))

# 4. Remove duplicate rows
# The drop_duplicates method looks at the 'prompt' column and keeps only the first occurrence.
df_clean = df_prompts.drop_duplicates(subset=['prompt'])

# 5. Print the new length
print("Cleaned row count:", len(df_clean))

# 6. Display the first 5 rows using the head() method
display(df_clean.head())
```

**Python Concepts Introduced:**
*   **Variables:** `dataset`, `df_prompts`, and `df_clean` are variables. They store the data in the computer's memory so we can use it on the next line.
*   **Functions:** `len()` is a built-in Python function that counts the number of items (in this case, rows).
*   **Methods:** `.to_pandas()`, `.drop_duplicates()`, and `.head()` are methods. A method is a function attached to a specific object (like our dataset or DataFrame).
</details>

### Group Discussion 2
**Question:** The code above successfully removed *exact* duplicate prompts. In the context of LLM training, why might dropping exact duplicates not be enough? What other types of duplicates exist that we should worry about?
<details>
<summary><strong>Click here to reveal the answer</strong></summary>
Exact match deduplication misses <b>semantic duplicates</b> (data that means the exact same thing but is written slightly differently). <br><br>
For example: <i>"Write a poem about a cat"</i> vs. <i>"Create a poem about a feline."</i><br><br>
If a dataset is flooded with semantic duplicates, the AI model will still over-index on those specific concepts, leading to biased training. Advanced data engineers use embeddings and similarity scores (like Cosine Similarity) to find and remove semantic duplicates!
</details>

---

## Part 2: Cleaning and Filtering Data

**Relates to reading material section:** *Clean and Filter Data*.
**Dataset:** `rotten_tomatoes` (A dataset of movie reviews and their sentiment labels).

**Context:** Datasets often contain low-quality examples that must be removed. In this step, we will use heuristics (rules of thumb) to filter out garbage data and ensure our labels are properly formatted.

**Your Task:**
Use Gemini to write Python code that accomplishes the following:
1. Loads the `rotten_tomatoes` dataset and converts it to a DataFrame.
2. Filters the data to remove any reviews (the `text` column) that are shorter than 30 characters.
3. The dataset currently uses `0` for negative reviews and `1` for positive reviews in the `label` column. Replace these numerical labels with the actual words "negative" and "positive".

**Suggested prompt for Gemini:**
> *"Write Python code for Google Colab using pandas. Load the Hugging Face dataset 'rotten_tomatoes'. Convert it to a DataFrame. First, filter the DataFrame so it only keeps rows where the 'text' string length is 30 characters or more. Second, map the 'label' column so that 0 becomes 'negative' and 1 becomes 'positive'. Display the first few rows."*

<details>
<summary><strong>Click here for the working code and Python explanations</strong></summary>

```python
from datasets import load_dataset
import pandas as pd

# 1. Load the dataset
dataset_rt = load_dataset("rotten_tomatoes", split="train")
df_movies = dataset_rt.to_pandas()

# 2. Filter out short text
# df_movies['text'].str.len() calculates the character count of each row.
# >= 30 creates a True/False filter.
# The outer df_movies[...] applies the filter to keep only the True rows.
df_filtered = df_movies[df_movies['text'].str.len() >= 30].copy()

# 3. Rename numerical labels to text
# A Python Dictionary {} maps keys to values. Here, 0 maps to "negative".
label_mapping = {0: "negative", 1: "positive"}
df_filtered['sentiment'] = df_filtered['label'].map(label_mapping)

# 4. View the updated data
display(df_filtered[['text', 'label', 'sentiment']].head())
```

**Python Concepts Introduced:**
*   **Boolean Indexing:** Writing `df[condition]` is the standard Pandas way to filter rows based on logical conditions (True/False).
*   **Dictionaries:** `label_mapping = {0: "negative", 1: "positive"}` is a dictionary. Dictionaries store data in key-value pairs.
*   **.map():** This method replaces values in a column by looking them up in a dictionary.
</details>

### Group Discussion 3
**Question:** Why did we decide to drop reviews that were shorter than 30 characters? Think about the "Garbage In, Garbage Out" principle from the reading material.
<details>
<summary><strong>Click here to reveal the answer</strong></summary>
Reviews shorter than 30 characters (e.g., <i>"It's ok"</i>, <i>"Bad movie"</i>, <i>"Loved it"</i>) lack sufficient context or reasoning. If an LLM is trained on these, it learns to give short, unhelpful, and superficial answers instead of nuanced explanations. By filtering out the "garbage," we force the model to learn from high-quality, descriptive text.
</details>

---

## Part 3: Formatting Data (Building Chat Templates)

**Relates to reading material section:** *Format Data*.
**Dataset:** `databricks/databricks-dolly-15k` (A dataset with distinct instruction and response columns).

**Context:** The reading material emphasizes that data must perfectly match the exact "chat template" expected by the target model. If you are training a Google model vs. a Meta model, the required formatting is completely different! In this lab, you will format the exact same data in two different ways: one for **Gemma** (Google) and one for **Llama 3** (Meta).

**Your Task:**
Use Gemini to write Python code that creates a custom function to format the dataset.
*   **Gemma Template:** `<start_of_turn>user\n[INSTRUCTION]<end_of_turn>\n<start_of_turn>model\n[RESPONSE]<end_of_turn>`
*   **Llama 3 Template:** `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n[INSTRUCTION]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[RESPONSE]<|eot_id|>`

**Suggested prompt for Gemini:**
> *"I have a pandas DataFrame containing the dataset 'databricks/databricks-dolly-15k'. Write a Python function that takes a row and creates two f-strings using the 'instruction' and 'response' columns. 
> 1. A Gemma template using `<start_of_turn>user\n...` 
> 2. A Llama 3 template using `<|begin_of_text|><|start_header_id|>user...`
> Return both strings as a pandas Series. Use `.apply()` to run this on the dataframe and create two new columns: 'gemma_formatted' and 'llama3_formatted'. Print the first row of each to compare."*

<details>
<summary><strong>Click here for the working code and Python explanations</strong></summary>

```python
from datasets import load_dataset
import pandas as pd

# 1. Load the dataset
dataset_dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
df_dolly = dataset_dolly.to_pandas()

# 2. Define the formatting function
def format_templates(row):
    instruction = row['instruction']
    response = row['response']
    
    # f-strings allow us to inject variables inside curly brackets {}. \n creates a line break.
    
    # Gemma Format
    gemma = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
    
    # Llama 3 Format
    llama3 = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    
    # Return as a Pandas Series so it creates two separate columns
    return pd.Series([gemma, llama3])

# 3. Apply the function to the DataFrame
df_dolly[['gemma_formatted', 'llama3_formatted']] = df_dolly.apply(format_templates, axis=1)

# 4. Print to compare the differences!
print("--- GEMMA FORMAT ---")
print(df_dolly['gemma_formatted'].iloc[0])
print("\n--- LLAMA 3 FORMAT ---")
print(df_dolly['llama3_formatted'].iloc[0])
```

**Python Concepts Introduced:**
*   **Custom Functions (`def`):** Functions group code into reusable blocks. They take an input, perform an action, and `return` an output.
*   **f-strings:** Placing an `f` before a string allows you to easily inject Python variables directly into the text using `{curly_brackets}`. This is the industry standard for constructing prompt templates in AI engineering.
*   **.apply():** This is a powerful Pandas method that runs a custom function across an entire dataset (thousands of rows) instantly. 
</details>

### Group Discussion 4
**Question:** Look at the printout comparing the Gemma format to the Llama 3 format. They look completely alien to each other! What would happen if you successfully fine-tuned a Llama 3 model on the `llama3_formatted` column, but when you deployed it to your app, your app's code sent user messages using the `gemma_formatted` template?
<details>
<summary><strong>Click here to reveal the answer</strong></summary>
The model would likely hallucinate, output absolute garbage, or fail to respond entirely. <br><br>
LLMs treat these special tokens (like <code>&lt;start_of_turn&gt;</code> or <code>&lt;|eot_id|&gt;</code>) as strict mathematical boundaries. If the boundaries suddenly change, the model literally does not understand where the user's prompt ends and where its own answer is supposed to begin. <b>Your live application prompts must exactly match the template your model was trained on!</b>
</details>

---

**Conclusion:** 
Congratulations! You have successfully implemented the core steps of a Dataset Engineering pipeline. Your processed datasets are now deduplicated, cleaned, and perfectly formatted to be fed into an LLM training algorithm. Ensure your Colab notebook is saved for future reference.

Happy coding!