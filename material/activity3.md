# Activity 3: Practical Guide to Pydantic

Learn how to validate, structure, and format data in Python — from simple objects to fine-tuned model outputs.

Beginner-friendly

Python

Pydantic v2

LLM outputs

# Activity 3: Practical Guide to Pydantic

## Learning objectives

By the end of this activity, you will be able to:

1. Explain what Pydantic is and why it is useful.

2. Create a Pydantic model using Python type annotations.

3. Validate data and understand validation errors.

4. Add constraints using `Field`.

5. Work with nested models and lists.

6. Format and validate structured outputs from a fine-tuned language model.

## 1. What is Pydantic?

When we work with Python data, we often expect it to follow a specific structure.

For example, a student record might look like this:

Python

Run

```
student = {
    "name": "Sara",
    "age": 21,
    "course": "Data Science"
}
```

But what happens if someone provides:

Python

Run

```
student = {
    "name": "Sara",
    "age": "twenty-one",
    "course": "Data Science"
}
```

Or forgets to include the student's name?

Python dictionaries do not automatically check whether the data follows the structure we expect. This is where Pydantic becomes useful.

### What does Pydantic do?

Pydantic is a Python library that uses type annotations to:

* Define the structure of data.

* Check whether incoming data is valid.

* Convert some compatible data types.

* Report errors when data does not match the expected schema.

* Convert validated objects into dictionaries or JSON.

A simple way to think about Pydantic:

> Pydantic is like a quality-control checkpoint for your data. It checks whether incoming information follows the rules before your program uses it.

### Installation

Run the following in a Jupyter Notebook or Google Colab cell:

Python

Run

```
!pip install pydantic
```

We will use Pydantic v2 syntax throughout this activity.

# Example 1: Your First Pydantic Model

### Goal

Create a simple model for a student and learn how Pydantic checks data types and required fields.

## Step 1: Define the schema

Python

Run

```
from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int
    course: str
```

Let's understand this step by step.

### What is `BaseModel`?

Python

Run

```
class Student(BaseModel):
```

`BaseModel` is the main class provided by Pydantic. When our class inherits from it, Pydantic knows that it should treat the class as a data model and validate its fields.

### What are the fields?

Python

Run

```
name: str
age: int
course: str
```

These are the rules for our student data.

|
Field

|

Expected type

|
| --- | --- |
|

`name`

|

String (`str`)

|
|

`age`

|

Integer (`int`)

|
|

`course`

|

String (`str`)

|

The fields are required because we have not given them default values.

## Step 2: Create a student

Python

Run

```
student = Student(
    name="Sara",
    age=21,
    course="Data Science"
)

print(student)
```

Output:

```
name='Sara' age=21 course='Data Science'
```

Pydantic has created a `Student` object that follows our schema.

## Step 3: Check the data types

Python

Run

```
print(student.name)
print(student.age)
print(type(student.age))
```

Output:

```
Sara
21
<class 'int'>
```

The age is stored as an integer.

## Step 4: Try invalid data

What happens if we provide an age that is not a valid integer?

Python

Run

```
from pydantic import ValidationError

try:
    student = Student(
        name="Sara",
        age="twenty-one",
        course="Data Science"
    )

except ValidationError as e:
    print("Validation failed!")
    print(e)
```

Pydantic raises a `ValidationError` because `"twenty-one"` cannot be converted into an integer.

### Important concept: Type coercion

Pydantic can sometimes convert compatible types.

For example:

Python

Run

```
student = Student(
    name="Sara",
    age="21",
    course="Data Science"
)

print(student.age)
print(type(student.age))
```

Output:

```
21
<class 'int'>
```

The string `"21"` is converted into the integer `21`.

However, this does not mean Pydantic can convert every string into a number.

|
Input

|

Result

|
| --- | --- |
|

`"21"`

|

Can be converted to `int`

|
|

`"twenty-one"`

|

Validation error

|
|

`21`

|

Valid integer

|

Beginner takeaway: Type annotations describe the expected data, and Pydantic checks those expectations at runtime.

# Example 2: Default Values and Validation Rules

### Goal

Learn how to make some fields optional and add rules about what values are allowed.

Imagine we are building a simple system for tracking books in a library.

Each book should have:

* A title.

* An author.

* A year of publication.

* A number of available copies.

We want to make sure that the publication year is reasonable and the number of copies is not negative.

## Step 1: Create the model

Python

Run

```
from pydantic import BaseModel, Field

class Book(BaseModel):
    title: str
    author: str
    publication_year: int = Field(ge=1900, le=2026)
    available_copies: int = Field(default=0, ge=0)
```

Let's break down the important parts.

### What does `Field` do?

`Field` allows us to add extra validation rules to a field.

For example:

Python

Run

```
publication_year: int = Field(ge=1900, le=2026)
```

This means:

* The value must be an integer.

* The value must be greater than or equal to `1900`.

* The value must be less than or equal to `2026`.

### Common constraints

|
Constraint

|

Meaning

|
| --- | --- |
|

`gt=0`

|

Greater than 0

|
|

`ge=0`

|

Greater than or equal to 0

|
|

`lt=100`

|

Less than 100

|
|

`le=100`

|

Less than or equal to 100

|
|

`min_length=3`

|

Minimum string length of 3

|
|

`max_length=50`

|

Maximum string length of 50

|

## Step 2: Create a valid book

Python

Run

```
book = Book(
    title="Python for Beginners",
    author="Alex Brown",
    publication_year=2024,
    available_copies=5
)

print(book)
```

Output:

```
title='Python for Beginners' author='Alex Brown' publication_year=2024 available_copies=5
```

## Step 3: Use the default value

We can leave out `available_copies`.

Python

Run

```
book = Book(
    title="Learning AI",
    author="Maya Lee",
    publication_year=2025
)

print(book.available_copies)
```

Output:

```
0
```

Because we defined:

Python

Run

```
available_copies: int = Field(default=0, ge=0)
```

Pydantic uses `0` when the field is missing.

## Step 4: Try invalid values

Python

Run

```
try:
    book = Book(
        title="Learning AI",
        author="Maya Lee",
        publication_year=1800,
        available_copies=-3
    )

except ValidationError as e:
    print("Book validation failed!")

    for error in e.errors():
        print(error)
```

Both values violate the schema:

* `1800` is earlier than the allowed publication year.

* `-3` is less than zero.

### Understanding `e.errors()`

The method:

Python

Run

```
e.errors()
```

returns a list of dictionaries containing information about each validation error.

For example, an error might include:

Python

Run

```
{
    "type": "greater_than_equal",
    "loc": ("available_copies",),
    "msg": "Input should be greater than or equal to 0"
}
```

The most useful fields for beginners are:

* `loc`: Which field caused the error.

* `msg`: A human-readable explanation.

* `type`: The type of validation error.

### Beginner takeaway

`Field` helps you describe not only what type of data is expected, but also which values are allowed.

# Example 3: Nested Models and Lists

### Goal

Learn how to represent data that contains other data models.

Real-world data is often more complex than a single flat dictionary.

For example, an online course may have:

* A course name.

* An instructor.

* A list of lessons.

* Each lesson has a title and duration.

Instead of putting everything into one large model, we can create smaller models and combine them.

## Step 1: Create a model for a lesson

Python

Run

```
from pydantic import BaseModel

class Lesson(BaseModel):
    title: str
    duration_minutes: int
```

This model describes one lesson.

Example:

Python

Run

```
lesson = Lesson(
    title="Introduction to Python",
    duration_minutes=30
)

print(lesson)
```

## Step 2: Create a model for the course

Python

Run

```
class Course(BaseModel):
    name: str
    instructor: str
    lessons: list[Lesson]
```

The important field is:

Python

Run

```
lessons: list[Lesson]
```

This means:

> `lessons` must be a list, and every item in that list must follow the `Lesson` schema.

## Step 3: Validate nested data

Suppose we receive this dictionary from an API:

Python

Run

```
course_data = {
    "name": "Python Basics",
    "instructor": "Maya Lee",
    "lessons": [
        {
            "title": "Variables and Data Types",
            "duration_minutes": 20
        },
        {
            "title": "Functions",
            "duration_minutes": 35
        }
    ]
}
```

We can validate it using:

Python

Run

```
course = Course.model_validate(course_data)

print(course)
```

Pydantic will:

1. Check the course name.

2. Check the instructor.

3. Check that `lessons` is a list.

4. Validate every lesson.

5. Convert each valid lesson dictionary into a `Lesson` object.

## Step 4: Access nested data

Python

Run

```
print(course.name)

print(course.lessons[0].title)

print(type(course.lessons[0]))
```

Output:

```
Python Basics
Variables and Data Types
<class '__main__.Lesson'>
```

The first lesson is no longer just a dictionary. It is a validated `Lesson` object.

## Step 5: Export the model

After validating the data, we may want to convert it back into a dictionary.

Python

Run

```
course_dict = course.model_dump()

print(course_dict)
print(type(course_dict))
```

Output:

```
{
    'name': 'Python Basics',
    'instructor': 'Maya Lee',
    'lessons': [
        {
            'title': 'Variables and Data Types',
            'duration_minutes': 20
        },
        {
            'title': 'Functions',
            'duration_minutes': 35
        }
    ]
}
<class 'dict'>
```

### Export to JSON

JSON is a common format for APIs and machine learning applications.

Python

Run

```
course_json = course.model_dump_json(indent=2)

print(course_json)
```

This produces a JSON string:

JSON

```
{
  "name": "Python Basics",
  "instructor": "Maya Lee",
  "lessons": [
    {
      "title": "Variables and Data Types",
      "duration_minutes": 20
    },
    {
      "title": "Functions",
      "duration_minutes": 35
    }
  ]
}
```

### Beginner takeaway

Nested Pydantic models help us organize complex data into smaller, reusable structures.

# Example 4: Formatting Outputs from a Fine-Tuned Model

### Goal

Use Pydantic to validate and structure the output of a fine-tuned language model.

This is especially useful when a model is expected to return information in a specific format.

## The problem

Imagine you fine-tuned a language model to classify customer messages.

Input:

```
My package has not arrived yet.
```

You want the model to return:

JSON

```
{
  "category": "delivery",
  "sentiment": "negative",
  "confidence": 0.95
}
```

But language models sometimes return:

* Missing fields.

* Incorrect field names.

* Extra explanations.

* Invalid JSON.

* Wrong data types.

* Confidence values outside the expected range.

For example:

JSON

```
{
  "category": "delivery",
  "sentiment": "negative",
  "confidence": 1.5
}
```

A confidence of `1.5` is not valid if we expect confidence to be between `0` and `1`.

Pydantic can help us check the output before we use it in our application.

## Step 1: Define the expected output schema

Python

Run

```
from typing import Literal
from pydantic import BaseModel, Field

class ClassificationOutput(BaseModel):
    category: Literal["delivery", "billing", "technical", "other"]
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)
```

Let's understand this model.

### `Literal`

Python

Run

```
category: Literal["delivery", "billing", "technical", "other"]
```

This means the category must be exactly one of the allowed values.

Valid:

Python

Run

```
"delivery"
```

Invalid:

Python

Run

```
"shipping"
```

unless `"shipping"` is added to the allowed list.

### Confidence constraints

Python

Run

```
confidence: float = Field(ge=0, le=1)
```

This means the confidence score must be between `0` and `1`, inclusive.

Valid:

Python

Run

```
0.95
```

Invalid:

Python

Run

```
1.5
```

## Step 2: Simulate a model output

For this activity, we will pretend that the fine-tuned model has already generated the following dictionary.

Python

Run

```
model_output = {
    "category": "delivery",
    "sentiment": "negative",
    "confidence": 0.95
}
```

Validate the output:

Python

Run

```
result = ClassificationOutput.model_validate(model_output)

print(result)
```

Output:

```
category='delivery' sentiment='negative' confidence=0.95
```

## Step 3: Access the validated fields

Python

Run

```
print("Category:", result.category)
print("Sentiment:", result.sentiment)
print("Confidence:", result.confidence)
```

Output:

```
Category: delivery
Sentiment: negative
Confidence: 0.95
```

Now our Python program can safely use these fields according to the schema.

## Step 4: Detect an invalid model output

Suppose the model returns:

Python

Run

```
bad_model_output = {
    "category": "shipping",
    "sentiment": "negative",
    "confidence": 1.5
}
```

Validate it:

Python

Run

```
try:
    result = ClassificationOutput.model_validate(bad_model_output)

except ValidationError as e:
    print("The model output is invalid!")

    for error in e.errors():
        print(error)
```

Pydantic will report errors for:

* `category`: The value `"shipping"` is not in the allowed list.

* `confidence`: The value `1.5` is greater than the maximum of `1`.

### Why is this useful for fine-tuned models?

A fine-tuned model may be trained to produce a certain output format, but training alone does not guarantee that every output will follow the format perfectly.

Pydantic gives us a validation step:

Fine-tuned model

Generates a prediction or structured response

Raw output

Pydantic validation

Checks fields, types, and allowed values

Validated result or error

Your Python application

Uses the validated data

# Example 5: Converting a Fine-Tuned Model's JSON Text into Pydantic

### Goal

Learn how to handle model outputs that arrive as a JSON string instead of a Python dictionary.

In real applications, a model might return text like this:

Python

Run

```
model_response = """
{
    "category": "billing",
    "sentiment": "neutral",
    "confidence": 0.88
}
"""
```

This is a string, not a Python dictionary.

We can use Pydantic's `model_validate_json()` method to parse and validate JSON text.

Python

Run

```
model_response = """
{
    "category": "billing",
    "sentiment": "neutral",
    "confidence": 0.88
}
"""

result = ClassificationOutput.model_validate_json(model_response)

print(result)
```

Output:

```
category='billing' sentiment='neutral' confidence=0.88
```

### What is the difference?

|
Method

|

Input

|
| --- | --- |
|

`model_validate()`

|

Python dictionary or compatible object

|
|

`model_validate_json()`

|

JSON string or bytes

|

For example:

Python

Run

```
# Python dictionary
ClassificationOutput.model_validate({
    "category": "billing",
    "sentiment": "neutral",
    "confidence": 0.88
})
```

Python

Run

```
# JSON string
ClassificationOutput.model_validate_json("""
{
    "category": "billing",
    "sentiment": "neutral",
    "confidence": 0.88
}
""")
```

Both produce a validated `ClassificationOutput` object.

### Export the result

Python

Run

```
print(result.model_dump())
```

This returns a Python dictionary.

Python

Run

```
print(result.model_dump_json(indent=2))
```

This returns a JSON string.

Important: Pydantic validates the JSON structure and values. It does not guarantee that the model's prediction is factually correct or that its confidence score is calibrated.

# Quick Reference: Important Pydantic Methods

|
Method

|

What it does

|
| --- | --- |
|

`Model(...)`

|

Create a model from keyword arguments

|
|

`Model.model_validate(data)`

|

Validate a Python dictionary or object

|
|

`Model.model_validate_json(text)`

|

Parse and validate JSON text

|
|

`model.model_dump()`

|

Convert a model to a Python dictionary

|
|

`model.model_dump_json()`

|

Convert a model to a JSON string

|
|

`e.errors()`

|

Get detailed validation errors

|

# Practice Exercises

Try these exercises after completing the examples.

## Exercise 1: Create a movie model

Create a Pydantic model named `Movie` with the following fields:

|
Field

|

Type

|

Rule

|
| --- | --- | --- |
|

`title`

|

`str`

|

Required

|
|

`director`

|

`str`

|

Required

|
|

`rating`

|

`float`

|

Between 0 and 10

|
|

`release_year`

|

`int`

|

Between 1900 and 2026

|

Tasks:

* Create a valid movie.

* Try a movie with a rating of `12`.

* Try a movie with a missing title.

* Print the validation errors.

## Exercise 2: Create a nested model

Create two models:

Python

Run

```
class Ingredient(BaseModel):
    name: str
    quantity: float
```

Python

Run

```
class Recipe(BaseModel):
    name: str
    ingredients: list[Ingredient]
```

Create a recipe with at least three ingredients and export it to JSON.

## Exercise 3: Validate a fine-tuned model output

Create a Pydantic model named `SentimentOutput` with:

* `sentiment`: Must be `"positive"`, `"negative"`, or `"neutral"`.

* `confidence`: Must be between `0` and `1`.

* `explanation`: A string.

Test the model with:

Python

Run

```
{
    "sentiment": "positive",
    "confidence": 0.92,
    "explanation": "The customer is happy with the service."
}
```

Then test this invalid output:

Python

Run

```
{
    "sentiment": "very positive",
    "confidence": 1.4,
    "explanation": "The customer is happy."
}
```

Explain why the second output fails validation.

## Final Summary

In this activity, you learned that Pydantic helps us create structured and validated data in Python.

The main ideas are:

1. `BaseModel` defines a data schema.

2. Type annotations describe the expected data types.

3. Pydantic validation checks incoming data.

4. `Field` adds rules such as minimums, maximums, and string lengths.

5. Nested models allow us to represent complex data.

6. `model_validate_json()` is useful when working with JSON text from a model.

7. `model_dump()` and `model_dump_json()` convert validated objects back into formats that applications can use.

### The key connection to fine-tuning

A fine-tuned model learns to generate outputs based on training examples. Pydantic provides a separate way to check whether those outputs follow the structure your application expects.

Model training produces predictions. Pydantic validates the format of those predictions.


<!-- 
# Activity 3: Practical Guide to Pydantic: Core Concepts and Examples

Pydantic is a data validation and parsing library for Python. It uses Python type annotations to validate data schemas, coerce compatible types, and serialize objects.

### Installation

```bash
!pip install pydantic
```

---

## Example 1: Basic Model, Type Coercion, and Validation

This example demonstrates how to define a schema, how Pydantic automatically coerces compatible types, and how it handles invalid data.

### Code

```python
from pydantic import BaseModel, ValidationError

# 1. Define the schema
class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True  # Default value if not provided


# 2. Case A: Valid data with type coercion
valid_payload = {
    "id": "101",            # Passed as a string
    "name": "Alex Smith",
    "email": "alex@example.com"
}

user = User(**valid_payload)
print("User created successfully:")
print(f"ID: {user.id} -> Type: {type(user.id).__name__}")
print(f"Active: {user.is_active}")


# 3. Case B: Invalid data
invalid_payload = {
    "id": "not-an-integer",  # Cannot be converted to int
    "name": "Alex Smith",
    "email": "alex@example.com"
}

try:
    invalid_user = User(**invalid_payload)
except ValidationError as e:
    print("\nValidation failed as expected:")
    print(e)
```

### Code Explanation

1. **`class User(BaseModel):`**
   Inheriting from `BaseModel` marks the class as a Pydantic schema. Pydantic inspects the type annotations (`int`, `str`, `bool`) to generate runtime validation rules.
2. **`is_active: bool = True`**
   Assigning a value directly sets a default. If `"is_active"` is missing from the incoming dictionary, Pydantic defaults it to `True`.
3. **`user = User(**valid_payload)`**
   We unpack the dictionary into keyword arguments. Notice that `"id"` was passed as the string `"101"`. Pydantic attempts **type coercion**: because `"101"` can safely become an integer, it converts it to `101` automatically.
4. **`except ValidationError as e:`**
   When passing `"not-an-integer"` to the `id` field, Pydantic cannot coerce the value. It raises a `ValidationError`, pointing directly to the field name and explaining the error.

---

## Example 2: Field Constraints with `Field`

The `Field` class allows you to attach validation constraints (such as value boundaries and character limits) directly to schema attributes.

### Code

```python
from pydantic import BaseModel, Field, ValidationError

class InventoryItem(BaseModel):
    item_id: str = Field(min_length=3, max_length=10)
    name: str
    price: float = Field(gt=0, description="Price must be strictly greater than 0")
    quantity: int = Field(default=0, ge=0, description="Quantity must be >= 0")


# Case A: Valid item
item = InventoryItem(
    item_id="SKU-100",
    name="Mechanical Keyboard",
    price=99.50,
    quantity=15
)
print("Valid Item:", item)


# Case B: Violating constraints
try:
    bad_item = InventoryItem(
        item_id="SKU-101",
        name="Mouse",
        price=-10.0,   # Violates gt=0
        quantity=-1    # Violates ge=0
    )
except ValidationError as e:
    print("\nConstraint Errors Caught:")
    for err in e.errors():
        print(f"- Field '{err['loc'][0]}': {err['msg']}")
```

### Code Explanation

1. **`Field(min_length=3, max_length=10)`**
   Restricts string length. Strings with fewer than 3 or more than 10 characters are rejected.
2. **`Field(gt=0)` and `Field(ge=0)`**
   * `gt=0` enforces that `price` must be strictly **g**reater **t**han `0` (positive numbers only).
   * `ge=0` enforces that `quantity` must be **g**reater than or **e**qual to `0` (non-negative integers).
3. **`e.errors()`**
   When validation fails, `e.errors()` returns a list of dictionaries where each entry details the exact field (`loc`) and the error reason (`msg`).

---

## Example 3: Nested Models and `model_validate`

In real applications, data structures are rarely flat. Pydantic models can be nested inside other models. This example covers loading dictionaries, validating nested lists, and exporting the results.

### Code

```python
from typing import List
from pydantic import BaseModel

# Sub-model
class Address(BaseModel):
    city: str
    postal_code: str
    country: str = "US"

# Parent model containing a list of sub-models
class Company(BaseModel):
    name: str
    locations: List[Address]


# Raw incoming data (e.g., from an API or database)
payload = {
    "name": "HealthCorp",
    "locations": [
        {"city": "Boston", "postal_code": "02108"},
        {"city": "Chicago", "postal_code": "60601", "country": "US"}
    ]
}

# 1. Parse and validate the dictionary
company = Company.model_validate(payload)

print("Parsed Company Object:")
print(f"Company Name: {company.name}")
print(f"First Branch City: {company.locations[0].city}")
print(f"Type of location entry: {type(company.locations[0]).__name__}")

# 2. Export back to a Python dictionary
company_dict = company.model_dump()
print("\nExported to Dict:", type(company_dict))

# 3. Export to JSON string
company_json = company.model_dump_json(indent=2)
print("\nExported to JSON String:")
print(company_json)
```

### Code Explanation

1. **`locations: List[Address]`**
   This defines a list where **every item must conform to the `Address` schema**. Pydantic validates each item in the list recursively.
2. **`company = Company.model_validate(payload)`**
   * **What it does:** This is the standard Pydantic method to ingest and validate a Python dictionary (or mapping object). 
   * **How it works:** Instead of manually unpacking variables (`Company(**payload)`), `model_validate` checks each top-level key. When it encounters the `locations` key, it reads the list of raw dictionaries and automatically transforms each one into an instance of the `Address` model.
   * If any dictionary inside `locations` is missing a required field (such as `city`), the entire call will fail with a `ValidationError`.
3. **`company.model_dump()`**
   Converts the validated Pydantic model (and all nested models) back into a standard Python `dict`.
4. **`company.model_dump_json(indent=2)`**
   Serializes the model directly into a formatted JSON string. This avoids having to import Python's standard `json` module. 
-->