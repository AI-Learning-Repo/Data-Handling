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