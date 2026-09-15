# DataFrame to csv

### Save Features (`X`) Alone

Since `X` is a pandas DataFrame, you can save it directly as a `.csv` file. 

```python
# Save features alone
X.to_csv("features_X.csv", index=False)
print("Features saved as features_X.csv")
```
*(Note: `index=False` prevents pandas from adding an unnecessary row index column like `Unnamed: 0`)*.

### Merge Features and Target, Then Save

You can combine `X` (DataFrame) and `y` (Series) side-by-side along the columns using `pd.concat()` or `.assign()`:

#### Option A: Using `pd.concat` (Most Common)
```python
import pandas as pd

# Concatenate side-by-side along columns (axis=1)
df_merged = pd.concat([X, y], axis=1)

# Save the combined DataFrame
df_merged.to_csv("complete_dataset.csv", index=False)
print("Merged dataset saved as complete_dataset.csv")

# Quick check
df_merged.head()
```

#### Option B: Using `.assign()` (Clean One-Liner)
```python
# Create a new DataFrame with mpg added, then save
df_merged = X.assign(mpg=y)
df_merged.to_csv("complete_dataset.csv", index=False)
```
