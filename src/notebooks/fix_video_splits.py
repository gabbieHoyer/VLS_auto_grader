# fix_video_splits.py
# Cleans video_splits.csv, deduplicates by Video_ID, and reassigns train/val/test splits with 70/15/15 ratios.
# Handles rare classes by assigning them to train, then stratifies remaining data by Sean_Review.
# Preserves all original columns, only updating the 'split' column.
# Saves updated dataset as video_splits_corrected.csv.
# Compatible with dataset.py, train.py, and TrainingEngine.

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
DATASET_PATH = '/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_grader/data/video_splits.csv'
OUTPUT_DIR = '/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_grader/data/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names (grader-specific)
SEAN_CLASS_NAMES = ['1', '2', '2b', '2c', '3', '3b', '3c', '4b']
SEAN_NUM_CLASSES = len(SEAN_CLASS_NAMES)
SEAN_CLASS_TO_IDX = {name: idx for idx, name in enumerate(SEAN_CLASS_NAMES)}

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, 'Split ratios must sum to 1.0'

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
    print(f'Loaded dataset with {len(df)} samples.')
except FileNotFoundError:
    print(f'Error: {DATASET_PATH} not found. Exiting.')
    exit(1)

# Verify columns
expected_columns = ['Video_ID', 'Processed_video_path', 'Sean_Review', 'Santiago_Review', 'split']
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f'Dataset must contain columns: {expected_columns}')

# Log all columns in the dataset
print(f'\nColumns in dataset: {list(df.columns)}')

# Clean dataset: Drop Sean_Review='x'
initial_rows = len(df)
df = df[df['Sean_Review'] != 'x'].copy()
print(f'Dropped {initial_rows - len(df)} rows with Sean_Review="x". Rows remaining: {len(df)}')

# Deduplicate by Video_ID
initial_rows = len(df)
dedup_df = df.drop_duplicates(subset='Video_ID', keep='first')
final_rows = len(dedup_df)
print(f'\nDeduplication:')
print(f'Initial rows: {initial_rows}')
print(f'Final rows after deduplication: {final_rows}')
print(f'Duplicates dropped: {initial_rows - final_rows}')
print(f'Unique Video_IDs: {len(dedup_df["Video_ID"].unique())}')
print(f'Unique Sean_Review values: {sorted(dedup_df["Sean_Review"].dropna().unique())}')

# Validate Sean_Review values
invalid_sean_reviews = dedup_df[~dedup_df['Sean_Review'].isin(SEAN_CLASS_NAMES)]['Sean_Review'].unique()
if len(invalid_sean_reviews) > 0:
    print(f'Warning: Found invalid Sean_Review values: {invalid_sean_reviews}')

# Handle rare classes (fewer than 2 samples)
class_counts = dedup_df['Sean_Review'].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()
print(f'\nRare classes (fewer than 2 samples): {rare_classes}')

# Split data: rare classes to train, stratify the rest
if rare_classes:
    rare_df = dedup_df[dedup_df['Sean_Review'].isin(rare_classes)]
    non_rare_df = dedup_df[~dedup_df['Sean_Review'].isin(rare_classes)]
    print(f'Assigning {len(rare_df)} rare class samples to train split.')
else:
    rare_df = pd.DataFrame(columns=dedup_df.columns)
    non_rare_df = dedup_df

# Stratify non-rare data
if len(non_rare_df) > 0:
    # Include all columns except 'Sean_Review' (for stratification) and 'split' (to be reassigned)
    feature_columns = [col for col in non_rare_df.columns if col not in ['Sean_Review', 'split']]
    X = non_rare_df[feature_columns]
    y = non_rare_df['Sean_Review']

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=42
    )

    # Second split: train vs val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, stratify=y_temp, random_state=42
    )

    # Combine splits, preserving all columns
    train_df = pd.concat([X_train, y_train.rename('Sean_Review')], axis=1)
    val_df = pd.concat([X_val, y_val.rename('Sean_Review')], axis=1)
    test_df = pd.concat([X_test, y_test.rename('Sean_Review')], axis=1)

    # Include rare classes in train
    train_df = pd.concat([train_df, rare_df], ignore_index=True)
else:
    # All data is rare
    train_df = rare_df
    val_df = pd.DataFrame(columns=dedup_df.columns)
    test_df = pd.DataFrame(columns=dedup_df.columns)

# Assign split labels
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

# Merge back to original DataFrame, ensuring all columns are preserved
updated_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Reorder columns to match original DataFrame
updated_df = updated_df[dedup_df.columns]

# Verify columns
print(f'\nColumns in updated dataset: {list(updated_df.columns)}')
if set(updated_df.columns) != set(dedup_df.columns):
    print(f'Warning: Column mismatch between original and updated DataFrame.')
    print(f'Original columns: {list(dedup_df.columns)}')
    print(f'Updated columns: {list(updated_df.columns)}')

# Verify split sizes
split_sizes = {
    'Split': ['Train', 'Validation', 'Test', 'Total'],
    'Size': [len(train_df), len(val_df), len(test_df), len(updated_df)],
    'Proportion': [
        len(train_df) / len(updated_df) if len(updated_df) > 0 else 0,
        len(val_df) / len(updated_df) if len(updated_df) > 0 else 0,
        len(test_df) / len(updated_df) if len(updated_df) > 0 else 0,
        1.0
    ]
}
split_sizes_df = pd.DataFrame(split_sizes)
print('\nSplit Sizes:')
print(split_sizes_df)

# Save updated dataset
output_path = os.path.join(OUTPUT_DIR, 'video_splits_corrected.csv')
updated_df.to_csv(output_path, index=False)
print(f'\nSaved corrected dataset to: {output_path}')