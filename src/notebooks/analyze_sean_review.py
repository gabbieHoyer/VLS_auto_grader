# analyze_sean_review.py
# Analyzes class distribution of Sean_Review and Santiago_Review in video_splits_corrected.csv.
# Uses grader-specific class lists for accurate distribution analysis.
# Generates tables, plots, and grader disagreement analysis.
# Compatible with dataset.py, train.py, and TrainingEngine.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATASET_PATH = '/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_grader/data/video_splits_corrected.csv'
OUTPUT_DIR = '/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_grader/work_dir/finetuning/VLS-3D-Grading/i3d_ssl_supervised/Run_1/figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grader-specific class names
SEAN_CLASS_NAMES = ['1', '2', '2b', '2c', '3', '3b', '3c', '4b']
SANTIAGO_CLASS_NAMES = ['1', '2', '2b', '2c', '2d', '3', '3b', '4b']
SEAN_NUM_CLASSES = len(SEAN_CLASS_NAMES)
SANTIAGO_NUM_CLASSES = len(SANTIAGO_CLASS_NAMES)

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

# Validate grader values
invalid_sean_reviews = df[~df['Sean_Review'].isin(SEAN_CLASS_NAMES)]['Sean_Review'].unique()
invalid_santiago_reviews = df[~df['Santiago_Review'].isin(SANTIAGO_CLASS_NAMES)]['Santiago_Review'].unique()
if len(invalid_sean_reviews) > 0:
    print(f'Warning: Found invalid Sean_Review values: {invalid_sean_reviews}')
if len(invalid_santiago_reviews) > 0:
    print(f'Warning: Found invalid Santiago_Review values: {invalid_santiago_reviews}')

# Extract splits
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

# Compute split sizes
split_sizes = {
    'Split': ['Train', 'Validation', 'Test', 'Total'],
    'Size': [len(train_df), len(val_df), len(test_df), len(df)],
    'Proportion': [
        len(train_df) / len(df) if len(df) > 0 else 0,
        len(val_df) / len(df) if len(df) > 0 else 0,
        len(test_df) / len(df) if len(df) > 0 else 0,
        1.0
    ]
}
split_sizes_df = pd.DataFrame(split_sizes)
print('\nSplit Sizes:')
print(split_sizes_df)
split_sizes_df.to_csv(os.path.join(OUTPUT_DIR, 'split_sizes.csv'), index=False)

# Analyze class distribution
class_counts = {
    'Split': [],
    'Grader': [],
    'Class': [],
    'Count': [],
    'Proportion': []
}

for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df), ('Total', df)]:
    for grader, class_names in [('Sean_Review', SEAN_CLASS_NAMES), ('Santiago_Review', SANTIAGO_CLASS_NAMES)]:
        counts = split_df[grader].value_counts().reindex(class_names, fill_value=0)
        proportions = counts / counts.sum() if counts.sum() > 0 else counts
        for class_name in class_names:
            class_counts['Split'].append(split_name)
            class_counts['Grader'].append(grader)
            class_counts['Class'].append(class_name)
            class_counts['Count'].append(counts.get(class_name, 0))
            class_counts['Proportion'].append(proportions.get(class_name, 0))

class_counts_df = pd.DataFrame(class_counts)
print('\nClass Distribution:')
print(class_counts_df)
class_counts_df.to_csv(os.path.join(OUTPUT_DIR, 'class_distribution.csv'), index=False)

# Visualize class distribution
plt.figure(figsize=(14, 6))
g = sns.catplot(data=class_counts_df, x='Split', y='Count', hue='Class', col='Grader', kind='bar', height=5, aspect=1.2)
g.set_titles('{col_name}')
plt.suptitle('Class Counts per Split by Grader', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_counts_bar.png'))
plt.close()

# Check for class imbalance (Sean_Review only, used for training)
sean_counts = class_counts_df[(class_counts_df['Grader'] == 'Sean_Review') & (class_counts_df['Split'] == 'Total')]
imbalanced_classes = sean_counts[sean_counts['Proportion'] < 0.1]
if not imbalanced_classes.empty:
    print('\nImbalanced Classes Detected (Sean_Review):')
    print(imbalanced_classes)
    print('\nRecommendations:')
    print('- Use weighted loss in train.py (e.g., weights = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0]).to(device)).')
    print('- Evaluate with per-class F1-score in metrics.py.')
    print('- Add temporal jittering in augmentations.py for minority classes.')
else:
    print('\nNo significant class imbalance detected.')

# Compare grader agreement
agreement = (df['Sean_Review'] == df['Santiago_Review']).mean()
print(f'\nGrader Agreement: {agreement:.2%}')
disagreements = df[df['Sean_Review'] != df['Santiago_Review']][['Video_ID', 'Processed_video_path', 'Sean_Review', 'Santiago_Review', 'split']]
if not disagreements.empty:
    print('\nDisagreement Cases:')
    print(disagreements)
    disagreements.to_csv(os.path.join(OUTPUT_DIR, 'grader_disagreements.csv'), index=False)