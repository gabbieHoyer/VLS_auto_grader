
Experiment list:

train:
python src/train_multigrader.py --config_file nonhierarchical_singlegrader_nossl.yaml


eval:
python src/eval_multigrader.py --config_file nonhierarchical_singlegrader_nossl.yaml --checkpoint_path runs/vls_grading/<run_id>/checkpoints/best_model.pth --split test


Scenario Complexity Analysis
Single-Grader vs. Multi-Grader:
Single-Grader: Uses one grader’s labels (e.g., Sean_Review), reducing the complexity of label aggregation and metric computation. The updates to MultiGraderDataset, TrainingEngine, and EvalEngine ensure that single-grader mode duplicates labels for compatibility, simplifying the logic.
Multi-Grader: Involves label aggregation (taking the more severe label) and computing metrics for both graders (Sean_Review and Santiago_Review), adding complexity to debugging if there are discrepancies between graders.
Hierarchical vs. Non-Hierarchical:
Hierarchical: Involves splitting labels into base (1, 2, 3, 4) and subclasses (a, b, c, d, none), requiring additional logic for label parsing, aggregation, and dual-output model predictions (base and subclass logits). This increases the risk of bugs in label handling or model output processing.
Non-Hierarchical: Treats labels as flat classes (1, 2, 2b, 2c, 3, 3b, 3c, 4b), simplifying label mapping and model output (single set of logits). This reduces complexity in the dataset and engine logic.
SSL Pretraining vs. No SSL Pretraining:
SSL Pretraining + Training: Adds a pretraining phase where the model learns representations using self-supervised learning (e.g., contrastive loss on augmented video pairs), followed by supervised fine-tuning. This introduces additional complexity in the training loop, data augmentation, and loss computation.
No SSL Pretraining: Skips the pretraining phase, directly training the model with supervised learning. This simplifies the training process, as it only involves standard classification loss and metric computation.
Simplest Scenario for Testing and Debugging
Based on the analysis, the simplest scenario to test and debug your system is:

Non-Hierarchical, Single-Grader, No SSL Pretraining
Why this scenario?

Non-Hierarchical: Avoids the complexity of base/subclass label parsing and dual-output model predictions. The model outputs a single set of logits for flat classes (1, 2, 2b, etc.), and the dataset maps labels directly to indices.
Single-Grader: Eliminates the need for label aggregation between two graders. The dataset duplicates the single grader’s labels (e.g., Sean_Review), and the engine skips second-grader metrics, reducing potential points of failure.
No SSL Pretraining: Skips the self-supervised pretraining phase, simplifying the training loop to standard supervised learning with cross-entropy loss. This allows you to focus on the core supervised pipeline (data loading, model training, metric computation, and evaluation).