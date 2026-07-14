import json
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------
# 1. Project configuration
# ---------------------------------------------------------

MODEL_NAME = "bert-base-uncased"

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
SAVED_MODEL_DIR = BASE_DIR / "saved_model"
METRICS_FILE = BASE_DIR / "metrics.json"

# Keep this True during the first test.
# It trains on a smaller part of the dataset.
FAST_MODE = True

# Samples used only when FAST_MODE is True.
FAST_TRAIN_SAMPLES = 3000
FAST_VALIDATION_SAMPLES = 600
FAST_TEST_SAMPLES = 1000

# AG News label mappings.
ID_TO_LABEL = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology",
}

LABEL_TO_ID = {
    label: label_id for label_id, label in ID_TO_LABEL.items()
}


# ---------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------

def take_sample(dataset, number_of_samples):
    """
    Shuffle a dataset and select a smaller number of rows.

    This is useful for testing the project on a CPU before
    training on the complete dataset.
    """
    sample_size = min(number_of_samples, len(dataset))

    return (
        dataset
        .shuffle(seed=42)
        .select(range(sample_size))
    )


def compute_metrics(evaluation_prediction):
    """
    Calculate accuracy, weighted F1-score and macro F1-score.

    Trainer passes two values:
    1. Prediction logits
    2. Correct labels
    """
    logits, labels = evaluation_prediction

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "f1_weighted": weighted_f1,
        "f1_macro": macro_f1,
    }


# ---------------------------------------------------------
# 3. Main training function
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("NEWS TOPIC CLASSIFIER USING BERT")
    print("=" * 60)

    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Training device: {device_name}")

    # -----------------------------------------------------
    # Load the AG News dataset
    # -----------------------------------------------------

    print("\n1. Downloading and loading AG News dataset...")

    complete_dataset = load_dataset("fancyzhx/ag_news")

    print(complete_dataset)

    # AG News does not provide a separate validation split.
    # We create one from the original training data.
    train_validation_split = complete_dataset["train"].train_test_split(
        test_size=0.1,
        seed=42,
    )

    raw_datasets = DatasetDict(
        {
            "train": train_validation_split["train"],
            "validation": train_validation_split["test"],
            "test": complete_dataset["test"],
        }
    )

    # Use smaller datasets while checking that the project works.
    if FAST_MODE:
        print("\nFAST_MODE is enabled.")
        print("Using smaller dataset samples for initial testing.")

        raw_datasets = DatasetDict(
            {
                "train": take_sample(
                    raw_datasets["train"],
                    FAST_TRAIN_SAMPLES,
                ),
                "validation": take_sample(
                    raw_datasets["validation"],
                    FAST_VALIDATION_SAMPLES,
                ),
                "test": take_sample(
                    raw_datasets["test"],
                    FAST_TEST_SAMPLES,
                ),
            }
        )

    print("\nDataset sizes:")
    print(f"Training examples:   {len(raw_datasets['train'])}")
    print(f"Validation examples: {len(raw_datasets['validation'])}")
    print(f"Testing examples:    {len(raw_datasets['test'])}")

    # -----------------------------------------------------
    # Load the BERT tokenizer
    # -----------------------------------------------------

    print("\n2. Loading BERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_text(batch):
        """
        Convert news text into token IDs understood by BERT.
        """
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    # Tokenize all three dataset splits.
    print("\n3. Tokenizing dataset...")

    tokenized_datasets = raw_datasets.map(
        tokenize_text,
        batched=True,
        remove_columns=["text"],
    )

    # Dynamically pad each batch to the length of its longest text.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # -----------------------------------------------------
    # Load the pretrained BERT model
    # -----------------------------------------------------

    print("\n4. Loading bert-base-uncased model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # One epoch is enough for the first working test.
    # Use two epochs when running the final version.
    number_of_epochs = 1 if FAST_MODE else 2

    # -----------------------------------------------------
    # Configure training
    # -----------------------------------------------------

    training_arguments = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        num_train_epochs=number_of_epochs,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=25,

        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,

        save_total_limit=1,
        report_to="none",

        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,

        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],

        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # -----------------------------------------------------
    # Train the model
    # -----------------------------------------------------

    print("\n5. Starting BERT fine-tuning...")

    trainer.train()

    # -----------------------------------------------------
    # Evaluate using the test dataset
    # -----------------------------------------------------

    print("\n6. Evaluating the trained model...")

    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test",
    )

    print("\nFinal test results:")

    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")

    # -----------------------------------------------------
    # Save the model and tokenizer
    # -----------------------------------------------------

    print("\n7. Saving trained model...")

    trainer.save_model(str(SAVED_MODEL_DIR))
    tokenizer.save_pretrained(str(SAVED_MODEL_DIR))

    # Save evaluation results to a JSON file.
    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=4)

    print("\nTraining completed successfully.")
    print(f"Model saved in: {SAVED_MODEL_DIR}")
    print(f"Metrics saved in: {METRICS_FILE}")


if __name__ == "__main__":
    main()