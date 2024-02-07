import os
import boto3
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer, pipeline,
)
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import s3fs
import json
from sagemaker.remote_function import remote
import sagemaker

sm_session = sagemaker.Session(boto_session=boto3.session.Session(region_name="us-east-1"))
s3_root_folder = f"s3://{sm_session.default_bucket()}/remote_function_demo/huggingface"
settings = dict(
    sagemaker_session=sm_session,
    role="AmazonSageMaker-ExecutionRole-20240207T090351",  # REPLACE WITH YOUR OWN ROLE HERE
    instance_type="ml.g5.xlarge",
    dependencies='./requirements.txt',
    s3_root_uri=s3_root_folder
)


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


@remote(**settings)
def train_hf_model(
        train_input_path,
        test_input_path,
        s3_output_path=None,
        *,
        epochs=1,
        train_batch_size=32,
        eval_batch_size=64,
        warmup_steps=500,
        learning_rate=5e-5,
):
    model_dir = "model"

    train_data = load_from_disk(train_input_path, keep_in_memory=True)
    test_data = load_from_disk(test_input_path, keep_in_memory=True)

    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        logging_dir="logs/",
        learning_rate=float(learning_rate),
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
    )

    print("Starting model training..")
    trainer.train()

    trainer.save_model(model_dir)

    print("Evaluating the model...")
    eval_result = trainer.evaluate(eval_dataset=test_data)

    if s3_output_path:
        fs = s3fs.S3FileSystem()
        with fs.open(os.path.join(s3_output_path, "eval_results.txt"), "w") as file:
            json.dump(eval_result, file)

        fs.put(model_dir, os.path.join(s3_output_path, model_dir), recursive=True)

    return os.path.join(s3_output_path, model_dir), eval_result


if __name__ == "__main__":
    # tokenizer used in preprocessing
    tokenizer_name = "distilbert-base-uncased"

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # download dataset
    train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])

    # for demo, smaller the size of the datasets
    test_dataset = test_dataset.shuffle().select(range(5000))

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_data_path = os.path.join(s3_root_folder, "data", "train")
    test_data_path = os.path.join(s3_root_folder, "data", "test")

    train_dataset.save_to_disk(train_data_path)
    test_dataset.save_to_disk(test_data_path)

    # train the model
    model_path, evaluation = train_hf_model(
        train_data_path, test_data_path, os.path.join(s3_root_folder, "run_1/output")
    )

    # classify text using our trained model
    print("Downloading the model from S3. It will take few minutes.")
    fs = s3fs.S3FileSystem()
    fs.get(model_path, "model", recursive=True)
    print("Model downloaded successfully.")

    trained_model = AutoModelForSequenceClassification.from_pretrained("model")

    inputs = "I love using SageMaker."
    classifier = pipeline("text-classification", model=trained_model, tokenizer=tokenizer)

    result = classifier(inputs)
    print(f"*** result: {result} ***")
