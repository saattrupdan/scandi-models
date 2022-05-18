'''Trainer initialisation'''

import pandas as pd
from typing import Tuple
from datasets import Dataset
#from scandeval import load_dataset
from transformers import (Trainer,
                          TrainingArguments,
                          DataCollatorForTokenClassification,
                          DataCollatorWithPadding,
                          EarlyStoppingCallback)

from .model import get_ner_model, get_bin_model
from .preprocess import ner_preprocess_data, bin_preprocess_data
from .compute_metrics import ner_compute_metrics, bin_compute_metrics


def get_bin_trainer(train: Dataset,
                    val: Dataset,
                    pretrained_model_id: str,
                    new_model_id: str) -> Tuple[Trainer, Dataset]:
    '''Prepare data for BIN training.

    Args:
        train (Dataset):
            Input training data, with columns 'text' and 'label'.
        val (Dataset):
            Input validation data, with columns 'text' and 'label'.
        pretrained_model_id (str):
            The model ID of a pretrained model.
        new_model_id (str):
            The model ID of the new finetuned model.

    Returns:
        tuple:
            A pair of a trainer and the test dataset, for evaluating the
            performance after training.
    '''
    # Load model and tokenizer
    logger.info('Loading model')
    model, tokenizer = get_bin_model(pretrained_model_id)

    # Tokenize and align labels
    logger.info('Tokenising dataset')
    train = bin_preprocess_data(train, tokenizer)
    val = bin_preprocess_data(val, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=new_model_id,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        report_to='none',
        save_total_limit=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=100,
        warmup_ratio=0.1,
        gradient_accumulation_steps=4,
        metric_for_best_model='mcc',
        load_best_model_at_end=True
    )

    # Set up data collator for feeding the data into the model
    data_collator = DataCollatorWithPadding(tokenizer, padding='longest')

    # Set up early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    # Initialise the Trainer object
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train,
                      eval_dataset=val,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics=bin_compute_metrics,
                      callbacks=[early_stopping])

    # Return the trainer
    return trainer


def get_ner_trainer(df: pd.DataFrame,
                    pretrained_model_id: str,
                    new_model_id: str) -> Tuple[Trainer, Dataset]:
    '''Prepare data for NER training.

    Args:
        df (Pandas DataFrame):
            Input data, with columns 'doc', 'token' and 'ner_tags'.
        pretrained_model_id (str):
            The model ID of a pretrained model.
        new_model_id (str):
            The model ID of the new finetuned model.

    Returns:
        tuple:
            A pair of a trainer and the test dataset, for evaluating the
            performance after training.
    '''
    # Load model and tokenizer
    logger.info('Loading model')
    model, tokenizer = get_ner_model(pretrained_model_id)

    # Convert dataframe to HuggingFace Dataset
    dataset_dct = dict(doc=df.doc,
                       tokens=df.tokens,
                       orig_labels=df.ner_tags)
    dataset = Dataset.from_dict(dataset_dct)

    # Tokenize and align labels
    logger.info('Tokenising and aligning dataset')
    dataset = ner_preprocess_data(dataset, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=new_model_id,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        report_to='none',
        save_total_limit=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=100,
        warmup_steps=len(dataset) * 0.9 / 32,
        gradient_accumulation_steps=4,
        metric_for_best_model='micro_f1',
        load_best_model_at_end=True,
        push_to_hub=True
    )

    # Split the dataset into a training and validation dataset
    split = dataset.train_test_split(0.1, seed=4242)

    # Set up data collator for feeding the data into the model
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Set up early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)

    # Initialise the Trainer object
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=split['train'],
                      eval_dataset=split['test'],
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics=ner_compute_metrics,
                      callbacks=[early_stopping])

    # Set up test dataset
    _, dane_X_test, _, dane_y_test = load_dataset('dane')
    test_df = pd.concat((dane_X_test, dane_y_test), axis=1)
    test_dataset_dct = dict(doc=test_df.doc,
                            tokens=test_df.tokens,
                            orig_labels=test_df.ner_tags)
    test_dataset = Dataset.from_dict(test_dataset_dct)
    test_dataset = ner_preprocess_data(test_dataset, tokenizer)

    # Return the trainer, the training dataset and the validation dataset
    return trainer, test_dataset
