'''Trainer initialisation'''

import pandas as pd
from typing import Tuple
from datasets import Dataset
import logging
import transformers.utils.logging as tf_logging
from scandeval import load_dataset
from transformers import (Trainer,
                          TrainingArguments,
                          DataCollatorForTokenClassification,
                          EarlyStoppingCallback)

from .model import get_ner_model
from .preprocess import ner_preprocess_data
from .compute_metrics import ner_compute_metrics
from .labels import NER_LABELS


logging_fmt = '%(asctime)s [%(levelname)s] %(message)s'
logger = logging.getLogger(__name__)
#tf_logging.set_verbosity_error()


def get_ner_trainer(df: pd.DataFrame,
                    model_id: str) -> Tuple[Trainer, Dataset]:
    '''Prepare data for training.

    Args:
        df (Pandas DataFrame):
            Input data, with columns 'doc', 'token' and 'ner_tags'.
        model_id (str):
            The model ID of a pretrained model.

    Returns:
        tuple:
            A pair of a trainer and the test dataset, for evaluating the
            performance after training.
    '''
    # Load model and tokenizer
    logger.info('Loading model')
    model, tokenizer = get_ner_model(model_id)

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
        output_dir='.',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        report_to='none',
        save_total_limit=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=1000,
        warmup_steps=len(dataset) * 0.9,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        label_names=NER_LABELS,
        push_to_hub=True,
        push_to_hub_model_id='nbailab-base-scandi-ner'
    )

    # Split the dataset into a training and validation dataset
    split = dataset.train_test_split(0.1, seed=4242)

    # Set up data collator for feeding the data into the model
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Set up early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

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
