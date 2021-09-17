'''Trainer initialisation'''

import pandas as pd
from typing import Tuple
from datasets import Dataset
import logging
import transformers.utils.logging as tf_logging
from transformers import (Trainer,
                          TrainingArguments,
                          DataCollatorForTokenClassification,
                          EarlyStoppingCallback)

from .model import get_ner_model
from .preprocess import ner_preprocess_data
from .compute_metrics import ner_compute_metrics


logging_fmt = '%(asctime)s [%(levelname)s] %(message)s'
logger = logging.getLogger(__name__)
tf_logging.set_verbosity_error()


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
            A pair of a trainer and the evaluation dataset, for evaluating the
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
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=1000,
        warmup_steps=len(dataset) * 0.9,
        gradient_accumulation_steps=1,
        load_best_model_at_end=True
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

    # Return the trainer, the training dataset and the validation dataset
    return trainer, split['test']
