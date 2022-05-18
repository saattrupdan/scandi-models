'''Model loading'''

from typing import Tuple
from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModelForTokenClassification,
                          AutoModelForSequenceClassification,
                          PreTrainedModel,
                          PreTrainedTokenizer)
from .labels import NER_LABELS, BIN_LABELS


def get_bin_model(model_id: str) -> Tuple[PreTrainedModel,
                                          PreTrainedTokenizer]:
    '''Load a BIN model.

    Args:
        model_id (str): The model ID of the pretrained model.

    Returns:
        tuple:
            The pretrained model and the pretrained tokenizer
    '''
    config = dict(num_labels=len(BIN_LABELS),
                  id2label={id:lbl for id, lbl in enumerate(BIN_LABELS)},
                  label2id={lbl:id for id, lbl in enumerate(BIN_LABELS)})
    config = AutoConfig.from_pretrained(model_id, **config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               config=config)
    return model, tokenizer


def get_ner_model(model_id: str) -> Tuple[PreTrainedModel,
                                          PreTrainedTokenizer]:
    '''Load a NER model.

    Args:
        model_id (str): The model ID of the pretrained model.

    Returns:
        tuple:
            The pretrained model and the pretrained tokenizer
    '''
    config = dict(num_labels=len(NER_LABELS),
                  id2label={id:lbl for id, lbl in enumerate(NER_LABELS)},
                  label2id={lbl:id for id, lbl in enumerate(NER_LABELS)})
    config = AutoConfig.from_pretrained(model_id, **config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id,
                                                            config=config)
    return model, tokenizer
