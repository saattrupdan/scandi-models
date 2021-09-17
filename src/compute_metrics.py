'''Script with compute_metrics functions for `transformers` training'''

from typing import Dict
import numpy as np
from datasets import load_metric
from .labels import NER_LABELS


def ner_compute_metrics(predictions_and_labels: tuple) -> Dict[str, float]:
    '''Compute the metrics needed for NER evaluation.

    Args:
        predictions_and_labels (pair of arrays):
            The first array contains the probability predictions and the
            second array contains the true labels.

    Returns:
        dict:
            A dictionary with the names of the metrics as keys and the
            metric values as values.
    '''
    metric = load_metric('seqeval')

    # Get the predictions from the model
    predictions, labels = predictions_and_labels

    raw_predictions = np.argmax(predictions, axis=-1)

    # Remove ignored index (special tokens)
    predictions = [
        [NER_LABELS[pred] for pred, lbl in zip(prediction, label)
            if lbl != -100]
        for prediction, label in zip(raw_predictions, labels)
    ]
    labels = [
        [NER_LABELS[lbl] for _, lbl in zip(prediction, label)
            if lbl != -100]
        for prediction, label in zip(raw_predictions, labels)
    ]

    results = metric.compute(predictions=predictions, references=labels)

    # Remove MISC labels from predictions
    for i, prediction_list in enumerate(predictions):
        for j, ner_tag in enumerate(prediction_list):
            if ner_tag[-4:] == 'MISC':
                predictions[i][j] = 'O'

    # Remove MISC labels from labels
    for i, label_list in enumerate(labels):
        for j, ner_tag in enumerate(label_list):
            if ner_tag[-4:] == 'MISC':
                labels[i][j] = 'O'

    results_no_misc = metric.compute(predictions=predictions,
                                     references=labels)

    return dict(micro_f1=results["overall_f1"],
                micro_f1_no_misc=results_no_misc['overall_f1'])
