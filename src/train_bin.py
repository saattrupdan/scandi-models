'''Train a binary classification model.'''

from datasets import load_dataset
from datasets.utils import disable_progress_bar, logging as ds_logging
import transformers.utils.logging as tf_logging
from transformers import ProgressCallback, PrinterCallback
import logging
from .trainer import get_bin_trainer
from .utils import NeverLeaveProgressCallback


def main():

    # Disable logging
    disable_progress_bar()
    tf_logging.set_verbosity_error()
    ds_logging.set_verbosity_error()
    logging.getLogger('filelock').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    logging.getLogger('transformers.trainer').setLevel(logging.ERROR)

    # Load the datasets
    dataset_id = 'saattrupdan/grammar-correction-fo'
    train = load_dataset(dataset_id, split='small_train')
    val = load_dataset(dataset_id, split='val')
    test = load_dataset(dataset_id, split='test')

    # Get the trainer
    models = [
        'KBLab/bert-base-swedish-cased-new',  # da=0%, nb=0%, nn=%, sv=68%, is=0%, fo=%
        'NbAiLab/nb-bert-base',  # da=46%, nb=66%, nn=%, sv=61%, is=4%, fo=%
        'Maltehb/danish-bert-botxo',  # da=24%, nb=5%, nn=%, sv=0%, is=0%, fo=%
        'vesteinn/IceBERT',  # da=0%, nb=0%, nn=%, sv=0%, is=0%, fo=%
        'TurkuNLP/bert-base-finnish-cased-v1',  # da=0%, nb=0%, nn=%, sv=0%, is=0%, fo=%
        'vesteinn/ScandiBERT'  # da=56%, nb=62%, nn=%, sv=62%, is=61%, fo=%
    ]
    for model in models:
        trainer, new_test = get_bin_trainer(
            train=train,
            val=val,
            test=test,
            pretrained_model_id=model,
            new_model_id='saattrupdan/grammar-checker-da'
        )

        # Set transformers logging back to error
        tf_logging.set_verbosity_error()

        # Remove trainer logging
        trainer.log = lambda _: None

        # Remove the callback which prints the metrics after each evaluation
        trainer.remove_callback(PrinterCallback)

        # Remove the progress bar callback
        trainer.remove_callback(ProgressCallback)

        trainer.add_callback(NeverLeaveProgressCallback)

        # Train the model
        trainer.train()

        # Evaluate the model on the test set
        metrics = trainer.evaluate(new_test)

        print(model)
        print(metrics)


if __name__ == '__main__':
    main()
