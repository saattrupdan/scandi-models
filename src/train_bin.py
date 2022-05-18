'''Train a binary classification model.'''

from datasets import load_dataset
from datasets.utils import disable_progress_bar, logging as ds_logging
import transformers.utils.logging as tf_logging
from transformers import ProgressCallback, PrinterCallback
import logging
import sys
from .trainer import get_bin_trainer
from .utils import NeverLeaveProgressCallback


def main():

    # Get arguments
    languages = sys.argv[1:]

    for language in languages:
        print(f'Benchmarking {language} dataset')

        # Disable logging
        disable_progress_bar()
        tf_logging.set_verbosity_error()
        ds_logging.set_verbosity_error()
        logging.getLogger('filelock').setLevel(logging.ERROR)
        logging.getLogger('absl').setLevel(logging.ERROR)
        logging.getLogger('transformers.trainer').setLevel(logging.ERROR)

        # Load the datasets
        dataset_id = f'saattrupdan/grammar-correction-{language}'
        train = load_dataset(dataset_id, split='small_train')
        val = load_dataset(dataset_id, split='val')
        test = load_dataset(dataset_id, split='test')

        # Get the trainer
        # NOTE: These scores are with early stopping patience 10!
        models = [
            'KBLab/bert-base-swedish-cased-new',
            # da=16%
            # nb=0% ??
            # nn=22%
            # sv=64%
            # is=9%
            # fo=%

            'patrickvonplaten/norwegian-roberta-base',
            # da=36%
            # nb=49%
            # nn=45%
            # sv=19%
            # is=%
            # fo=%

            'Maltehb/aelaectra-danish-electra-small-cased',
            # da=60%
            # nb=35%
            # nn=21%
            # sv=9%
            # is=%
            # fo=%

            'vesteinn/IceBERT',
            # da=0%
            # nb=0%
            # nn=8%
            # sv=0%
            # is=%
            # fo=%

            'TurkuNLP/bert-base-finnish-cased-v1',
            # da=0%
            # nb=0%
            # nn=0%
            # sv=0%
            # is=%
            # fo=%

            'vesteinn/ScandiBERT',
            # da=59%
            # nb=61%
            # nn=52%
            # sv=0% ??
            # is=%
            # fo=%
        ]
        for model in models:
            trainer, new_test = get_bin_trainer(
                train=train,
                val=val,
                test=test,
                pretrained_model_id=model,
                new_model_id=f'saattrupdan/grammar-checker-{language}'
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

            # Push model to the Hugging Face Hub
            trainer.push_to_hub()

        print()


if __name__ == '__main__':
    main()
