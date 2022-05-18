'''Train a binary classification model.'''

from datasets import load_dataset
from datasets.utils import disable_progress_bar
import transformers.logging as tf_logging
from .trainer import get_bin_trainer


def main():

    disable_progress_bar()
    tf_logging.set_verbosity_error()

    # Load the datasets
    dataset_id = 'saattrupdan/grammar-correction-da'
    train = load_dataset(dataset_id, split='small_train')
    val = load_dataset(dataset_id, split='val')
    test = load_dataset(dataset_id, split='test')

    # Get the trainer
    models = [
        'KBLab/bert-base-swedish-cased-new',  # 0%
        'NbAiLab/nb-bert-base',  # 60%
        'Maltehb/danish-bert-botxo',  # 27%
        'vesteinn/IceBERT',  # 0%
        'TurkuNLP/bert-base-finnish-cased-v1',  # 0%
        'vesteinn/ScandiBERT'  # 66%
    ]
    for model in models:
        trainer, new_test = get_bin_trainer(
            train=train,
            val=val,
            test=test,
            pretrained_model_id=model,
            new_model_id='saattrupdan/grammar-checker-da'
        )

        # Train the model
        trainer.train()

        # Evaluate the model on the test set
        metrics = trainer.evaluate(new_test)

        print(model)
        print(metrics)


if __name__ == '__main__':
    main()
