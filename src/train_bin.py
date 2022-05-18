'''Train a binary classification model.'''

from datasets import load_dataset
from datasets.utils import disable_progress_bar
import transformers.utils.logging as tf_logging
from .trainer import get_bin_trainer


def main():

    disable_progress_bar()
    tf_logging.set_verbosity_error()

    # Load the datasets
    dataset_id = 'saattrupdan/grammar-correction-is'
    train = load_dataset(dataset_id, split='small_train')
    val = load_dataset(dataset_id, split='val')
    test = load_dataset(dataset_id, split='test')

    # Get the trainer
    models = [
        'KBLab/bert-base-swedish-cased-new',  # da=0%, nb=0%, sv=68%, is=%, fo=%
        'NbAiLab/nb-bert-base',  # da=46%, nb=66%, sv=61%, is=%, fo=%
        'Maltehb/danish-bert-botxo',  # da=24%, nb=5%, sv=0%, is=%, fo=%
        'vesteinn/IceBERT',  # da=0%, nb=0%, sv=0%, is=%, fo=%
        'TurkuNLP/bert-base-finnish-cased-v1',  # da=0%, nb=0%, sv=0%, is=%, fo=%
        'vesteinn/ScandiBERT'  # da=56%, nb=62%, sv=62%, is=%, fo=%
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
