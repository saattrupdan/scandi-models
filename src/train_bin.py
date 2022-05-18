'''Train a binary classification model.'''

from datasets import load_dataset
from .trainer import get_bin_trainer


def main():

    # Load the datasets
    dataset_id = 'saattrupdan/grammar-correction-da'
    train = load_dataset(dataset_id, split='train')
    val = load_dataset(dataset_id, split='val')
    test = load_dataset(dataset_id, split='test')

    breakpoint()

    # Get the trainer
    trainer = get_bin_trainer(
        train=train,
        val=val,
        pretrained_model_id='Maltehb/aelaectra-danish-electra-small-cased',
        new_model_id='saattrupdan/grammar-checker-da'
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the test set
    metrics = trainer.evaluate(test)

    print(metrics)

    breakpoint()


if __name__ == '__main__':
    main()
