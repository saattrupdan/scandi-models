'''Train a binary classification model.'''

from datasets import load_dataset
from trainer import get_bin_trainer


def main():

    # Load the dataset dictionary
    dataset_dict = load_dataset('saattrupdan/grammar-correction-da')

    # Extract the training, validation and test splits
    train = dataset_dict['small_train']
    val = dataset_dict['val']
    test = dataset_dict['test']

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
