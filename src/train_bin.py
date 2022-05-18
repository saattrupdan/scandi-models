'''Train a binary classification model.'''

from datasets import load_dataset
from .trainer import get_bin_trainer


def main():

    # Load the datasets
    dataset_id = 'saattrupdan/grammar-correction-da'
    train = load_dataset(dataset_id, split='train')
    val = load_dataset(dataset_id, split='val')
    test = load_dataset(dataset_id, split='test')

    # Get the trainer
    models = [
        'KBLab/bert-base-swedish-cased-new',  # 0%
        'NbAiLab/nb-bert-base',  # 57%
        'Maltehb/danish-bert-botxo',  #
        'vesteinn/IceBERT',  # 75% acc
        'TurkuNLP/bert-base-finnish-cased-v1',  # 75% acc
        'vesteinn/ScandiBERT'  # 83% acc
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
