'''Preprocess data'''

from datasets import Dataset
from .labels import NER_LABELS, SENT_LABELS


def sent_preprocess_data(dataset: Dataset, tokenizer) -> Dataset:
    '''Preprocess a dataset to sentiment analysis by tokenizing the labels.

    Args:
        dataset (HuggingFace dataset):
            The dataset to preprocess.
        tokenizer (HuggingFace tokenizer):
            A pretrained tokenizer.

    Returns:
        HuggingFace dataset: The preprocessed dataset.
    '''
    def tokenise(examples: dict) -> dict:
        doc = examples['doc']
        return tokenizer(doc,
                         truncation=True,
                         padding=True,
                         max_length=512)

    tokenised = dataset.map(tokenise, batched=True)

    def create_numerical_labels(examples: dict) -> dict:
        conversion_dict = dict(Negative='Negative',
                               negativ='Negative',
                               neutral='Neutral',
                               Neutral='Neutral',
                               positiv='Positive',
                               Positive='Positive')
        examples['label'] = [SENT_LABELS.index(conversion_dict[lbl])
                             for lbl in examples['orig_label']]
        return examples

    preprocessed = tokenised.map(create_numerical_labels, batched=True)

    return preprocessed.remove_columns(['doc', 'orig_label'])


def ner_preprocess_data(dataset: Dataset,
                        tokenizer,
                        only_label_first_subtoken: bool = True) -> Dataset:
    '''Preprocess a dataset to NER by tokenizing and aligning the labels.

    Args:
        dataset (HuggingFace dataset):
            The dataset to preprocess.
        tokenizer (HuggingFace tokenizer):
            A pretrained tokenizer.
        only_label_first_subtoken (bool, optional):
            Whether only the first subtoken in each token should be labelled,
            with the rest of the subtokens being ignored for the purposes of
            evaluation. Useful for test sets, not for training sets. Defaults
            to True.

    Returns:
        HuggingFace dataset: The preprocessed dataset.
    '''

    label2id = {lbl: idx for idx, lbl in enumerate(NER_LABELS)}

    def tokenize_and_align_labels(examples: dict):
        '''Tokenise all texts and align the labels with them.

        Args:
            examples (dict):
                The examples to be tokenised.

        Returns:
            dict:
                A dictionary containing the tokenized data as well as labels.
        '''
        tokenized_inputs = tokenizer(
            examples['tokens'],
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word)
            is_split_into_words=True,
            truncation=True,
            max_length=512
        )
        all_labels = []
        for i, labels in enumerate(examples['orig_labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:

                # Special tokens have a word id that is None. We set the label
                # to -100 so they are automatically ignored in the loss
                # function
                if word_idx is None:
                    label_ids.append(-100)

                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label = labels[word_idx]
                    label_id = label2id[label]
                    label_ids.append(label_id)

                # For the other tokens in a word set the label to -100, unless
                # `only_label_first_subtoken` is True.
                else:
                    if only_label_first_subtoken:
                        label_ids.append(-100)
                    else:
                        label = labels[word_idx]
                        label_id = label2id[label]
                        label_ids.append(label_id)

                previous_word_idx = word_idx

            all_labels.append(label_ids)
        tokenized_inputs['labels'] = all_labels
        return tokenized_inputs

    map_fn = tokenize_and_align_labels
    tokenised_dataset = dataset.map(map_fn, batched=True)
    return tokenised_dataset
