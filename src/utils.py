'''Utility classes and functions.'''

from transformers import ProgressCallback
from tqdm.auto import tqdm
from collections.abc import Sized


class NeverLeaveProgressCallback(ProgressCallback):
    '''Progress callback which never leaves the progress bar'''

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            desc = 'Finetuning model'
            self.training_bar = tqdm(total=None, leave=False, desc=desc)
        self.current_step = 0

    def on_prediction_step(self, args, state, control, eval_dataloader=None,
                           **kwargs):
        correct_dtype = isinstance(eval_dataloader.dataset, Sized)
        if state.is_local_process_zero and correct_dtype:
            if self.prediction_bar is None:
                desc = 'Evaluating model'
                self.prediction_bar = tqdm(total=len(eval_dataloader),
                                           leave=False,
                                           desc=desc)
            self.prediction_bar.update(1)
