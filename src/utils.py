import json
from tabnanny import verbose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from datetime import datetime
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback

# Import custom code in development for this project
from src.architecture import build_model

class TrainingExperiment:
    """Represents one set of training runs from a new model to a trained moded
    
    Model will be saved after each run with a name based on the training experiment name
    """
    
    def __init__(self, name, train_ds, val_ds, p2drive, path2dir=None):

        if path2dir is None:
            self.p2saved = p2drive / 'saved/cnn_virus'
        else:
            self.p2saved = path2dir
        if not self.p2saved.is_dir():
            os.makedirs(self.p2saved, exist_ok=True)

        self.name = name
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.exp_start = datetime.now()
        self.timestamp = f"{self.exp_start.strftime('%y%m%d-%H%M')}-UTC"
        
        # check if model exist with this name
        saved_models = [f for f in self.p2saved.glob(f"{self.name}*") if f.is_dir()]
        if saved_models:
            self._get_last_saved_model(saved_models)
            
        else:
            self._build_new_model()

        print(f"Experiment {self.name} ready.\nModels will be saved in {self.p2saved.absolute()}")


    def run_epochs(self, lr=1e-4, epochs=100, patience=5, save_model=True):
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.run_count = self.run_count + 1

        self.model.compile(
            optimizer=Adam(learning_rate=self.lr, name='adam'),
            loss={'output1':CategoricalCrossentropy(),'output2':CategoricalCrossentropy()},
	        metrics=['accuracy'],
        )

        es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        if epochs>10:
            pb = TqdmCallback(verbose=1)
            cbs = [es, pb]
            verbose = 0
        else:
            cbs = [es]
            verbose = 1
            
        self.result = self.model.fit(
                self.train_ds,
                epochs=epochs,
                validation_data = self.val_ds,
                verbose=verbose,
                callbacks=cbs,
        )

        self._update_description_file()
        if save_model:
            self.save_model()

    def _build_new_model(self):
        """Build new model for the experiment"""
        self.run_count = 0
        self.result = None

        self.model = build_model()
        print("New model built")

        self._create_new_description_file()

    def _get_last_saved_model(self, saved_models):
        """Retrieve the last model saved under this experiment"""
        runs = np.array([int(f.stem[-3:]) for f in saved_models])
        last_run_idx = np.argmax(runs)
        self.run_count = int(runs[last_run_idx])
        self.p2last_model = saved_models[last_run_idx]
        self.model = load_model(self.p2last_model)
        self.name = self.p2last_model.stem.split('--')[0]
        descr_files =[f for f in self.p2saved.glob(f"{self.name}*-description.txt") if f.is_file()]
        if descr_files:
            self.descr_path =descr_files[0]
            txts = f'Continue experiment with saved model ; loaded from {self.p2last_model}'.split(' ; ')
            self._append_to_description_file([f"{'-'*80}"])
            self._append_to_description_file(txts)
        else:
            self.__create_new_description_file()

    def _create_new_description_file(self):
        self.descr_path = self.p2saved / f"{self.name}-{self.timestamp}--description.txt"
        description_header = f"Description of experiment: {self.name}\n"
        with open(self.descr_path, 'w') as fp:
            fp.write(description_header)
            print(f"Experiment Description file created in {fp.name}")
    
    def _append_to_description_file(self, list_of_strings):
        with open(self.descr_path, 'a') as fp:
            fp.write('\n'.join(list_of_strings))
            fp.write('\n')
            print(f"Experiment Description updated in {fp.name}")

    def _update_description_file(self):
        """End of run info saved into the description file"""
        losses = pd.DataFrame(self.result.history)
        real_epochs = losses.shape[0]
        lot = [f"{'-'*80}"]
        lot.append(f"{self.run_count:2d}.")
        lot.append(f"Scheduled for {self.epochs} epochs with lr: {self.lr:.2e} with early stopping")
        lot.append(f"Ran for {real_epochs} epochs using patience {self.patience}.")
        lot.append(f"Last epoch losses:")
        lot.append('    '.join([f"{c:<10s}" for c in losses.columns]))
        lot.append('    '.join([f"{l:1.4e}" for l in losses.values[-1:, :].tolist()[0]]))
        self._append_to_description_file(lot)

    def add_comment_to_description_file(self, txt):
        self._append_to_description_file([f"Added comment during run {self.run_count}", txt])

    def plot_losses(self, saved_losses=None):
        if saved_losses is None:
            losses = pd.DataFrame(self.result.history)
        else:
            losses = pd.DataFrame(saved_losses)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
        losses[[c for c in losses.columns if 'loss' in c]].plot(ax=ax1)
        ax1.legend(loc='upper right')
        ax1.set_title('Losses')
        losses[[c for c in losses.columns if 'loss' not in c]].plot(ax=ax2)
        ax2.legend(loc='upper right')
        ax2.set_title('Accuracy')
        plt.show()

    def save_model(self):
        model_path = self.p2saved / f"{self.name}--{self.timestamp}-model-{self.run_count:03d}"
        loss_path =  self.p2saved / f"{self.name}--{self.timestamp}-losses-{self.run_count:03d}.json"
        
        self.model.save(model_path, save_format='tf')
        with open(loss_path, 'w') as fp:
            json.dump(self.result.history, fp, indent=4)

        lot = [f"Saved model and losses in {self.p2saved.absolute()} as ",
                f" - {model_path.name}",
                f" - {loss_path.name}"]
        self._append_to_description_file(lot)
        print('\n'.join(lot))        