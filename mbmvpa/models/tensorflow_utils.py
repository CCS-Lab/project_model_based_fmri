import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from ..data import loader
from pathlib import Path
import numpy as np

class ExperimenterTF():
    def __init__(self, 
                 chk_path='.',
                 result_path=',',
                 extractor=None,
                 n_sample=10000,
                 n_epoch=100,
                 n_patience=10,
                 n_batch=64,
                 validation_split_ratio=0.2,
                 save_pred=True,
                 use_bipolar_balancing=False,
                 verbose=1):
        
        self.n_sample = n_sample
        self.n_epoch = n_epoch
        self.n_patience = n_patience
        self.n_batch = n_batch
        self.validation_split_ratio = validation_split_ratio
        self.save_pred = save_pred
        self.use_bipolar_balancing = use_bipolar_balancing
        self.chk_path = Path(chk_path)
        self.result_path = Path(result_path)
        self.verbose = verbose
        
    def __call__(self, model, i, X, y):
        # random sampling "n_samples" if the given number of X,y instances is bigger
        # than maximum allowed number for training
        np.random.seed(i)
        tf.random.set_seed(i) # also need to set random seed in tensorflow
        ids = np.arange(X.shape[0])

        if X.shape[0] > self.n_sample:
            np.random.shuffle(ids)
            ids = ids[:self.n_sample]

        # split data to training set and validation set
        train_ids, test_ids = train_test_split(
            ids, test_size=self.validation_split_ratio, random_state=i
        )
        train_steps = len(train_ids) // self.n_batch
        val_steps = len(test_ids) // self.n_batch

        assert train_steps > 0
        assert val_steps > 0

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        # create helper class for generating data
        # support mini-batch training implemented in Keras
        train_generator = loader.DataGenerator(
            X_train, y_train, self.n_batch, shuffle=True,
            use_bipolar_balancing=self.use_bipolar_balancing)
        val_generator = loader.DataGenerator(
            X_test, y_test, self.n_batch, shuffle=False,
            use_bipolar_balancing=self.use_bipolar_balancing)
        # should be implemented in the actual model

        best_model_filepath = str(self.chk_path / \
        f"repeat_{i:03}.ckpt")

        # temporal buffer for intermediate training results (weights) of training.
        mc = ModelCheckpoint(
            best_model_filepath,
            save_best_only=True, save_weights_only=True,
            monitor="val_loss", mode="min")

        # device for early stopping. if val_loss does not decrease within patience, 
        # the training will stop
        es = EarlyStopping(monitor="val_loss", patience=self.n_patience)
        
        model.fit(train_generator, epochs=self.n_epoch,
              verbose=0, callbacks=[mc, es],
              validation_data=val_generator,
              steps_per_epoch=train_steps,
              validation_steps=val_steps)

        # load best model
        model.load_weights(best_model_filepath)
        # validation 
        y_pred = model.predict(X_test)
        len(y_pred)
        error = mean_squared_error(y_pred, y_test)
        if self.save_pred:
            total_pred = model.predict(X)
            usedtrain_map = np.zeros((X.shape[0],1))
            usedtrain_map[train_ids] = 1
            pred_data = np.concatenate([total_pred, usedtrain_map],-1)
            pred_path = self.result_path / f"repeat_{i:03}_pred.npy"
            np.save(pred_path, pred_data)

        if self.verbose > 0:
            print(f"[{i:03}] - val_loss: {error:.04f}")
       
        return model, error