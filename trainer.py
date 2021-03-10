import numpy as np
import tensorflow as tf
import kerastuner as kt


class Model(object):
    def __init__(self,
                 exp_name,
                 classes,
                 image_size=224,
                 channel=3,
                 train=True):
        self._exp_name = exp_name
        self._image_shape = (image_size, image_size, channel)
        self._classes = classes
        self._ckpt_path = f'checkpoints/{self._exp_name}'
        self._train = train

        if train:
            self._model = self._build_tuner()
        else:
            self._model = self._build_model()
    
    # build model
    def _build_model(self, hp=None):
        if not self._train:
            model = tf.keras.models.load_model(self._ckpt_path)
            return model

        model = tf.keras.applications.MobileNetV2(
            input_shape=self._image_shape, include_top=False
            )
        out = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        outputs = tf.keras.layers.Dense(self._classes, activation='softmax')(out)
        model = tf.keras.Model(inputs=model.input, outputs=outputs)

        hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        model.compile(optimizer=tf.keras.optimizers.Adam(hp_lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy']
                     )

        return model

    def _build_tuner(self):
        tuner = kt.Hyperband(self._build_model,
                             objective='val_loss',
                             max_epochs=10,
                             factor=3,
                             directory='tuner',
                             project_name=self._exp_name)
        return tuner
        

    def _callback(self):
        callback = [
            tf.keras.callbacks.EarlyStopping(patience=50, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(self._ckpt_path,
                                               verbose=1,
                                               save_best_only=True
                                               ),
            tf.keras.callbacks.TensorBoard(log_dir=f'logs/{self._exp_name}')
                ]
        return callback

    def search(self, train_data, val_data, epochs, train_steps, val_steps):
        self._model.search(train_data.repeat(), epochs=epochs,
                           steps_per_epoch=train_steps, validation_data=val_data.repeat(),
                           validation_steps=val_steps, verbose=1, shuffle=True)
        self._best_hps = self._model.get_best_hyperparameters(num_trials=1)[0]

    # train model
    def train(self, train_data, val_data, epochs, lr, train_steps, val_steps):
        self._model = self._build_model(self._best_hps)
        self._model.fit(train_data.repeat(), epochs=epochs, callbacks=self._callback(),
                       steps_per_epoch=train_steps, validation_data=val_data.repeat(),
                       validation_steps=val_steps, verbose=1, shuffle=True)
                       
    def test(self, test_data):
        test_loss, test_acc = self._model.evaluate(test_data, verbose=1)
        print('\nTest accuracy: ', test_acc)

    def predict(self, image):
        predictions = self._model.predict(image)
        args, vals = np.argmax(predictions)
        for arg, val in zip(args, vals):
            print(f'class: {arg}, confidence: {val:.4f}')

if __name__ == '__main__':
   import os
   from utils import gpu_select
   gpu_select()
   model = Model('test',  224, 3, 10)
   print(os.getenv('CUDA_VISIBLE_DEVICES'))
