import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self,
                 exp_name,
                 image_size,
                 channel,
                 classes,
                 weights=False):
        self.exp_name = exp_name
        self.image_shape = (image_size, image_size, channel)
        self.classes = classes
        self.ckpt_path = f'checkpoints/{self.exp_name}'
        self.weights = weights

        self.model = self._build_model()
    
    # build model
    def _build_model(self):
        model = tf.keras.applications.MobileNetV2(
            input_shape=self.image_shape, include_top=False
            )
        out = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        outputs = tf.keras.layers.Dense(self.classes, activation='softmax')(out)
        model = tf.keras.Model(inputs=model.input, outputs=outputs)
        if self.weights:
            model.load_weights(self.ckpt_path)

        return model

    def _callback(self):
        callback = [
            tf.keras.callbacks.EarlyStopping(patience=50, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(self.ckpt_path,
                                               verbose=1,
                                               save_best_only=True
                                               ),
            tf.keras.callbacks.TensorBoard(log_dir=f'logs/{self.exp_name}')
                ]
        return callback

    # train model
    def train(self, train_data, val_data, epochs):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy']
                           )
        self.model.fit(train_data, epochs=epochs, callbacks=slef._callback(),
                       validation_data=val_data)
                       
    def test(self, test_data):
        test_loss, test_acc = self.model.evaluate(test_data, verbose=2)
        print('\nTest accuracy: ', test_acc)

    def predict(self, image):
        predictions = self.model.predict(image)
        args, vals = np.argmax(predictions)
        for arg, val in zip(args, vals):
            print(f'class: {arg}, confidence: {val:.4f}')

if __name__ == '__main__':
   import os
   from utils import gpu_select
   gpu_select()
   model = Model('test',  224, 3, 10)
   print(os.getenv('CUDA_VISIBLE_DEVICES'))
