# Continual Learning Demo App using a ReplayBuffer

### Overview

The application is an extension from the latest Tensorflow-Lite model personalization demo app.

We expanded the demo app and added a replayBuffer which stores a portion of the samples used in the previous training cycles to enable continua learning capabilities. As a result, we can achieve correct inference in class incremental scenarios as well

## Reference to the Tensorflow-Lite model personalization demo app:

https://github.com/tensorflow/examples/tree/master/lite/examples/model_personalization

## Structure of the model:

Base Model: tf.keras.applications.MobileNetV2, a pre-trained MobileNetV2 model, which is a good fit for image recognition tasks.

Head Model: a single dense layer followed by softmax activation.

Optimizer: tf.keras.optimizers.Adam with the default setting learning_rate=0.001.


