<body>
  <h1>Handwriting Recognition</h1>

  <h2>Overview</h2>
  <p>This project aims to build a handwriting recognition model using deep learning techniques. The model is trained on a dataset of handwritten names, allowing it to recognize and classify handwritten characters effectively.</p>

  <h2>Dataset</h2>
  <p>The dataset used in this project is the <a href="https://www.kaggle.com/datasets/landlord/handwriting-recognition">Handwriting Recognition Dataset</a> available on Kaggle. It consists of over 400,000 images of handwritten names collected through charity projects.</p>

  <h3>License</h3>
  <p>The dataset is licensed under <a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0-1.0</a>.</p>

  <h2>Requirements</h2>
  <p>To run this project, you need the following libraries:</p>
  <ul>
      <li>TensorFlow</li>
      <li>Keras</li>
      <li>NumPy</li>
      <li>Matplotlib (optional, for visualization)</li>
  </ul>
  <p>You can install the required libraries using pip:</p>
  <pre><code>!pip install tensorflow keras numpy matplotlib</code></pre>

  <h2>Installation</h2>
  <ol>
      <li><strong>Install Kaggle API:</strong> Ensure that the Kaggle API is installed in your environment.
          <pre><code>!pip install kaggle</code></pre>
      </li>
      <li><strong>Download the Dataset:</strong> Use the Kaggle API to download the dataset.
          <pre><code>!kaggle datasets download landlord/handwriting-recognition</code></pre>
      </li>
      <li><strong>Extract the Dataset:</strong> Extract the contents of the downloaded zip file.
          <pre><code>from zipfile import ZipFile

file_name = "handwriting-recognition.zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('done')</code></pre>
      </li>
  </ol>

  <h2>Usage</h2>
  <ol>
      <li><strong>Set Up Directories:</strong> Define the paths for training, validation, and test datasets.
          <pre><code>test_dir = "/content/test_v2"
train_dir = "/content/train_v2"
validation_dir = "/content/validation_v2"</code></pre>
      </li>
      <li><strong>Data Augmentation:</strong> Create data generators for training and validation datasets.
          <pre><code>from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size=(224, 224),
  batch_size=32,
  class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
  validation_dir,
  target_size=(224, 224),
  batch_size=32,
  class_mode='categorical'
)</code></pre>
      </li>
      <li><strong>Build the Model:</strong> Define the architecture of the convolutional neural network (CNN).
          <pre><code>from tensorflow.keras import layers, models

model = models.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
  layers.MaxPooling2D(2, 2),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D(2, 2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(64, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='softmax')
])</code></pre>
      </li>
      <li><strong>Compile the Model:</strong> Compile the model with an optimizer and loss function.
          <pre><code>model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
  loss='categorical_crossentropy',
  metrics=['accuracy']
)</code></pre>
      </li>
      <li><strong>Train the Model:</strong> Fit the model to the training data.
          <pre><code>class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') > 0.93:
          print("\nReached 93% accuracy so cancelling training!")
          self.model.stop_training = True

history = model.fit(
  train_generator,
  epochs=5,
  validation_data=validation_generator,
  callbacks=[myCallback()]
)</code></pre>
      </li>
      <li><strong>Evaluate the Model:</strong> Evaluate the model's performance on the validation set.
          <pre><code>test_loss, test_acc = model.evaluate(validation_generator)
print('Test accuracy:', test_acc)</code></pre>
      </li>
      <li><strong>Make Predictions:</strong> Use the model to make predictions on the validation set.
          <pre><code>import numpy as np

prediction = model.predict(validation_generator)
y_pred = np.argmax(prediction, axis=1)
print(y_pred)
print(prediction)</code></pre>
      </li>
  </ol>

  <h2>Conclusion</h2>
  <p>This project demonstrates the process of building a handwriting recognition model using deep learning techniques. The model can be further improved by experimenting with different architectures, hyperparameters, and data augmentation techniques.</p>

  <h2>Acknowledgments</h2>
  <ul>
      <li><a href="https://www.kaggle.com/">Kaggle</a> for providing the dataset.</li>
      <li><a href="https://www.tensorflow.org/">TensorFlow</a> and <a href="https://keras.io/">Keras</a> for the deep learning framework.</li>
  </ul>

  <h2>License</h2>
  <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
</body>
