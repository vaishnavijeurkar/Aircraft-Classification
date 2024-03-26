# Deep Learning Aircraft Image Classification with InceptionV3

This project initializes a deep learning model for image classification using the InceptionV3 architecture, a state-of-the-art convolutional neural network (CNN) pre-trained on the ImageNet dataset.

## Model Architecture
- The base model is loaded with pre-trained weights from ImageNet.
- Setting `include_top` to `False` excludes the fully connected layers at the top, allowing for the addition of custom layers.
- The architecture includes global average pooling, two fully connected dense layers with ReLU activation, batch normalization, and dropout regularization.
- The model is compiled with the Adam optimizer and trained to classify images into 100 classes using a softmax activation function.

## Transfer Learning and Fine-Tuning
- `base_model.trainable=True` enables fine-tuning transfer learning.
- Fine-tuning involves unfreezing some or all the layers in the pre-trained model and training them along with the custom layers added for the specific task.

## Hyperparameter Tuning
- Hyperparameter tuning was performed to optimize model performance.
- Best hyperparameters: 1024 units in dense layers, dropout rate of 0.5, and a batch size of 32.
- A learning rate scheduler was employed to determine the optimal learning rate for the model.

## Evaluation Metrics
- After training, the model achieved an accuracy of 83.078% on the test set.
- Precision: 0.838
- Recall: 0.831
- F1 Score: 0.830

## Re-Running the Code
To re-run the code for training a new model:
1. Mount your Google Drive.
2. Add the paths to your test, train, and validation CSV files.
3. Add the path to your image folder.
4. Run each cell to regenerate a new model. 
5. Save the model by running the cell with the code to save model and add the path to the saved_model variable to produce the evaluation metrics.
