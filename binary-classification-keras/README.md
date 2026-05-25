## Binary-Classification-with-the-Keras

### 1. Description of the Dataset

This is a dataset that describes sonar chirp returns bouncing off different services. The 60 input variables are the strength of the returns at different angles. It is a binary classification problem that requires a model to differentiate rocks from metal cylinders.

It is a well-understood dataset. All the variables are continuous and generally in the range of 0 to 1. The output variable is a string “M” for mine and “R” for rock, which will need to be converted to integers 1 and 0.

### 2. Neural Network Model Performance

- We will use scikit-learn to evaluate the model using stratified k-fold cross validation. This is a resampling technique that will provide an estimate of the performance of the model. It does this by splitting the data into k-parts and training the model on all parts except one, which is held out as a test set to evaluate the performance of the model. This process is repeated k-times, and the average score across all constructed models is used as a robust estimate of performance. It is stratified, meaning that it will look at the output values and attempt to balance the number of instances that belong to each class in the k-splits of the data.
- To use Keras models with scikit-learn, we must use the KerasClassifier wrapper from the SciKeras module. This class takes a function that creates and returns our neural network model. It also takes arguments that it will pass along to the call to fit(), such as the number of epochs and the batch size.
- The weights are initialized using a small Gaussian random number. The Rectifier activation function is used. The output layer contains a single neuron in order to make predictions. It uses the sigmoid activation function in order to produce a probability output in the range of 0 to 1 that can easily and automatically be converted to crisp class values.

### 3. Re-Run the Baseline Model with Data Preparation

- Standardization is an effective data preparation scheme for tabular data when building neural network models. This is where the data is rescaled such that the mean value for each attribute is 0, and the standard deviation is 1. This preserves Gaussian and Gaussian-like distributions while normalizing the central tendencies for each attribute

### 4. Tuning Layers and Number of Neurons in the Model

- There are many things to tune on a neural network, such as weight initialization, activation functions, optimization procedure, and so on.
- One aspect that may have an outsized effect is the structure of the network itself, called the network topology. In this section, we will look at two experiments on the structure of the network: making it smaller and making it larger.
#### Smaller Network
- The data describes the same signal from different angles. Perhaps some of those angles are more relevant than others. So you can force a type of feature extraction by the network by restricting the representational space in the first hidden layer.
#### Larger Network
- A neural network topology with more layers offers more opportunities for the network to extract key features and recombine them in useful nonlinear ways.
