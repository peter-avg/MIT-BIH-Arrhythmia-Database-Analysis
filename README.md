# MIT-BIH Arrhythmia Database Analysis

## 1. Dataset
The dataset used in this study is provided freely by Physionet.org and originates from the laboratories of Beth Israel Hospital in Boston and MIT University. It is named MIT-BIH Arrhythmia Database. This dataset consists of 48 half-hour recordings, each containing measurements from two channels of electrocardiography, collected from 47 patients. 23 patients were randomly selected from a mixed population, with 60% being hospital patients and 40% non-hospital patients. The remaining 25 patients were specifically chosen for clinically significant arrhythmias they exhibited. Each recording was digitized at 360 samples per second, with 11-bit resolution on a scale of 10mV. Two or more cardiologists are responsible for annotating the recordings.

## 2. Experimental Setup
After data acquisition, it was transformed to facilitate our study. Specifically, the data for each patient is stored in 4 files in the format atr, dat, hea, and xws. After reading them using appropriate Python libraries, they were segmented into beats, and annotations were assigned to each beat. The annotations for this experiment were defined as normal and abnormal, as we proceed with binary classification. Padding of 300 points was added to all beats to define features for the learning process. Finally, the data was normalized. For abnormal annotations, data augmentation was used to produce an equal number of normal and abnormal annotations. Gaussian noise for each feature in a beat was the augmentation tool.

Subsequently, the data was split into training, validation, and testing sets:
1. **Training Set:** Used for training machine learning models or neural networks to learn the characteristics and correlations of the data.
2. **Validation Set:** Used to validate the learning of the model during training, to prevent overfitting of the data in the training set, aiming for better generalization of the model.
3. **Testing Set:** Used to test the model after its training is completed.

Training was conducted using 20 epochs with early stopping (termination of model training if it doesn't perform well on the validation set) and multiple parameterizations of three models, as per the literature review.

## 3. Model Implementation
A neural network, known for achieving high scores in this experiment, was implemented using CNN layers.
We chose to implement a 1D Convolutional Neural Network which, after 2 or 4 CNN layers and pooling layers, is connected to a fully connected Feed Forward Neural Network. Here, dropout technique was used for faster results and to avoid overfitting.
Multiple hyperparameterizations were also implemented in this network by changing the number of filters, kernel size, and number of nodes in the fully connected network. Significant results of around 0.92 test accuracy were achieved.


