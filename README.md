# Accent-Detector/Accent-Classifier
## **Objective:**
  - Project to detect accent of an individual in spoken english.
  - Use Machine Learning models to train 4 classes of accents and predict the output of a given audio file.

## **Table of Content:**
1. [Motivation](#motivation)
2. [Tools Used](#tools-used)
3. [Dataset](#dataset)
4. [Data Analysis](#data-analysis) and [Pre-Processing.](#pre-preprocessing)
5. [Audio Processing.](#audio-processing)
6. [Data Preparation](#data-preparation)
7. [Training Machine Learning Models.](#training-ml-models)
8. [Model Performance Comparision](#model-performance-comparision)
9. [Training Neural Network.](#training-neural-network)
10. [Techniques to handle imbalance in the dataset.](#Techniques-used-to-handle-imbalanced-data)
11. [Future enhancements.](#future-enhancements)

## **Motivation:**
The motivation behind developing this project is, since I am living as an Indian student in The US, I wanted to improve my american accent to learn more about the culture and blend in. But there was no tool online that could tell me my accent and how well it is. That is when I decided I want to create a tool that could help people like me identify and suggest improvements on their accent.

## **Tools used:** 
 - Jupyter Notebook/Google Colab
 - _Librosa_ for Audio Processing (Frequency Domain Features + Time Domain Features):
    - MFCC(Mel Frequency Capstral Coefficent)
 - _Numpy, Pandas_ for Data Processing and Analysis.
 - _scikit-learn_ for Machine Learning models.
 - _Tensorflow_ and keras for Deep learning models.

## **Dataset:** 
[This Dataset](https://www.kaggle.com/rtatman/speech-accent-archive) on kaggle.
 - Since the dataset has few samples, I choose to classify only on 4 accents: (Will improve when I get more samples.)
    - Indian
    - American
    - British
    - Chinese

## **Data Analysis:**
The dataset contains: 
 - 2172 samples of speakers in total(audio in mp3 format).
 - Samples from 177  different countries.
 - Samples of 214 different languages.
 - Each user is speaking the *passage*: 

"Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."

## **Pre processing:**
 - Merged samples from languages in and around India to create "_Indian accent_"(110).
 - Grouped samples to create "_American"(373), "_British"(65) and "_Chinese_"(88) accents classes.
 - Removed other samples of audios and from dataframe.
 - Removed unnecessary columns. (age, birthplace, speakerid, file_missing)
 

## **Audio Processing:**
 - Converted all the mp3 files to a "_wav_" format. (uncompressed version).
 - Trim/pad all the audio files to a standard length of 30 seconds.
 - Extracted 13 **MFCCs:** Mel Frequency Cepstral Coefficents from each audio file.
***MFCC*** in a sentence, is a "*representation*" of the vocal tract that produces the sound. Think of it like an x-ray of your mouth.
 - Extracted MFCCs of an audio file will be of shape (1, 2584, 13)
 - Add [Gaussian noise](https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8) to each sample.
 - Oversample minority classes to handle imbalance in the dataset. 
 - Dump the data in *json* format to access it later.

## **Data preparation:**
 - Load mfccs and targets from json.
 - Convert it in a 2D format of (1445, 33592) -> (#samples, (2584*13))
 - Create Train and Test data using _train_test_split_.
 - Use [SMOTE](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) (Synthetic Minority Over-Sampling Technique) to balance samples.

## **Training ML models:**
1. **Support Vector Machine(SVM):**
 - Using a "*rbf*" kernel SVC.
 - Metrics: Accuracy of **93%** with an amazing f1-score.

      <img src="https://user-images.githubusercontent.com/13129747/116898902-f37f8780-ac04-11eb-86a5-95c937679a41.png" width=500 height=300 />

2. **Random Forest Classifier(RFC):**
 - Using a RFC of *max_depth=16* and *n_estimators=250*.
 - Metrics: Accuracy of **92%**, slightly lower than SVC, but pretty decent and almost similar f1-scores.

      <img src="https://user-images.githubusercontent.com/13129747/116899202-448f7b80-ac05-11eb-8beb-54b3396fb414.png" width=500 height=300 />

3. **K-Nearest-Neighbors(KNN):**
 - Using a KNN of *n_neighbors=3*.
 - Metrics: Accuracy of **76.57%** with imbalanced f1-scores. Performs poor compared to "*svc*" and "*rfc*".

      <img src="https://user-images.githubusercontent.com/13129747/116899537-a819a900-ac05-11eb-8f83-ffad847bde22.png" width=500 height=300 />

4. **Logistic Regression(LR):**
 - Metrics: Accuracy of **87%** with an almost balanced f1-score.

      <img src="https://user-images.githubusercontent.com/13129747/116899682-da2b0b00-ac05-11eb-9862-b4ef45335a77.png" width=500 height=300 />

## **Model Performance Comparision:**
This table compares different models and its metrics.
| Model | Accuracy | American(f1) | British(f1) | Chinese(f1) | Indian(f1) | 
| --- | --- | --- | --- | --- | --- |
| SVM | 93% | 85% | 98% | 99% | 89% |
| RFC | 92% | 85% | 100% | 96% | 87% |
| KNN | 76% | 55% | 83% | 89% | 75% |
| LR | 87% | 79% | 96% | 94% | 82% |

**Note:** We can see that SVM performs best here.

## **Training neural network:**
1. **Recurrent Neural Network(RNN):**
 - Using Keras's Sequential model with 1 input layer, 3 hidden layers(with dropouts)(*activation=relu*) and 1 output layer(*Softmax activation*).
 - Using *Adam* optimizer to compile and run it for 50 epochs.
 - Metrics: Accuracy of **59%**. (Disappointing results)
2. **Convolutional Neural Network(CNN):**
 - Using Keras's Sequential model with 1 input layer, 3 convolution layers(with BatchNormalization), 1 Dense Layer(with Dropout) and 1 output layer(*softmax activation*).
 - Using *Adam* optimizer to compile and run for 30 epochs.
 - Metrics: Accuracy of **68%** on test data. (Performed really poorly on real data sample.)
 
## **Techniques used to handle imbalanced data:**
Before, the data was heavily imbalanced, and training an SVC just gave an accuracy of 59% with extremely poor f1-scores for each accent.
To handle the imbalance in our dataset, I used techniques like: 
1. **SMOTE:** (Synthetic Minority Over-Sampling Technique)
 - addresses imbalanced datasets by oversampling the minority class. 
 - The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. 
 - Instead, new examples can be synthesized from the existing examples.

  **Results:** SMOTE improved accuracy from 59% to around 65%. (Better, but not acceptable.)

2. **Undersampling:**
 - reduces the samples of American(373) to 110 and then trained SVC.

  **Results:** Extremely poor accuracy of 41% with poor f1-scores.

3. **Oversampling:**:
 - Duplicate samples of minority classes to match close to american samples.
 - Get a Duplication ratio: i.e. 373/373 = 1, 373/110 = 3, 373/88 = 4, 373/65=5
 - Dupliate each sample by this duplication ratio.

  **Results:** Give a really good training accuracy of 90%, but performed subpar on test data. 

4. **Oversampling with Noise:**
 - Duplicate samples of minority class in a similar way as above, but
 - Extract a random gaussian noise from the signal, and add it to the audio file and then oversample.

  **Results:** Performed best, with an overall accuracy of 93% with nicely balanced f1 scores.

**Resources relevant to the project:** 
  
   - https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0 : Audio Processing for Machine Learning.
   - https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&ab_channel=ValerioVelardo-TheSoundofAI: Audio Processing for Deep Learning with Python

