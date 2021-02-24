# AccentDetector
**Objective:**
  - Project to make an android + web app detect accent of any individual. 
  - Train an individual for a particular accent.
  - Possibly predict the birthplace of that individual.

**Dataset:** https://www.kaggle.com/rtatman/speech-accent-archive
 - Data contains a speech that is recorded in English by people from different countries and different accent.
 - Data contains male and female speakers speaking the same passage.
 - Data contains speakers from 176 countries.

**Idea:** 
 - Use Machine Learning models to train all the classes of accents and predict the output of a given audio file.
    - Get all audio files of a particular accent.
    - Get the timestamps of each word in the speech and when it is occuring in each audio.
    - Train all the words in that accent and do that for all the classes.

**Tools to use:** 
 - ReactJS + React Native/Babble
 - Flask
 - Jupyter Notebook
 - Tensorflow
 - Audio Processing (Frequency Domain Features + Time Domain Features):
    - MFCC(Mel Frequency Capstral Coefficent)
    - Log Mel Spectogram
    - Harmonic Percussive Source Separation
    - Audio Waves.
 
 **Questions?:**
  - How to label the data? Basically we need to know at what time stamp a particular word is being said. 
      - How to get those timestamp of each word in each audio file under each accent?
  - What can we do about the remaining words which are not in the dataset? 
  - Which models should we use to train and predict? 
  - How can we process the audio files first to tailor it to our model? (Data Processing)
  - How to improve accuracy of the model and tailor it to detect more words?
      - Can we add more words to our dataset like for instance: If a user speaks something, store all those words under corresponding accent class.
  - Gather more data on countries with highest # of english speakers.
  
  **Resources relevant to the project:** 
  
   - https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0 : Audio Processing for Machine Learning.
   - https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&ab_channel=ValerioVelardo-TheSoundofAI: Audio Processing for Deep Learning with Python

