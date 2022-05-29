# Semi-Supervised Classifier
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) 	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

A Semi-Supervised Solution that involves periodic human interaction to extract Technical Skill from raw dataset of over 30k datapoints which contains technical skills and a lot of jargon mixed in.

Technical skills are demonstrable and quantifiable skills. They can be tested to prove their capacity an individual possesses.

## Working
1. **Loading and Preprocessing** : We load two files ***(Raw_Skills_Dataset.csv)*** and ***(Example_Technical_Skills.csv)***, theses files contain the dataset to classify and some example technical skills that act as initial labels for data in noisy dataset. Simple Preprocessing is done on text that replaces all puntuation with " " and then spilts based on it taking that very first word only.

2. **Labeling Dataset** : We use example technical skills to label the same skills if present in the dataset and add in some common skills as ***known_tech_skills***.

3. **Tokenization and Model** : We use the processed text to create and ***TextVectorization layer*** and a simple model to train our ***Word Embeddings*** using *binarycrossentropy*.

4. **Cosine Similarity** : Extract the *word embeddings* and find *coine similarity* between all, for all *known_tech_skills* take ***top-k*** similar skill and mark them as *tech skills* and all others as *soft skills*. 

5. **Repeat** : Repeat the procedure *x number of times* for the same model using the ***modified dataset*** with new tech skills on the same model.

## Results
![image](https://user-images.githubusercontent.com/56474719/170863921-16efaa13-36b1-4ae7-bd31-f18148d1d7ec.png)

### PCA on Word Embeddings (3 Iterations Only)
- **Far appart cluster of tech skills (*represented on  right side list*) well seperated from soft skills.**
![image](https://user-images.githubusercontent.com/56474719/170863859-eb7b0659-37c5-4d3c-8a4d-8d4dd1d7800c.png)

- **Cluster of mixed classes with require human validation to improve classification.**
![image](https://user-images.githubusercontent.com/56474719/170863883-50b0d880-4c8c-4d99-9767-2b9c69a02663.png)

- **Cluster well seperated from tech skills cluster representing soft skills.**
![image](https://user-images.githubusercontent.com/56474719/170863904-1bca539f-76d1-4aff-8e85-bba766a18886.png)


> Note: Trained under time constrain can be improved further easily.
> .tsv files can be used [Embedding Projector](http://projector.tensorflow.org/ "Embedding Projector") to visualize embeddings.
