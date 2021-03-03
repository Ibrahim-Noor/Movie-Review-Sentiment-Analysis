#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import os
from string import punctuation
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


# f = open(os.path.join("Dataset","negative_words.txt"), "r")
dataNegative = np.loadtxt(os.path.join("Dataset","negative_words.txt"), delimiter='\n', dtype=str)
dataPositive = np.loadtxt(os.path.join("Dataset","positive_words.txt"), delimiter='\n', dtype=str)


# In[3]:


stopWords = np.loadtxt(os.path.join("Dataset","stop_words.txt"), delimiter='\n', dtype=str)


# In[4]:


#################################Data Cleaning######################################
#  removes stopwords, single characters, integers, punctuation and lower case all words  
def preProcessing(pathToFile):
    tempList = []
    file = open(pathToFile, "r", encoding="utf8")
    words = file.readline().split()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if len(word) <= 1:
            continue
        word = word.replace("<br />", "")
        word  = "".join([char for index, char in enumerate(word) if char not in punctuation.replace("+","") or (char == "-")])
        word = re.sub(r'''^\d+.*\d+$''', "", word)
        if not word:
            continue
        tempList.append(word)
    return tempList


# In[5]:


###########################Clean Training Data Positive Reviews##############################
trainingDataPositiveCleanWords = []
for file in os.listdir(os.path.join("Dataset", 'train', "pos")):
    cleanWordList = preProcessing(os.path.join("Dataset", 'train', "pos", file))
    trainingDataPositiveCleanWords.append(cleanWordList)


# In[6]:


###########################Clean Training Data Negative Reviews##############################
trainingDataNegativeCleanWords = []
for file in os.listdir(os.path.join("Dataset", 'train', "neg")):
    cleanWordList = preProcessing(os.path.join("Dataset", 'train', "neg", file))
    trainingDataNegativeCleanWords.append(cleanWordList)


# In[7]:


###########################Clean Test Data Positive Reviews##############################
testDataPositiveCleanWords = []
for file in os.listdir(os.path.join("Dataset", 'test', "pos")):
    cleanWordList = preProcessing(os.path.join("Dataset", 'test', "pos", file))
    testDataPositiveCleanWords.append(cleanWordList)


# In[8]:


###########################Clean Test Data Negative Reviews##############################
testDataNegativeCleanWords = []
for file in os.listdir(os.path.join("Dataset", 'test', "neg")):
    cleanWordList = preProcessing(os.path.join("Dataset", 'test', "neg", file))
    testDataNegativeCleanWords.append(cleanWordList)


# In[9]:


####################Feature Extraction#############################
def createFeatureVectors(sentiment, listOfReviews):
    featureVectors = np.ones((4, len(listOfReviews)))
    for index, review in enumerate(listOfReviews):
        featureVector = np.ones(4)
        for word in review:
            if word in dataPositive:
                featureVector[1] += 1
            if word in dataNegative:
                featureVector[2] += 1
        featureVectors[:, index] = featureVector
        if sentiment == "negative":
            featureVectors[3] = np.zeros(1)
    return featureVectors


# In[10]:


#############################Feature Vector For Training Positive Reviews##########################
trainingPositiveReviewsFeatureVectors = createFeatureVectors("positive", trainingDataPositiveCleanWords)


# In[26]:


trainingTrainPositiveReviewsFeatureVectors = trainingPositiveReviewsFeatureVectors[:, :10000]
trainingTestPositiveReviewsFeatureVectors = trainingPositiveReviewsFeatureVectors[:, 10000:]


# In[27]:


print(trainingTrainPositiveReviewsFeatureVectors.shape)
print(trainingTestPositiveReviewsFeatureVectors.shape)


# In[11]:


#############################Feature Vector For Training Negative Reviews##########################
trainingNegativeReviewsFeatureVectors = createFeatureVectors("negative", trainingDataNegativeCleanWords)


# In[28]:


trainingTrainNegativeReviewsFeatureVectors = trainingNegativeReviewsFeatureVectors[:, :10000]
trainingTestNegativeReviewsFeatureVectors = trainingNegativeReviewsFeatureVectors[:, 10000:]


# In[29]:


trainingTrainReviewsFeatureVectors = np.concatenate((trainingTrainPositiveReviewsFeatureVectors, trainingTrainNegativeReviewsFeatureVectors), axis=1)
trainingTestReviewsFeatureVectors = np.concatenate((trainingTestPositiveReviewsFeatureVectors, trainingTestNegativeReviewsFeatureVectors), axis=1)
print(trainingTrainReviewsFeatureVectors.shape)
print(trainingTestReviewsFeatureVectors.shape)


# In[12]:


#############################Feature Vector For All Training Reviews#############################
trainingReviewsFeatureVectors = np.concatenate((trainingPositiveReviewsFeatureVectors, trainingNegativeReviewsFeatureVectors), axis=1)


# In[13]:


###########################PART 1#####################################


# In[14]:


def sigmoid(z):
    result = 1/(1+math.exp(-z))
    return result


# In[15]:



def getProbabilities(thetas, X):
    probabilities = []
    zs = np.matmul(thetas, X)
#     print(zs)
    for z in zs:
        probabilities.append(sigmoid(z))
    return np.array(probabilities)


# In[46]:


###########################Cost Function###############################
def crossEntropyLoss(thetas, X):
    m = X.shape[1]
    probabilities = getProbabilities(thetas, X[0:3])
#     print(probabilities)
    logPositiveProbabilities = np.log2(probabilities)
    logNegativeProbabilities = np.log2(1-probabilities)
    sumValues = sum((X[3]*logPositiveProbabilities) + (1-X[3])*logNegativeProbabilities)
    loss = -(sumValues)/m
    return loss, probabilities


# In[93]:


#########################Gradient Descent#############################
def batchGradientDescent(X, alpha, n_epoch, X_V = None):
    m = X.shape[1]
    costsTrain = []
    costsValidate = []
    thetas = np.zeros((X.shape[0] - 1))
    for epoch in range(n_epoch):
        lossTrain, probabilities = crossEntropyLoss(thetas, X)
        costsTrain.append(lossTrain)
        for i in range(thetas.shape[0]):
            thetas[i] = thetas[i] - ((alpha/m)*sum((probabilities - X[3]) * X[i]))
        if X_V is not None:
            lossValidate, probs = crossEntropyLoss(thetas, X_V)
            costsValidate.append(lossValidate)
    lossTrain, probabilities = crossEntropyLoss(thetas, X)
    costsTrain.append(lossTrain)
    if X_V is not None:
        return costsTrain, costsValidate
    else:
        return thetas, costsTrain


# In[94]:


###############################Finding the optimal value of alpha##################################
costsTrainingForAllAlphas = []
costsValidationForAllAlphas = []
alphas = [0.030,0.025,0.020,0.015,0.01, 0.009, 0.008, 0.007, 0.006, 0.005]

# alpha = 0.01
n_epoch = 800
for alpha in alphas:
    costsTraining, costsValidating = batchGradientDescent(trainingTrainReviewsFeatureVectors, alpha, n_epoch, trainingTestReviewsFeatureVectors)
    costsTrainingForAllAlphas.append(costsTraining)
    costsValidationForAllAlphas.append(costsValidating)


# In[110]:


for i, (trainingCost, validationCost) in enumerate(zip(costsTrainingForAllAlphas,costsValidationForAllAlphas)):
    print("training cost: ",trainingCost[-1])
    print("gap between last training and validation cost: ", validationCost[-1] - trainingCost[-1])
    print("gap between last two training costs: ", trainingCost[-2] - trainingCost[-1])
    print("gap between last two validation cost: " ,validationCost[-2] - validationCost[-1])
    plt.plot(range(0, n_epoch), trainingCost[1:n_epoch+1], label="training data cost")
    plt.plot(range(0, n_epoch), validationCost, label="validation data cost")
    plt.xlabel("number of iterations (epoch)")
    plt.ylabel("cost")
    plt.title("iterations over cost graph " +str(alphas[i]))
    plt.legend()
    plt.show()


# In[96]:


## optimal learning rate in 0.03 as it results least cost, most stable training cost and least difference 
## between training and validation cost


# In[98]:


finalAlpha = 0.03
final_n_epoch = 800
finalThetas, finalCosts = batchGradientDescent(trainingReviewsFeatureVectors, finalAlpha, final_n_epoch)


# In[99]:


#############################Feature Vector For Test Positive Reviews##########################
testPositiveReviewsFeatureVectors = createFeatureVectors("positive", testDataPositiveCleanWords)


# In[100]:


#############################Feature Vector For Test Negative Reviews##########################
testNegativeReviewsFeatureVectors = createFeatureVectors("negative", testDataNegativeCleanWords)


# In[101]:


#############################Feature Vector For All Test Reviews##########################
testReviewsFeatureVectors = np.concatenate((testPositiveReviewsFeatureVectors, testNegativeReviewsFeatureVectors), axis=1)


# In[102]:


##########################Rounding Off Probabilities To Get Predictions#############################
def predictionsOnTestData(calculatedThetas, X):
    probabilities = getProbabilities(calculatedThetas, X[0:3])
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    return np.array(predictions)


# In[103]:


predictions = predictionsOnTestData(finalThetas, testReviewsFeatureVectors)


# In[104]:


def classificationMetrics(testDataPredictions, labels):
    total = 0
    scores = np.zeros(2)
    confusionMatrix = np.zeros((2,2))
    for predictedValue, goldLabel in zip(testDataPredictions, labels):
        if predictedValue == goldLabel:
            scores[predictedValue] += 1
#         print(predictedValue)
        confusionMatrix[predictedValue][int(goldLabel)] += 1
        total += 1
    return total, scores, confusionMatrix


# In[105]:


def accuracy(score, total):
    return score/total


# In[106]:


totalData, scoreAcheived, C_F = classificationMetrics(predictions, testReviewsFeatureVectors[3])


# In[107]:


print("CONFUSION MATRIX: ")
df3 = pd.DataFrame(C_F)
df3.columns = ['Gold negative', 'Gold positive',]
df3.index = ['Predicted negative', 'Predicted positive']
display(df3)


# In[108]:


print("accuracy: ", accuracy(sum(scoreAcheived), totalData))


# In[109]:


plt.figure(num=1, figsize=(20, 10))
plt.plot(range(0, n_epoch), finalCosts[1:n_epoch+1], "k-")
plt.xlabel("number of iterations (epoch)")
plt.ylabel("cost")
plt.title("iterations over cost graph")
plt.show()


# In[30]:


###########################PART 2########################################


# In[31]:


############################Logistic Regression Using Scikit#########################
def logisticRegressionUsingScikit(trainingData, testData):
    model = LR(fit_intercept = False).fit(trainingData[0:3].transpose(), trainingData[3])

    predicted= model.predict(testData[0:3].transpose())


    testGoldLabels = testData[3]


    accuracy = accuracy_score(testGoldLabels, predicted)
    confusionMatrix = confusion_matrix(testGoldLabels, predicted)
        
    return accuracy, confusionMatrix


# In[32]:


def displayConfusionMatrix(confusionMatrix):
#     plt.figure(num=2, figsize=(10, 7))
    dfCM = pd.DataFrame(confusionMatrix, index=["Gold Negative", "Gold Positive"], columns=["Predicted Negative", "Predicted Positive"])
    display(dfCM)
#     sn.set(font_scale=1.3) # for label size
#     ax = sn.heatmap(dfCM, annot=True, annot_kws={"size": 25}, fmt="d", cbar=False)
#     ax.tick_params(length=0)
#     ax.set_title("Confusion Matrix")


# In[33]:


skLearnAccuracy, skLearnC_F = logisticRegressionUsingScikit(trainingReviewsFeatureVectors, testReviewsFeatureVectors)


# In[34]:


print("Accuracy using SKLearn: ", skLearnAccuracy)


# In[35]:


displayConfusionMatrix(skLearnC_F)


# In[ ]:




