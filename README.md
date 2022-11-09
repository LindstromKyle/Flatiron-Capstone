# Using Object Detection Algorithms to Identify and Translate American Sign Language


Training computer vision networks to detect simple hand signals and complex motions 


![](./Images/image1.png)
![](./Images/image2.png)
![](./Images/image3.png)
![](./Images/image4.png)
![](./Images/image5.png)
![](./Images/Action.gif)
![](./Images/Object_Detection.gif)

## Old work to be replaced below:

## **Introduction**


Pneumonia is the single deadliest disease for children across the planet. This is especially true in developing nations in Southeast Asia and Sub-Saharan Africa, where many countries are dealing with a shortage of available doctors. Our client, the American Red Cross, is interested in easing the burden on medical professionals in these environments by developing new tools to identify pneumonia and flag children who are most at risk. They have asked us to train a model that is able classify the presence of pneumonia in a dataset of pediatric x-rays. These model predictions can then be verified by medical specialists and the children can be treated accordingly. 


## **Exploratory Data Analysis**


This dataset was published in 2018 and consists of around 6,000 chest x-rays taken of children ages 1 through 5. These were taken at the Guangzhou Women and Children's Medical Center and collected, cleaned, and published by researchers at UC San Diego. The labels for the images were verified separately by 3 medical experts. Examples from this dataset are shown below in *Figure 1*:


![](./images/image2.png)


> *Figure 1*


## **Data Augmentation**


Because of the limited size of this dataset, and the fact that machine learning models thrive on more data, we decided to use augmentation to increase the number of training examples available to our model. Images we flipped horizontallly and rotated by a random angle of ± 20 degrees. These augmentations are shown in *Figure 2*:


![](./images/image3.png)


> *Figure 2*


## **Performance Metrics**


In order to judge the performance of our model, we decided on two metrics:


### **Recall**

Our first performance metric is recall- a measure of our model's true positive rate. We would like our recall to be high, to maximize the probability that if someone truly has pneumonia, the model predicts this correctly and flags them for a medical follow-up.

### **Accuracy**

Our second performance metric is accuracy- a measure of how many of our model's predictions are correct in total. Simply predicting every child has pneumonia would result in a 100% recall score, which is obviously not helpful in this context. We want to maximize accuracy so that we avoid a large false positive rate that does not reduce the strain on medical staff who are already spread too thin. 


## **Model Selection**

We fit many different models to our data in order to find the most effective solution. *Figure 3* below shows a table of the performance metrics of various models. 

![](./images/image4.png)

> *Figure 3*

All models performed exceedingly well in regards to recall, but where the convolutional neural network shines is in its overall accuracy. This is important in reducing those false positives, and thus it is the model we chose to pursue for this problem.

## **Final Model**

Our final model architecture consisted of an Xception model pretrained on the ImageNet dataset, with custom pooling and output layers specific to our binary classification problem. *Figure 4* shows our chosen metrics' evolution by training epoch and *Figure 5* shows the confusion matrix for this model's performance on our test data set:

![](./images/image5.png)

> *Figure 4*

![](./images/image6.png)

> *Figure 5*

### **Final Recall Score**

Out of 390 children in the test set with pneumonia, our model was able to recognize 389 of them, for a recall score of over 99%. This left only a single false negative, meaning our model is able to flag almost every child who truly has pneumonia for a follow up with a medical professional. 

### **Final Accuracy**

Out of 594 children in the test set total, our model correctly classified 545. This high accuracy is important for reducing our false positives and ensuring that valuable medical resources are budgeted correctly for those in need. 


## **Conclusions & Recommendations**

We find that our convolutional neural network provides the optimal predictive power for this business problem. We believbe this model will indeed help humanitarian aid workers use their valuable medical resources more efficiently. Doctors can be deployed where they are most needed, and the children in need of aid can be identified more quickly. We recommend this system not as a replacement for a medical professional's opinion, but as a supplement to these resources and a technique to help filter through what may currently be simply too much data. 

[Full Jupyter Notebook](https://github.com/hall-nicholas/flatiron-ds-project-4/blob/main/code/Draft_final.ipynb)  

[Non Technical Presentation](https://github.com/hall-nicholas/flatiron-ds-project-4/blob/main/Non%20Technical%20Presentation.pdf)  

[Original Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)

