# Sentiment-Analysis-in-Persian-Language_LSTM
Sentiment Analysis with LSTM in Persian

## First Phase
#### Data Aquisition
In this repository I have used LSTM for prediction of whether people would like or dislike a product based on the previous comments in the Digikala site. I have scraped the data from Digikala and have labeled them based on the stars people who had bought the products gave to them. I have also used another label from the same website which indicates people suggest others to buy that products or not. because many of the comments are noisy and do not provide a clean data for us and it is not such a reliable source by adding the second label to the data we can ensure a higher accuracy of our training data.
For more clarifying the labels:
1 indicates suggesting others to buy and 2 means otherwise, 3 illusterates a neutral opinion about the product and 4 means the person has rate the product, but not suggest whether to buy or not.
and the two or three digits number indicates the satisfaction percentage of the consumer with the preceding comment.

You can reach this data in the "totalReviewWithSuggestion.csv" file.

## Second Phase
#### Data Preparation
In this Phase I have cleaned my data with Hazm library and other modifications which has commented in my source code. Then, I have splitted my dataset to parts for training and testing.

## Third Phase
#### Build your own Neural Network in Tensorflow for LSTM
I have built my own graph for calculation the sentiments of each sentence based on the scores mentioning above.

## Forth Phase
#### compute the word embeddings
In this phase I have used the precious guide from other repository and I have included that in my repository in the "ipynb_checkpoints" folder for more guidence to who wants to become more familiar with ehat I have done. As it is mentioned there using a one-hot method is too cumbersome and unefficient I have prepared a dictionary of my vocabulary and convert that to a feature vector.

## Fifth Phase
#### Training and Testing
I have trained and have tested the code on my own dataset which have high accuracy near 93 percent.

#### Thanks
Thanks to Mr. [AminMozhgani](https://github.com/AminMozhgani) for his devoted assistance through the project.

