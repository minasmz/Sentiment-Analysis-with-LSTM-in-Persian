# Sentiment-Analysis-in-Persian-Language_LSTM
Sentiment Analysis with LSTM in Persian

## First Phase
#### Data Aquisition
In this repository, I have used LSTM for the prediction of whether people would like or dislike a product based on the previous comments on the Digikala site. I have scraped the data from Digikala and have labeled them based on the stars people who had bought the products gave to them. I have also used another label from the same website which indicates people suggest others to buy that product or not. because many of the comments are noisy and do not provide clean data for us and it is not such a reliable source by adding the second label to the data we can ensure higher accuracy of our training data.

For label clarifications:
    (1) -> Indicates customers suggesting others to buy 
    (2) -> Indicates otherwise
    (3) -> Illusterates a neutral opinion about the product
    (4) -> Customer has rate the product, but not suggested whether to buy it or not.
and the two or three digits number indicates the satisfaction percentage of the consumer with the preceding comment.

You can reach this data in the "totalReviewWithSuggestion.csv" file.

## Second Phase
#### Data Preparation
In this Phase, I have cleaned my data with Hazm library and other modifications which have been commented on in my source code. Then, I have split my dataset into parts for training and testing.

## Third Phase
#### Build your own Neural Network in Tensorflow for LSTM
I have built my graph for calculation the sentiments of each sentence based on the scores mentioning above.

## Forth Phase
#### compute the word embeddings
In this phase, I have used the precious guide from another repository and I have included that in my repository in the "ipynb_checkpoints" folder for more guidance to those who want to become more familiar with what I have done. As is mentioned there using a one-hot method is too cumbersome and inefficient I have prepared a dictionary of my vocabulary and convert that to a feature vector.

## Fifth Phase
#### Training and Testing
I have trained and have tested the code on my dataset which has high accuracy near 93 percent.

#### Thanks
Thanks to Mr. [AminMozhgani](https://github.com/AminMozhgani) for his devoted assistance through the project.

