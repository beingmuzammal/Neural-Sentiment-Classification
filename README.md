# Neural-Sentiment-Classification
Neural Sentiment Classification aims to classify the sentiment in a document with neural models, which has been the state-of-the-art methods for sentiment classification.
# Evaluation Results
Evaluation results on document-level sentiment classification. Acc.(Accuracy) and RMSE are the evaluation metrics. image

In the above table, baseline models including Majority, Trigram, TextFeature, UPF, AvgWordvec, SSWE, RNTN + RNN, Paragraph Vector, JMARS and UPNN are reported in [Tang et al., 2015].

# Data
We provide IMDB, Yelp13 and Yelp14 datasets we used for sentiment classification in [Download]. The dataset should be decompressed and put in the folder NSC/, NSC+LA/ or NSC+UPA/.

We prepocess the original data to make it satisfy the input format of our codes. The original datasets are released by the paper [Tang et al., 2015]. [Download]

Pre-trained word vectors are learned on each dataset (IMDB, Yelp13, Yelp14) separately.

The dataset in each domain contains seven files, using the following format:

train.txt: training file, format (userid productid class document), split by '\t'.
dev.txt: dev file, same format as train.txt.
test.txt: test file, same format as train.txt.
wordlist.txt: corresponding words with same sequence in pre-trained word vectors, one per line.
usrlist.txt: user ids in each dataset, per one line.
prdlist.txt: product ids in each dataset, per one line.
embinit.save: the pre-trained word embedding file, which is saved as pickle and can be loaded from pickle to numpy arrays.
The trained model can be found at this link.

# Codes
The source codes of various models are put in the folders NSC/src, NSC+LA/src, NSC+UPA/src.

# Train
For training, you need to type the following command in the folder src/ of each model:

THEANO_FLAGS="floatX=float32,device=gpu" python train.py $dataset $class
where dataset is the corresponding dataset folder, class is the number of corresponding domain.

For example, we use the following command when classfing the IMDB document:

THEANO_FLAGS="floatX=float32,device=gpu" python train.py IMDB 10
The training model file will be saved in the folder model/bestmodel/ of each model.

# Test
For testing, you need to type the following command in the folder src/ of each model:

THEANO_FLAGS="floatX=float32,device=gpu" python test.py $dataset $class
where dataset is the corresponding dataset folder, class is the number of corresponding domain.

For example, we use the following command when classfing the IMDB document:

THEANO_FLAGS="floatX=float32,device=gpu" python test.py IMDB 10
The testing result which reports the Accuracy and RMSE will be shown in screen.

