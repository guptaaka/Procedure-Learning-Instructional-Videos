Machine Learning for Procedure Learning from Instructions

ABSTRACT

We explore and compare the different Natural Language Processing
models to generate word embeddings and develop weakly supervised
machine learning algorithms on these embeddings for extracting the
grammar of complex tasks from instructional video data by using
the narration of its audio input. By exploring the audio narration,
we find interesting facts about its relation with the video as the
speaker tends to narrate the step while carrying it out; they might
give a precise note, or elaborate the step by talking about different
scenarios, or talk about the step before carrying it out, or after the
step is executed. The challenge we try to address is narrowing down
the audio narration to spot the key steps used in the video to carry
out a task. We explore the existing natural language processing
models and localize the key steps in a video by computing cosine
similarity of the sentences in audio narration with the task-related
key steps. Followed by building a neural network on top of the word
vectors, we show the effectiveness of neural networks to localize key
steps in video based on the audio narration data and how it overruns
the performance of the simple word representations of the narration
text.

INTRODUCTION

Understanding the key steps from an instructional video to carry
out a task is a complex problem. Training a network to localize
the key steps requires humongous amount of data. Extracting this
information from a new video given the annotations can be achieved
with procedure learning. Procedure learning can be used to train
autonomous agents perform complex tasks, or help humans identify
key steps, or build huge knowledge bases of instructions. Extracting
features from a video is a challenging task. Identification of key
steps from a video can be decomposed into multiple sub-problems.
Key steps can be extracted by understanding the grammar of the
visual data sans the audio, and vice-versa, followed by combining
the results from both audio and video. It is imperative to study both
the audio and visual data and the similarities and differences between
the two. A person could be talking about various possibilities to
achieve a task, or laying out general principles for achieving one,
or the possible outcomes from carrying out a particular key step, or
different alternatives to the current approach being carried out in the
video. So, it becomes important to filter the desired content that truly
represents the key steps required to perform a task. Understanding
the audio narration without using the visual cues makes it harder
to extract the mapping between the two since the natural language
can be altered just by a few words to represent different scenarios.
Even defining the key steps before studying the narration might not
always solve the problem since a model can study a few words or
sentences at a time. Building context while studying this model
can help in achieving a better outcome, otherwise distinguishing a
humorous sentence in a plain narrative might not be possible.

To explore the audio data has many possibilities: training a model
on the audio input directly, or converting the audio to narration and
build models on the textual data. The audio input is expected to
have a lot of noise which needs careful filtering to extract the real
information from a video. Running models directly on the audio
signals caters to noise a lot more than expected. Audio data can be
cleaned in a better way when the audio input is converted first to the
textual narration after carefully examining and filtering the results
and then refined again by cleaning the textual narration. Text can be
studied separately while also working on the visual data. Running
models on the textual narration of the audio data provides interesting
key sights in understanding the instructional videos. The person in
video tends to narrate the task while carrying it out, so it can provide
a crisper result when combined with a model running on the visual
data as compared to a model running only the visual data. However,
narrations and visual data might have some misalignment, in that, the
speaker sometimes misses to put the task in clear words while putting
the task in action, or introduces the steps before actually carrying
those out, or vice-versa. So, interpreting the natural language of the
narration to localize the key steps in the video is a very challenging
task but equally critical in procedure learning.

Extracting grammar from natural language has always been an
interesting problem. A neural network model can be trained to
learn word associations from a large corpus of text, detect synonymous
words, or capture the context of a word in a document. It is
an important task to identify the right type of model that fits in a
given scenario. The Natural Language Processing models can be
broadly divided into context-free and contextual models. Both types
of models cater to different utilities. A context-free model takes one
word from the document at a time and generates a vector for the
word, so a single word representation is generated for each word
in the vocabulary, no matter what context it is being used in. For
example, Word2Vec and GloVe can be used to detect similarities
mathematically, providing a good tool for building a recommendation
system. Contextual models, on the other hand, build a word
representation depending on the context where that word occurs,
meaning that the same word in different contexts can have different
representations. For example, ELMo and Bert can be used to
generate contextual word embeddings to build a good search engine.

EXPERIMENTS AND RESULTS

Analyzing the audio narration of the instruction videos involves
using Natural Language Processing and Neural Networks. We first
prepare the dataset and start training our models on this dataset. We
localize the key steps by using different NLP models and find out
which model fits best for this use case. Next, we move on to enhancing
the results by designing a neural network on word embeddings
produced by the NLP models. Lastly, we compare and show how
the results improve significantly with the latter enhancement, i.e., by
using neural network on the word embeddings produced by the NLP
models extract deeper information and yields a better performance.

Dataset

Building a model on the video narration that spans across multiple
instructional tasks requires a good deal of instructional video dataset
with carefully collected and cleaned audio narration. We use Procedure
Learning (ProceL) dataset6 for research on the audio narration.
The ProceL is a medium-scale dataset of 12 diverse tasks, such
as perform cardiopulmonary resuscitation (CPR), make coffee and
assemble clarinet. Each task consists of about 60 videos and on an
average contains 8 key-steps. Each task has a grammar of key-steps,
e.g. ‘perform CPR’ consists of ‘call emergency’, ‘check breathing’,
‘check dangerous’, ‘check pulse’, ‘check response’, ‘open airway’,
‘give compression’ and ‘give breath’. Each video is also annotated
with the key-steps based on the visual data; each video may not
contain all the key steps and the order of key steps in some videos
might be different, as there are multiple ways to perform the same
task.

Annotations

Given the key steps for a task and also for each video for the task,
we annotated each sentence from the narration with relevant key
steps, if present in the sentence. Some complex sentences covered
multiple key steps while some of the sentences did not cover any
key steps. For example, in ‘perform CPR’, the sentence ‘start the
CPR cycle 30 compressions 2 breaths’ covers two key steps ‘give
compression’ and ‘give breath’, while the sentence ‘you can’t help
anyone if you become a victim too’ covers no key step. We did this
for all sentences in the 60 videos of ‘make coffee’ and ‘perform
CPR’ tasks. For rest of the tasks, we covered 10 videos for each
task, annotating every sentence that occurs in these videos.
Sentences were annotated with relevant key steps by taking one
sentence at a time, picked exclusively. That is to say that while
annotating a sentence, its adjoining sentences were not considered to
add a context to the current sentence. While some sentences specify
one or more key steps explicitly, others provide a hint to one of the
key steps. For example, ‘tilt the head back lift the chin and give
two breaths’ is clear to talk about ‘give breath’ key step while ‘so
checking for circulation in a victim’ is a little ambiguous, in that
it could be mentioning either the ‘check breathing’ or the ‘check
pulse’ key step or both. We distinguish the two with two different
labels ‘confident’ for the former and ‘weakly confident’ for the
latter, trying to mark as many labels with ‘confident’ as possible and
narrowing down the occurrence of ‘weakly confident’ steps in the
text. This provided us a cleaner dataset to work with the confident
key steps and establish a model that works with good accuracy, with
a possibility of improvement by extending the training with both
types of labels.

Table 1: Similarity between key steps obtained from video and audio
narration
Task Levenshtein Damerau Levenshtein Jaro
Assemble Clarinet 0.571 0.577 0.944
Change Iphone Battery 0.545 0.554 0.919
Change Tire 0.440 0.440 0.919
Change Toilet Seat 0.504 0.512 0.918
Jump Car 0.550 0.560 0.917
Make Coffee 0.936 0.937 0.988
Make Pbj Sandwich 0.653 0.653 0.926
Make Salmon Sandwich 0.720 0.720 0.918
Perform CPR 0.930 0.931 0.984
Repot Plant 0.727 0.727 0.924
Setup Chromecast 0.799 0.799 0.924


Word embeddings

Since a computer works best with numbers, we transform the words
into relevant word embeddings. A word embedding is a mapping of a
variable or a word to a vector with continuous numbers which can be
used to extract the features of the text. Words with same meaning are
given a similar representation in the n-dimensional space. Finding
word embeddings is another field of research closely related to this
task. An ideal word embedding would accurately establish similarity
between the words, semantic relations, meaning and context of the
words. In context-free word embedding the representation of a word
is determined irrespective of the meaning of the word in a particular
sentence. So a single word representation is generated for each
word in the vocabulary, no matter what context it is being used in.
Contextual models, on the other hand, build a word representation
depending on the context where that word occurs, meaning that the
same word in different contexts can have different representations.
We first build embeddings for each sentence by using different
NLP models. We explore both context-free and contextual models
to study this problem. After obtaining word embeddings for a sentence,
we study different approaches to find key steps covered in a
sentence. A sentence can represent none, one or multiple key steps,
as discussed previously.
Starting with the Word2Vec context-free model, we find word
embedding for each sentence in each of the videos for a task. For
finding word embeddings, we first clean the data by removing all
punctuation and we take one word from the sentence at a time and
get the 300-dimensional vector representation for the word if present
in the model’s vocabulary. To compute a single embedding for the
sentence, we use MaxPooling to combine the vector results from
each word in the sentence. This 300-dimensional vector represents
one sentence and can be used to find the key-step covered by it, if
any.
Next, we use the GloVe context-free model to find embeddings for
all sentences in each video of a task. We use the ‘glove-twitter-25’
API for building the word embeddings. This model produces 25-
dimensional vector for each word, yielding a 25-dimensional vector
for a sentence after we use MaxPooling on vectors for all words
in the sentence. Similarly, we also use ELMo and Bert contextual
word embedding models in our comparison. These models take one
sentence as an input at a time and generate a vector for that sentence.
So, MaxPooling layer is not needed for the two contextual models.
ELMo produces a 1024-dimensional vector for a sentence, while
Bert produces 768-dimensional vector for a sentence.

Word embeddings and Cosine Similarity

Cosine similarity is a measure of similarity between two non-zero
vectors of an inner product space. This metric can be used to measure
how similar two sentences are irrespective of their sizes. To use this
approach, we compute vector embeddings for all key steps in a video
using each of the four models, Word2Vec, GloVe, ELMo and Bert.
Using one model at a time, we compute cosine similarity between
the n-dimensional vector of a sentence and the n-dimensional vector
of each key step. This way, we calculate the cosine similarity for
each sentence of every video in a task.

Table 2: F1 scores from the cosine similarity model
Task Word2Vec Bert ELMo GloVe
Assemble Clarinet 0.18 0.048 0.2 0.068
Change Iphone Battery 0.09 0.048 0.12 0.058
Change Tire 0.05 0.043 0.1 0.035
Change Toilet Seat 0.07 0.029 0.2 0.069
Jump Car 0.07 0.044 0.17 0.114
Make Coffee 0.18 0.051 0.22 0.057
Make Pbj Sandwich 0.06 0.063 0.2 0.077
Make Salmon Sandwich 0.08 0.051 0.15 0.076
Perform CPR 0.5 0.108 0.2 0.113
Repot Plant 0.02 0.050 0.1 0.094
Setup Chromecast 0.09 0.034 0.1 0.044

Word embeddings and Neural Network

For this milestone, we use Neural Network to extract denser information
from the sentence vectors to find the key steps covered in
the sentence. While the NLP models study each word with/without
the context and provide embeddings that help to identify similarities
between words or context of a word in a sentence, Neural network
can help uncover some unseen features from the dataset by training
the model along with the expected output for the training dataset.
This problem serves as a multi-label classification problem, because
each video has multiple key steps or classes and a sentence from the
video can be classified with none/one/many key-steps, making it a
multi-label classification problem.
We use Dense Neural layers to produce better results from the sentence
embeddings. We define the input as a vector of n-dimensions
(n depending on the NLP model used) for each sentence and the
output layer as a vector of probabilities of length equal to the number
of key steps defined for the video. We use a dense layer of 64 units
with a Rectified Linear activation to throw the negative results. We
define the output layer as a dense layers with number of units same
as the number of key steps in the video with a Sigmoid activation
to obtain a 0-1 range probability of the occurrence of a key step in
a sentence. We compile the model with binary cross-entropy loss
function and Adam optimizer. Using a 70% training to test ratio,
we train the model and evaluate its results. Even as the amount of
training data is low, we find good results from our Neural network
model.
Table 3 and Table 4 show the best results we obtain from our
Neural Network model for each video. We run this model on the
sentence vectors obtained from Word2Vec and ELMo models. The
tasks ‘Make Coffee’ and ‘Perform CPR’ again outperform the rest
of the tasks for the same reason i.e. because we have 60 videos
data for each of the two tasks as compared to 10 videos data for the
other tasks. We also observe from Table 2, Table 3 and Table 4 that
for each task, the F1 score obtained with Neural Network layers is
much better as compared to the F1 score obtained with the previous
approach of localizing the key steps based on cosine similarity and a
threshold value. We get good results by using ELMo embeddings for
most of the tasks, except for ‘Perform CPR’, ‘Make Pbj Sandwich’
and ‘Setup Chromecast’ tasks, where Word2Vec embeddings yield
better results.

Table 3: F1 scores from Neural Network model with Word2Vec embeddings
Task F1 score Precision Recall
Assemble Clarinet 0.303 0.405 0.242
Change Iphone Battery 0.265 0.361 0.210
Change Tire 0.333 0.417 0.278
Change Toilet Seat 0.314 0.455 0.240
Jump Car 0.276 0.500 0.190
Make Coffee 0.504 0.528 0.483
Make Pbj Sandwich 0.216 0.333 0.160
Make Salmon Sandwich 0.150 0.250 0.107
Perform CPR 0.834 0.880 0.792
Repot Plant 0.074 0.500 0.040
Setup Chromecast 0.437 0.514 0.380

Table 4: F1 scores from Neural Network model with ELMo embeddings
Task F1 score Precision Recall
Assemble Clarinet 0.318 0.538 0.226
Change Iphone Battery 0.326 0.5 0.242
Change Tire 0.433 0.542 0.361
Change Toilet Seat 0.367 0.628 0.260
Jump Car 0.357 0.625 0.250
Make Coffee 0.538 0.606 0.483
Make Pbj Sandwich 0.158 0.231 0.120
Make Salmon Sandwich 0.222 0.294 0.179
Perform CPR 0.772 0.826 0.725
Repot Plant 0.118 0.222 0.080
Setup Chromecast 0.4 0.533 0.320

CONCLUSION

We develop a neural network on the sentence vector embeddings
and show how the neural network model performs a better task in
localizing the key steps in an instructional video’s textual narration
data as compared to a model that localizes the key steps based on
cosine similarities of the sentence/key step vector embeddings.
