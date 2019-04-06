#UinversalSpeech 
##Providing an exhaustive resouce for those in need. 

Application
The application for our hack is for the travel industry. It allows travelers to easily translate the local language, in this case english to french, and provide a summary and some insight about the language being said. Often it is hard for a foreigner to understand the idiomatic and sarcastic expressions of a language, especially one like english which has had decades of redditors who have been developing new slang and sarcastic phrases. The more global use case for this hack is for anyone who struggles with understanding the local language.

Inspiration
All four of us really enjoy traveling to foreign countries, but we have often encountered language barriers at local restaurants and downtown. So we set out to simplify the travel experience with the Comprehensive Translator!

What it does
Our Comprehensive Translator comes with three features: Real-time Speech Translation, Sarcasm Detector, and Sentiment Analysis. We currently have one langauge translation option: English to French.

How we built it
We trained the AI computer algorithm with several sample data sets for deep learning.

REAL-TIME SPEECH TRANSLATOR
We used the Google Speech to Text API to convert spoken English to text. Then, we translated English text to French text with Seq2Seq model in Tensorflow. Finally, the Google Cloud Text to Speech (TTS) API allowed us to synthesize the spoken French reply.

SARCASM DETECTOR
The code for the sarcasm detector was completely from scratch, using text input as fasttext and using multiple rnns as a way to try and classify data. The format of the sarcasm data was the reddit comment, and the parent comment that induced the comment. Both of these were inputted to a 2 and 3 filter size convolution to capture bigrams and trigrams, then put into an rnn each. The outputs of the rnns were then concatenated and put into dense layers for the final prediction.

SENTIMENT ANALYSIS
We used the Keras Library which allows us to scan and remove all special symbols and syntax from the text file. After using Tockenizer to convert to vectors, the algorithm will output positive, neutral, or negative sentiment. This algorithm was trained on Twitter GOP Debate Dataset.

Challenges we ran into
Throughout this project, we frequently ran into problems such as a difficulty in training the sarcasm detector due to the ambiguity of the data, and the model frequently learned to simply output a single class without actually ever outputting the second class. It was also difficult to install some dependancies of the code onto the mac due to lack of admin access, but this was worked around using a different machine and a usb. Another issue we ran into was that karas would not correctly save the model.h5 file as a checkpoint, so we had to run the training of the sentiment classifier on a different device. The primary challenge was finding enough memory to load the full datasets.

Accomplishments that we're proud of
We are proud of the numerous APIs we were able to integrate into our hack, and proud of the success of our machine translation system. We are also proud of being able to use the latest version of tensorflow, 2.0.0 which completely changed the dynamics of our code.

What we learned
We learned about the multiple deep learning strategies used in NLP. # of us completely learned python from scratch and how to build a simple GUI. We also learned how important it is to know how to use APIs in python as opposed to "reinventing the wheel"

What's next for Comprehensive Translator
Abstractive summarization, accents in ascii encoding, more language options and Optical Character Recognition could be paired with the translator to translate written/printed text with more functionality and comprehensibility. These features will allow for a more comprehensive result, that is likely to give much more information to the new english speaker. Such a translator is both easy to use, and runs in real time (about 3 seconds). This runtime can be sped up by migrating all apis on the local computer instead of relying on internet speeds to access the api through python.

Built With
python
tensorflow
keras
Try it out
GitHub Repo
