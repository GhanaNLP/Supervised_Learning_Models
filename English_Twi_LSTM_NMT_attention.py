from __future__ import absolute_import, division, print_function, unicode_literals
# #### sequence to sequence with attention  model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import csv
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
import itertools
from pickle import load
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from pickle import load
from numpy import array
from numpy import argmax
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import unicodedata
import numpy as np
import os
import io
import time
import pandas as pd
from collections import defaultdict
# encoding=utf8
from importlib import reload
import sys
reload(sys)
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#read twi and english data
with open('/home/nanaboateng/notebooks/jw300.en-tw.tw', 'r') as f:
     #Twi_data = f.readlines()
      Twi_data= f.read().splitlines()

#Twi_data[0:5] 

english_data = [line.rstrip() for line in open('/home/nanaboateng/notebooks/jw300.en-tw.en')]
#english_data[0:5]




#number of examples to use for training
num_examples= 10000

#num_examples = 606197


# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence_english(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  #w = re.sub(r'[^Ɔ-ɔɛƐ]+', r' ', w)
  #strip() Parameters
  #chars (optional) - a string specifying the set of characters to be removed.
  #If the chars argument is not provided, all leading and trailing whitespaces are removed from the string.
  w = w.rstrip().strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w




  
def preprocess_sentence_twi(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-ZɛƐɔƆ?.!,¿]+", " ", w)
  #w = re.sub(r'[^Ɔ-ɔɛƐ]+', r' ', w)
  #strip() Parameters
  #chars (optional) - a string specifying the set of characters to be removed.
  #If the chars argument is not provided, all leading and trailing whitespaces are removed from the string.
  w = w.rstrip().strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w




#type(english_data)
# preprocess english and twi data
english_d  = list(map(preprocess_sentence_english,english_data))
#english_d[0:5]

twi_d  = list(map(preprocess_sentence_twi,Twi_data))
#twi_d[0:5]



# reduce the data to lines with length <= 40
#max_length = 40
#seq_length= [len(word) for word in english_d]
#d= pd.DataFrame({"English":english_d,"Twi":twi_d,"seq_length":seq_length}).query('seq_length <= 40 and seq_length > 17')
#english_d = d.English.tolist()
#twi_d  = d.Twi.tolist()
#all_data= (twi_d[0:30000]),(english_d[0:30000])
#[0][0:5]





def create_dataset_eng(path, num_examples):
  lines = io.open(path, encoding='utf-8').read().strip().split('\n')

  word_pairs = [[preprocess_sentence_english(w) for w in l.split('\t')]  for l in lines[:num_examples]]

  return zip(*word_pairs)


#encoding = ‘utf-8-sig’ is added to overcome the issue when exporting ‘Non-English’ languages.
def create_dataset_twi(path, num_examples):
  lines = io.open(path, encoding='utf-8-sig').read().strip().split('\n')

  word_pairs = [[preprocess_sentence_twi(w) for w in l.split('\t')]  for l in lines[:num_examples]]

  return zip(*word_pairs)



#oov_token: if given, it will be added to word_index and used to replace out-of-vocabulary
# words during text_to_sequence calls
en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',oov_token='OOV',lower=True)
en_tokenizer.fit_on_texts(english_d[:num_examples])

data_en = en_tokenizer.texts_to_sequences(english_d[:num_examples])
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')

#twi_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
twi_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',oov_token='OOV',lower=True)
twi_tokenizer.fit_on_texts(twi_d[:num_examples])

data_twi = twi_tokenizer.texts_to_sequences(twi_d[:num_examples])
data_twi = tf.keras.preprocessing.sequence.pad_sequences(data_twi,padding='post')

print(len(english_d[:num_examples]))
print(len(twi_d[:num_examples]))
#print(data_twi.shape)
#print(data_en.shape)


def max_len(tensor):
    #print( np.argmax([len(t) for t in tensor]))
    return max( len(t) for t in tensor)
    
 
X_train,  X_test, Y_train, Y_test = train_test_split(data_en,data_twi,test_size=0.2)
BATCH_SIZE = 64
BUFFER_SIZE = len(X_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32   #used to initialize DecoderCell Zero state
epsilon =  1e-8

Tx = max_len(data_en)
Ty = max_len(data_twi)  

input_vocab_size = len(en_tokenizer.word_index)+1  
output_vocab_size = len(twi_tokenizer.word_index)+ 1
dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
example_X, example_Y = next(iter(dataset))
print(example_X.shape) 
print(example_Y.shape)
print(Tx)
print(Ty) 




#ENCODER
class EncoderNetwork(tf.keras.Model):
    def __init__(self,input_vocab_size,embedding_dims, rnn_units ):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, 
                                                     return_state=True )
    
#DECODER
class DecoderNetwork(tf.keras.Model):
    def __init__(self,output_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims) 
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(dense_units,None,BATCH_SIZE*[Tx])
        self.rnn_cell =  self.build_rnn_cell(BATCH_SIZE)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, units,memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory = memory, 
                                          memory_sequence_length=memory_sequence_length)
        #return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell  
    def build_rnn_cell(self, batch_size ):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=dense_units)
        return rnn_cell
    
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state

encoderNetwork = EncoderNetwork(input_vocab_size,embedding_dims, rnn_units)
decoderNetwork = DecoderNetwork(output_vocab_size,embedding_dims, rnn_units)
optimizer = tf.keras.optimizers.Adam()



def loss_function(y_pred, y):
   
    #shape of y [batch_size, ty]
    #shape of y_pred [batch_size, Ty, output_vocab_size] 
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred + epsilon)
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss

#Use the tf.GradientTape context to calculate the gradients used to optimize your model:
def train_step(input_batch, output_batch,encoder_initial_cell_state):
    #initialize loss = 0
    loss = 0
    with tf.GradientTape() as tape:
        encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
        a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
                                                        initial_state =encoder_initial_cell_state)

        #[last step activations,last memory_state] of encoder passed as input to decoder Network
        
         
        # Prepare correct Decoder input & output sequence data
        decoder_input = output_batch[:,:-1] # ignore <end>
        #compare logits with timestepped +1 version of decoder_input
        decoder_output = output_batch[:,1:] #ignore <start>


        # Decoder Embeddings
        decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

        #Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
        decoderNetwork.attention_mechanism.setup_memory(a)
        decoder_initial_state = decoderNetwork.build_decoder_initial_state(BATCH_SIZE,
                                                                           encoder_state=[a_tx, c_tx],
                                                                           Dtype=tf.float32)
        
        #BasicDecoderOutput        
        outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
                                               sequence_length=BATCH_SIZE*[Ty-1])

        logits = outputs.rnn_output
        #Calculate loss

        loss = loss_function(logits, decoder_output)

    #Returns the list of all layer variables / weights.
    variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables  
    # differentiate loss wrt variables
    gradients = tape.gradient(loss, variables)

    #grads_and_vars – List of(gradient, variable) pairs.
    grads_and_vars = zip(gradients,variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss



#RNN LSTM hidden and memory state initializer
def initialize_initial_state():
        return [tf.zeros((BATCH_SIZE, rnn_units)), tf.zeros((BATCH_SIZE, rnn_units))]  



#checkpoint_dir = './training_lstm_checkpoints -  /home/nanaboateng/Desktop/training_lstm_checkpoints'
checkpoint_dir = './training_lstm_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoderNetwork,
                                 decoder=decoderNetwork)  




epochs = 50
for epoch in range(epochs):
    #for i in range(1, epochs+1):
        start = time.time()
        encoder_initial_cell_state = initialize_initial_state()
        total_loss = 0.0

        for ( batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
             batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
             total_loss += batch_loss
             if (batch+1)%100 == 0: 
                print("total loss: {} epoch {} batch {} ".format(batch_loss.numpy(), epoch+1, batch+1))
             # saving (checkpoint) the model every 2 epochs
        if (epoch )% 5  == 0:
           checkpoint.save(file_prefix = checkpoint_prefix)
           print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))









#In this section we evaluate our model on a raw_input converted to twi, for this the entire sentence has to be passed
#through the length of the model, for this we use greedsampler to run through the decoder
#and the final embedding matrix trained on the data is used to generate embeddings
def evaluate(input_raw):
   #input_raw='how are you'
    #input_raw = ''
# We have a transcrionce again , the word jew is not usedpt file containing English-Twi pairs
# Preprocess X
    input_lines = ['<start> '+input_raw+'']
    input_sequences = [[en_tokenizer.word_index[w] for w in line.split(' ')] for line in input_lines]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                maxlen=Tx, padding='post')
    inp = tf.convert_to_tensor(input_sequences)
#print(inp.shape)
    inference_batch_size = input_sequences.shape[0]
    encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),
                              tf.zeros((inference_batch_size, rnn_units))]
    encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
    a, a_tx, c_tx =   encoderNetwork.encoder_rnnlayer(encoder_emb_inp,
                                                initial_state =encoder_initial_cell_state)
    #print('a_tx :',a_tx.shape)
    #print('c_tx :', c_tx.shape)

    start_tokens = tf.fill([inference_batch_size],twi_tokenizer.word_index['<start>'])

    end_token = twi_tokenizer.word_index['<end>']

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    decoder_input = tf.expand_dims([twi_tokenizer.word_index['<start>']]* inference_batch_size,1)
    decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

    decoder_instance = tfa.seq2seq.BasicDecoder(cell = decoderNetwork.rnn_cell, sampler = greedy_sampler,
                                            output_layer=decoderNetwork.dense_layer)
    decoderNetwork.attention_mechanism.setup_memory(a)
#pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
    #print("decoder_initial_state = [a_tx, c_tx] :",np.array([a_tx, c_tx]).shape)
    decoder_initial_state = decoderNetwork.build_decoder_initial_state(inference_batch_size,
                                                                   encoder_state=[a_tx, c_tx],
                                                                   Dtype=tf.float32)
    #print("\nCompared to simple encoder-decoder without attention, the decoder_initial_state \
     #       is an AttentionWrapperState object containing s_prev tensors and context and alignment vector \n ")
    #print("decoder initial state shape :",np.array(decoder_initial_state).shape)
    #print("decoder_initial_state tensor \n", decoder_initial_state)

# Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
# One heuristic is to decode up to two times the source sentence lengths.
    maximum_iterations = tf.round(tf.reduce_max(Tx) * 2)

#initialize inference decoder
    decoder_embedding_matrix = decoderNetwork.decoder_embedding.variables[0] 
    (first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                             start_tokens = start_tokens,
                             end_token=end_token,
                             initial_state = decoder_initial_state)
#print( first_finished.shape)
    #print("\nfirst_inputs returns the same decoder_input i.e. embedding of  <start> :",first_inputs.shape)
    #print("start_index_emb_avg ", tf.reduce_sum(tf.reduce_mean(first_inputs, axis=0))) # mean along the batch

    inputs = first_inputs
    state = first_state  
    predictions = np.empty((inference_batch_size,0), dtype = np.int32)                                                                             
    for j in range(maximum_iterations):
         outputs, next_state, next_inputs, finished = decoder_instance.step(j,inputs,state)
         inputs = next_inputs
         state = next_state
         outputs = np.expand_dims(outputs.sample_id,axis = -1)
         predictions = np.append(predictions, outputs, axis = -1)
    return  predictions            
  

def translate(sentence):
    #prediction based on our sentence earlier
    print("English Sentence:{}".format(sentence))
    #print(input_raw)
    print("\n Twi Translation:")
    predictions = evaluate(sentence)
    for i in range(len(predictions)):
        
        line = predictions[i,:]
        seq = list(itertools.takewhile( lambda index: index !=2, line))
    return print(" ".join( [twi_tokenizer.index_word[w] for w in seq]))


#sentence = 'dad was upset about that'
sentence = 'kristo nkutoo ne yɛn gyefo .'

translate(sentence)  