from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from tqdm import tqdm

batch_size = 256  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 108674  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = './eng-fra.txt'

input_texts=[]
target_texts=[]
input_char=set()
target_char=set()

with open(data_path,"r",encoding="utf-8") as document:
    lines=document.read().split("\n")
    
for singl_line in tqdm(lines[:num_samples]):
    input_text,target_text=singl_line.split("\t")
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for s_char in input_text:
        if s_char not in input_char:
            input_char.add(s_char)
    for s_char in target_text:
        if s_char not in target_char:
            target_char.add(s_char)
            
            
            
input_char=sorted(list(input_char))
target_char=sorted(list(target_char))
number_encoder_tokens=len(input_char)
number_decoder_tokens=len(target_char)
max_encoder_seq_length=max([len(s) for s in input_texts])
max_decoder_seq_length=max([len(s) for s in target_texts])
print(number_encoder_tokens,number_decoder_tokens,max_encoder_seq_length,max_decoder_seq_length)

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', number_encoder_tokens)
print('Number of unique output tokens:', number_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index=dict([(char,i) for i,char in enumerate(input_char)])
target_token_index=dict([(char,i) for i,char in enumerate(target_char)])

encoder_input_data=np.zeros((len(input_texts),max_encoder_seq_length,number_encoder_tokens),dtype="float32")
decoder_input_data=np.zeros((len(input_texts),max_decoder_seq_length,number_decoder_tokens),dtype="float32")
decoder_target_data=np.zeros((len(input_texts),max_decoder_seq_length,number_decoder_tokens),dtype="float32")

for i,(input_text,output_text) in tqdm(enumerate(zip(input_texts,target_texts))):
    for t, char in enumerate(input_text):
        encoder_input_data[i,t,input_token_index[char]]=1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    
    for t, char in enumerate(output_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.
    
    
encoder_inputs = Input(shape=(None, number_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None, number_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(number_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('model.h5')

    



