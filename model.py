"""model.py

Build the language model using encoder-decoder with attention
"""

import numpy as np
import tensorflow as tf
import time
import os

class Model():

    def __init__(self, char2idx, idx2char, param_dict):

        # parse the parameters
        vocab_size = param_dict['vocab_size']
        embedding_dim = param_dict['embedding_dim']
        units = param_dict['units']
        num_layers = param_dict['num_layers']
        dropout = param_dict['dropout']

        # save global variables
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.param_dict = param_dict
        self.embedding_dim = embedding_dim

        # create encoder
        self.encoder = Encoder(vocab_size, embedding_dim, units, num_layers, dropout)

        # create decoder
        self.decoder = Decoder(vocab_size, embedding_dim, units, num_layers, dropout)

        # create optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # create checkpoint
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)

    def load_weights(self, checkpoint_dir):
        """ load weights of the TF model """

        try:
            # restore from model_dir
            status = self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpoint_dir)))
        except:
            print("No checkpoint found at {}".format(checkpoint_dir))

    def save_weights(self, checkpoint_dir):
        """ save the model weights """

        model_checkpoint = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=1)
        model_checkpoint.save()

    def train_word2vec(self, train_data, iter, word2vec_path):
        """ train the word2vec model """

        from gensim import models

        self.wv_model = models.Word2Vec(
            train_data,
            size=self.embedding_dim,
            min_count=1,
            window=len(train_data[0]),
            iter=iter
        )

        self.wv_model.save(word2vec_path)

    def load_word2vec(self, word2vec_path):
        """ load the pretrained word2vec model """

        from gensim import models

        self.wv_model = models.Word2Vec.load(word2vec_path)

    def transfer_embedding_weights(self, idx2char):
        """ use the word2vec weights as the embedding matrix """

        # get the embedding matrix
        embedding_matrix = self._get_word2vec_matrix(self.wv_model, idx2char, self.embedding_dim)

        # set the embedding matrix values to encoder and decoder
        self.encoder.set_embedding_matrix(embedding_matrix)
        self.decoder.set_embedding_matrix(embedding_matrix)

    def _get_word2vec_matrix(self, wv_model, idx2char, embedding_dim):
        """ return the word2vec matrix, reordered by char2idx vocabulary index """

        count = 0
        embedding_matrix = np.zeros((len(idx2char), embedding_dim))
        for idx, char in enumerate(idx2char):
            if char in wv_model.wv.vocab:
                wv_idx = wv_model.wv.vocab[char].index
                embedding_matrix[idx] = wv_model.wv.vectors[wv_idx]
            else:
                embedding_matrix[idx] = np.zeros((embedding_dim, ))
                count += 1

        print("There are {} characters not in the word2vec embedding".format(count))

        return embedding_matrix

    def _get_repeated_chars(self, inputs):
        """ get repeated characters by index in the input """

        first_seen_idx = {}
        repeated_chars = {}
        for i, char_idx in enumerate(inputs):
            if char_idx not in first_seen_idx.keys():
                first_seen_idx[char_idx] = i
            else:
                repeated_chars[i] = first_seen_idx[char_idx]

        return repeated_chars

    def predict(self, sentence, beam_width=20):
        """ use the model to predict """

        # check input sanity
        for i in sentence:
            if i not in self.char2idx.keys():
                return "抱歉，您的输入中有我还没学会的生僻字，呜呜呜"

        inputs = [self.char2idx[i] for i in sentence]
        sentence_len = len(inputs)

        # get repeated chars in the input
        repeated_chars = self._get_repeated_chars(inputs)

        inputs = tf.convert_to_tensor([inputs])

        enc_out, enc_hidden = self.encoder(inputs, training=False)
        dec_hidden = enc_hidden

        # the tuple that contains the score, the sequence, the hidden state, and the attention weights
        results = [(0, ['<s>'], dec_hidden)]

        for t in range(sentence_len):

            results_new = []
            for result in results:

                # take the score and all historical characters without the new prediction
                score = result[0]
                seq = result[1]
                dec_hidden = result[2]

                # update used character list to avoid bad prediction
                # include "," in the used char list
                used_char_idx = list(inputs[0].numpy()) + [self.char2idx[x] for x in seq]
                if self.char2idx['，'] not in list(inputs[0].numpy()):
                    used_char_idx += [self.char2idx['，']]

                # take the last element as the input
                dec_input = tf.expand_dims([self.char2idx[seq[-1]]], 0)

                # predict
                predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                dec_hidden,
                                                                enc_out,
                                                                training=False)

                predictions = tf.nn.softmax(predictions)

                if t in repeated_chars.keys():
                    # if this is a repeated chars in the input
                    # then the output should be repeated as well
                    first_seen_idx = repeated_chars[t]
                    prediction_id = self.char2idx[seq[first_seen_idx+1]]
                    score_new = score + np.log(predictions[0][prediction_id].numpy())
                    results_new.append((score_new, seq+[self.idx2char[prediction_id]], dec_hidden))
                else:
                    # if no repeated chars
                    # then take the k most likely predictions
                    _, top_k = tf.math.top_k(predictions, beam_width)

                    for prediction_id in top_k.numpy()[0]:
                        if prediction_id not in used_char_idx:
                            score_new = score + np.log(predictions[0][prediction_id].numpy())
                            results_new.append((score_new, seq+[self.idx2char[prediction_id]], dec_hidden))

            # keep only top k results in the beam search
            results = sorted(results_new, key=lambda x:x[0])[-beam_width:]

        # take the most likely one
        result = max(results, key=lambda x:x[0])[1][1:]

        return "".join(result)

    def _loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)

    def _preprocess_dataset(self, train, target, batch_size):
        """ preprocess the dataset """

        from sklearn.model_selection import train_test_split

        # train/eval split
        input_tensor_train, input_tensor_eval, target_tensor_train, target_tensor_eval = train_test_split(
            train,
            target,
            test_size=0.1,
            random_state=42
        )

        # train dataset
        buffer_size = len(input_tensor_train)
        steps_per_epoch = len(input_tensor_train)//batch_size
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # eval dataset
        buffer_size_eval = len(input_tensor_eval)
        steps_per_epoch_eval = len(input_tensor_eval)//batch_size
        dataset_eval = tf.data.Dataset.from_tensor_slices((input_tensor_eval, target_tensor_eval)).shuffle(buffer_size_eval)
        dataset_eval = dataset_eval.batch(batch_size, drop_remainder=True)

        return dataset, dataset_eval, steps_per_epoch, steps_per_epoch_eval

    def train(self, train, target, start_epoch, num_epoch, log_dir, checkpoint_dir, batch_size, learning_rate):
        """ train the model """

        # preprocess the dataset
        dataset, dataset_eval, steps_per_epoch, steps_per_epoch_eval = self._preprocess_dataset(train, target, batch_size)

        # create loss object
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        # update the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # checkpoint and log
        checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=2)

        # create log file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = open("{}/training.log".format(log_dir), 'w')

        for epoch in range(start_epoch, start_epoch+num_epoch):
            start = time.time()
            time_last = start

            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

                batch_loss = self.train_step(inp, targ, training=True)

                total_loss += batch_loss

                if batch % 1000 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))

                    print('Time taken for 1000 batch {} sec\n'.format(time.time() - time_last))
                    time_last = time.time()

            # saving (checkpoint) the model every 1 epoch
            checkpoint_manager.save()

            # calculate the evaluation set metrics
            eval_loss = 0
            for (batch, (inp, targ)) in enumerate(dataset_eval.take(steps_per_epoch_eval)):
                batch_loss = self.train_step(inp, targ, training=False)
                eval_loss += batch_loss
            print('Evaluation Loss {:.4f}'.format(eval_loss / steps_per_epoch_eval))

            # write metrics to log
            log_file.write('{} {:.4f} {:.4f}\n'.format(epoch,
                                               total_loss / steps_per_epoch,
                                               eval_loss / steps_per_epoch_eval))

            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        log_file.close()

    @tf.function
    def train_step(self, inp, targ, training=True):
        loss = 0

        with tf.GradientTape() as tape:

            enc_output, enc_hidden = self.encoder(inp, training=True)
            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.char2idx['<s>']] * inp.shape[0], 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):

                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output, training=True)

                loss += self._loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[0]))

        if training:
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, num_layers, dropout):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.num_layers = num_layers // 2   ## because we have bidirectional

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            trainable=False
        )

        lstm_cells = [tf.keras.layers.LSTMCell(self.enc_units, recurrent_initializer='glorot_uniform', dropout=dropout) for _ in range(num_layers)]
        lstm_stacked = tf.keras.layers.StackedRNNCells(lstm_cells)
        rnn = tf.keras.layers.RNN(lstm_stacked, return_sequences=True, return_state=True)
        self.bilayers = tf.keras.layers.Bidirectional(rnn)

    def set_embedding_matrix(self, embedding_matrix):
        """ use the embedding matrix as the pretrained embedding layer """

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False
        )

    def call(self, x, training=False):
        x = self.embedding(x)

        outputs = self.bilayers(x, training=training)

        # the returned output from the bidirectional LSTM layers
        output = outputs[0]

        # the hidden_state from the bidirectional LSTM layers
        # states = [layer_1, layer_2, etc.]
        # for each layer, hidden = tf.concat([forward_hidden, backward_hidden], -1)
        state_f = outputs[1:self.num_layers+1]
        state_b = outputs[self.num_layers+1:]
        states = []
        for i in range(self.num_layers):
            states.append([state_f[i][0], state_f[i][1]]) # hidden states in the forward i-th layer
            states.append([state_b[i][0], state_b[i][1]]) # hidden states in the backward i-th layer

        return output, states

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, num_layers, dropout):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            trainable=False
        )

        lstm_cells = [tf.keras.layers.LSTMCell(self.dec_units, recurrent_initializer='glorot_uniform', dropout=dropout) for _ in range(num_layers)]
        lstm_stacked = tf.keras.layers.StackedRNNCells(lstm_cells)
        self.rnn = tf.keras.layers.RNN(lstm_stacked, return_sequences=True, return_state=True)

        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def set_embedding_matrix(self, embedding_matrix):
        """ use the embedding matrix as the pretrained embedding layer """

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False
        )

    def call(self, x, hidden_states, enc_output, training=False):

        # the hidden_states passed in is a list
        hidden_states_concat = tf.reshape(hidden_states, (hidden_states[0][0].shape[0], -1))

        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden_states_concat, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the LSTM
        outputs = self.rnn(x, initial_state=hidden_states, training=False)

        # take the output
        output = outputs[0]

        # take the states
        states = []
        for i in range(self.num_layers):
            states.append(outputs[i+1]) # h + c

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, states, attention_weights
