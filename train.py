from reader import CoupletReader
from model import Model
import numpy as np
import tensorflow as tf
import pickle
import configparser

def print_hardware():
    print("GPU: {}".format(tf.test.gpu_device_name()))
    print("CPU/GPU Type: {}".format(tf.python.client.device_lib.list_local_devices()))

def print_config(config, sections):
    for section in sections:
        for key in config[section]:
            print("{} = {}".format(key, config[section][key]))

if __name__ == "__main__":

    # print the hardware
    print_hardware()

    # load and print the configs
    print('Loading configs')
    config = configparser.ConfigParser()
    config.read('baseconfig.ini')
    print('Done')
    print_config(config, ['model', 'io', 'training'])

    # read the couplet data
    print('Reading the couplet training data')
    couplet_reader = CoupletReader(
        input_file=config.get('io', 'input_file'),
        output_file=config.get('io', 'output_file'),
        vocab_file=config.get('io', 'vocab_file'),
        max_len=config.getint('model', 'max_len'),
        max_char=config.getint('model', 'max_char')
    )
    print('Done')

    # save the vocabulary file
    print('Saving the vocabulary to file')
    char2idx_path = config.get('io', 'char2idx_path')
    idx2char_path = config.get('io', 'idx2char_path')
    char2idx = couplet_reader.char2idx
    idx2char = couplet_reader.idx2char
    pickle.dump(char2idx, open(char2idx_path, "wb" ))
    print('char2idx saved to {}'.format(char2idx_path))
    pickle.dump(idx2char, open(idx2char_path, "wb" ))
    print('idx2char saved to {}'.format(idx2char_path))

    # set up the parameters for the model
    param_dict = {
        'vocab_size'   :config.getint('model', 'max_char'),
        'embedding_dim':config.getint('model', 'embedding_dim'),
        'units'        :config.getint('model', 'units'),
        'num_layers'   :config.getint('model', 'num_layers'),
        'dropout'      :config.getfloat('model', 'dropout'),
    }

    # create the encoder-decoder model
    print('Creating the encoder-decoder model')
    model = Model(char2idx, idx2char, param_dict)
    print('Done')

    # train or load a pre-trained word2vec model
    if config.getboolean('training', 'word2vec_pretrained'):
        print('Loading the word2vec model')
        model.load_word2vec(word2vec_path=config.get('io', 'word2vec_path'))
        print('word2vec model loaded')
    else:
        print('Training the word2vec model')
        model.train_word2vec(
            train_data=couplet_reader.data_padded + couplet_reader.target_padded,
            iter=100,
            word2vec_path=config.get('io', 'word2vec_path')
        )
        print('word2vec model trained')

    # transfer the word2vec weights into the embedding matrix
    print('Transferring word2vec model weights to the embedding matrix')
    model.transfer_embedding_weights(couplet_reader.idx2char)
    print('Done')

    # load pretrained model if needed
    if config.getboolean('training', 'train_from_scratch'):
        print('Training from scratch')
    else:
        print('Load pretrained model weights')
        model.load_weights(checkpoint_dir=config.get('io', 'model_weights_dir'))
        print('Done')

    # train the model
    print('Start training')
    model.train(
        train=couplet_reader.data_encoded,
        target=couplet_reader.target_encoded,
        start_epoch=config.getint('training', 'start_epoch'),
        num_epoch=config.getint('training', 'num_epoch'),
        log_dir=config.get('io', 'training_checkpoints_dir'),
        checkpoint_dir=config.get('io', 'training_checkpoints_dir'),
        batch_size=config.getint('training', 'batch_size'),
        learning_rate=config.getfloat('training', 'learning_rate')
    )
    print('Done')

    print('Saving training weights')
    model.save_weights(checkpoint_dir=config.get('io', 'model_weights_dir'))

    print('Saving config file')
    with open('{}/config.ini'.format(config.get('io', 'model_weights_dir')), 'w') as configfile:
        config.remove_section('io')
        config.remove_section('training')
        config.write(configfile)

    print('Done!')
