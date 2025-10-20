import numpy as np
import tensorflow.keras.backend as K

def standardise_sample(sample):

    return (sample - sample.min()) / ((sample.max() - sample.min()) + K.epsilon())


def normalise_sample(sample, mean, stddev):

    sample -= mean
    sample /= stddev + K.epsilon()

    return sample

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.shape[0]
    #print(batch_size)
    out = np.zeros(shape=(batch_size, dim))
    # print(out.shape)
    # print(labels)
    out[np.arange(batch_size), labels.astype(np.int)] = 1
    # print(out)
    return out.astype(np.int)