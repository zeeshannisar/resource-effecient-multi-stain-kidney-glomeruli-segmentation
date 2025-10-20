import numpy as np
import tensorflow as tf

cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)


def cosine_sim_dim1(x, y):
    v = cosine_sim_1d(x, y)
    return v


def cosine_sim_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return v


def dot_sim_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def dot_sim_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v


def get_negative_mask(batch_size):
    # Return a mask that removes the similarity score of equal/similar images.
    # This function ensures that only distinct pair of images get their similarity scores passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


def LossFunction1(z_i, z_j, T=1):
    """ Available at https://github.com/sthalles/SimCLR-tensorflow"""
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    batch_size = z_i.shape[0]

    # Normalize projection feature vectors
    z_i = tf.math.l2_normalize(x=z_i, axis=1)
    z_j = tf.math.l2_normalize(x=z_j, axis=1)

    loss_positives = dot_sim_dim1(x=z_i, y=z_j)
    loss_positives = tf.reshape(loss_positives, (batch_size, 1))
    loss_positives = loss_positives / T

    negatives = tf.concat([z_j, z_i], axis=0)

    loss = 0

    for positives in [z_i, z_j]:
        loss_negatives = dot_sim_dim2(x=positives, y=negatives)
        labels = tf.zeros(batch_size, dtype=tf.int32)

        loss_negatives = tf.boolean_mask(loss_negatives, get_negative_mask(batch_size))
        loss_negatives = tf.reshape(loss_negatives, (batch_size, -1))
        loss_negatives = loss_negatives / T

        logits = tf.concat([loss_positives, loss_negatives], axis=1)
        loss = loss + criterion(y_pred=logits, y_true=labels)

    loss = loss / (2 * batch_size)
    return loss


def LossFunction2(z_i, z_j, T=1):
    """
    Available at https://stackoverflow.com/questions/62793043/tensorflow-implementation-of-nt-xent-contrastive-loss-function
    Calculates the contrastive loss of the input data using NT_Xent. The equation can be found in the
    paper: https://arxiv.org/pdf/2002.05709.pdf. This is the Tensorflow implementation of the standard numpy version
    found in the NT_Xent function).

    Args:
        z_i: One half of the input data, shape = (batch_size, feature_1, feature_2, ..., feature_N)
        z_j: Other half of the input data, must have the same shape as z_i
        T: Temperature parameter (a constant), default = 1.

    Returns:
        loss: The complete NT_Xent Constrastive Loss
    """
    z = tf.cast(tf.concat((z_i, z_j), 0), dtype=tf.float32)
    loss = 0
    for k in range(z_i.shape[0]):
        # Numerator (compare i,j & j,i)
        i = k
        j = k + z_i.shape[0]
        # Instantiate the cosine similarity loss function
        cosine_sim = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        sim = tf.squeeze(- cosine_sim(tf.reshape(z[i], (1, -1)), tf.reshape(z[j], (1, -1))))
        numerator = tf.math.exp(sim / T)

        # Denominator (compare i & j to all samples apart from themselves)
        sim_ik = - cosine_sim(tf.reshape(z[i], (1, -1)), z[tf.range(z.shape[0]) != i])
        sim_jk = - cosine_sim(tf.reshape(z[j], (1, -1)), z[tf.range(z.shape[0]) != j])
        denominator_ik = tf.reduce_sum(tf.math.exp(sim_ik / T))
        denominator_jk = tf.reduce_sum(tf.math.exp(sim_jk / T))

        # Calculate individual and combined losses
        loss_ij = - tf.math.log(numerator / denominator_ik)
        loss_ji = - tf.math.log(numerator / denominator_jk)
        loss += loss_ij + loss_ji

    # Divide by the total number of samples
    loss /= z.shape[0]

    return loss
