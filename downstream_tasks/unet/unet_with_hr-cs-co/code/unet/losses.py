import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


# Experimental
def weighted_sparse_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.sparse_categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        y_true = to_categorical(y_true)

        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def dice():

    def loss(y_true, y_pred):

        dice_nominator = 2.0 * K.sum(y_true * y_pred, axis=-1)
        dice_denominator = K.sum(y_true + y_pred, axis=-1)

        dice_score = (dice_nominator + 1) / (dice_denominator + 1)

        return 1.0 - dice_score

    return loss


def dice_square():

    def loss(y_true, y_pred):

        dice_nominator = 2.0 * K.sum(y_true * y_pred, axis=-1)
        dice_denominator = K.sum(K.square(y_true) + K.square(y_pred), axis=-1)

        dice_score = (dice_nominator + 1) / (dice_denominator + 1)

        return 1.0 - dice_score

    return loss


def jaccard():

    def loss(y_true, y_pred):

        intersection = K.sum(y_true * y_pred, axis=-1)

        sum_ = K.sum(y_true + y_pred, axis=-1)

        epsilon_denominator = 0.00001

        jaccard = intersection / (sum_ - intersection + epsilon_denominator)

        return 1.0 - jaccard

    return loss


def jaccard_mean():

    def iou(y_true, y_pred, label):
        """
        Return the Intersection over Union (IoU) for a given label.
        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or softmax output
            label: the label to return the IoU for
        Returns:
            the IoU for the given label
        """
        # extract the label values using the argmax operator then
        # calculate equality of the predictions and truths to the label
        y_true = K.cast(y_true[..., label], K.floatx())
        y_pred = K.cast(y_pred[..., label], K.floatx())
        # calculate the |intersection| (AND) of the labels
        intersection = K.sum(y_true * y_pred)
        # calculate the |union| (OR) of the labels
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        # avoid divide by zero - if the union is zero, return 1
        # otherwise, return the intersection over union
        return 1 - ((intersection+1) / (union+1))
#        return K.switch(K.equal(union, 0), 1.0, intersection / union)

    def loss(y_true, y_pred):
        # get number of labels to calculate IoU for
        num_labels = K.int_shape(y_pred)[-1]
        # initialize a variable to store total IoU in
        total_iou = K.variable(0)
        # iterate over labels to calculate IoU for
        for label in range(num_labels):
            total_iou = total_iou + iou(y_true, y_pred, label)
        # divide total IoU by number of labels to get mean IoU
        return total_iou / num_labels

    return loss


def tversky(alpha=0.3, beta=0.7):

    def loss(y_true, y_pred):

        epsilon_denominator = 0.00001

        intersection = K.sum(y_true * y_pred, axis=-1)

        fp_and_fn = (alpha * K.sum(y_pred * (1 - y_true), axis=-1)) + (
                    beta * K.sum((1 - y_pred) * y_true, axis=-1))

        tversky = intersection / (intersection + fp_and_fn + epsilon_denominator)

        return 1.0 - tversky

    return loss


def border_loss():

    def loss(y_true, y_pred):

        return - tf.reduce_sum(y_true * y_pred, len(y_pred.get_shape()) - 1)

    return loss