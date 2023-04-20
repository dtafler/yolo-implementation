import tensorflow as tf
import keras
import keras.backend as K
from utils import compute_iou

from config import GRID_LEN, NUM_BOXES, NUM_CLASSES

# currently only work with NUM_BOXES == 2

class YoloLoss(keras.losses.Loss):
    def __init__(self, name="yolo_loss", **kwargs):
        super().__init__(name=name, **kwargs)


    def call(self, y_true, y_pred): 
        assert(NUM_BOXES == 2)
        # shape of y_true: batch-size, GRID_SIZE, GRID_SIZE, NUM_BOXES * 5 + NUM_CLASSES
        # e.g. if NUM_BOXES is 2 and there are no classes: [conf, x, y, w, h, conf, x, y, w, h]
        
        #TODO: anchor box with shape of car as we know from data (cars are very similar in size)
        # get y_pred into same format as y_true:
        y_pred = tf.cast(K.reshape(y_pred, (-1, GRID_LEN, GRID_LEN, 5 * NUM_BOXES + NUM_CLASSES)), 
                         dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)
        # tf.print(f'LOSS: y_pred.shape: {y_pred.shape}')
        # tf.print(f'LOSS: y_true.shape: {y_true.shape}')


        # compute ious (each iou of shape [batchsize, gridsize, gridsize, 1], one iou for each cell):
        iou_bb1 = K.expand_dims(compute_iou(y_pred[..., 1:5], y_true[..., 1:5]), axis=0)
        iou_bb2 = K.expand_dims(compute_iou(y_pred[..., 6:10], y_true[..., 1:5]), axis=0)

        # shape:  [2, batchsize, gridsize, gridsize, 1]
        ious = K.concatenate([iou_bb1, iou_bb2], axis=0) 

        # bestbox: box that is responsible for a given cell [batchsize, gridsize, gridsize, 1]:
        bestbox = K.cast(K.argmax(ious, axis=0), dtype=tf.float32) 

        # best_ious: ious of responsible boxes (max ious)
        best_ious = tf.where(
            K.cast(bestbox, tf.bool),
            K.reshape(iou_bb2, (-1, GRID_LEN, GRID_LEN, 1)),
            K.reshape(iou_bb1, (-1, GRID_LEN, GRID_LEN, 1))
        )

        # exists_object: for each cell in every batch, does there exist an object? 
        # shape: [batchsize, gridsize, gridsize, 1]
        exists_object = K.expand_dims(y_true[..., 0], axis=3)

        ################
        ### box loss ###
        ################
        # batchsize = y_pred.shape[0]
        # box_predictions = tf.Variable(tf.zeros((batchsize, GRID_SIZE, GRID_SIZE, 4)))
        # box_targets = tf.Variable(tf.zeros((batchsize, GRID_SIZE, GRID_SIZE, 4)))

        # if an object exists, use predictions of best box:
        xy_pred = (bestbox * y_pred[..., 6:8]) + ((1 - bestbox) * y_pred[..., 1:3])
        box_predictions_xy = (exists_object * xy_pred)
        box_targets_xy = (exists_object * y_true[..., 1:3])

        # square-root of width and height(same change is less important in larger box):
        wh_pred = ((bestbox * y_pred[..., 8:10]) + (1 - bestbox) * y_pred[..., 3:5])

        # derivative of squareroot as you go to zero is infinity, so add 1e-6 for numerical stability
        box_predictions_wh = (K.sign(exists_object * wh_pred) * 
                              K.sqrt(K.abs(exists_object * wh_pred)+ 1e-6)) 
        box_targets_wh = (K.sqrt(exists_object * y_true[..., 3:5])) 
  
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        box_loss = mse(box_predictions_xy, box_targets_xy) + mse(box_predictions_wh, box_targets_wh)

        ###################
        ### object loss ###
        ###################
        confidence = (bestbox * y_pred[..., 5:6]) + ((1 - bestbox) * y_pred[..., 0:1])
        # object_loss = mse((exists_box * confidence), (exists_box * y_true[..., 0:1])) 

        object_loss = mse((exists_object * confidence), 
                          (exists_object * best_ious * y_true[..., 0:1])) 
        # tf.print(f'LOSS:object_loss: {object_loss}')

        ######################
        ### no object loss ###
        ######################
        # confidence should be 0 when no object is present
        no_object_loss = mse(((1 - exists_object) * confidence), 
                             ((1 - exists_object) * y_true[..., 0:1])) # this term is all zeros
        # tf.print(f'LOSS:no_object_loss: {no_object_loss}')


        ##################
        ### class loss ###
        ##################
        if NUM_CLASSES > 0:
            class_loss = mse((exists_object * y_pred[..., 10:]), (exists_object * y_true[..., 10:]))
        # try categorical crossentropy: 
        # cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        # class_loss = cce((exists_box * y_pred[..., 10:]), (exists_box * y_true[..., 10:]))
        
        # tf.print(f'LOSS:class_loss: {class_loss}')


        ##################
        ### total loss ###
        ##################
        lambda_coord = 5
        lambda_noobj = 0.5
        loss = (
            lambda_coord * box_loss
            + object_loss
            + lambda_noobj * no_object_loss
            + class_loss
        )
        
        return loss


