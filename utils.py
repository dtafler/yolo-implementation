import numpy as np
import cv2
import keras.backend as K
import warnings
import tensorflow as tf


def draw_bb(img, coords, color=(0,0,255)):
    """
    Draws bounding boxes
    
    Parameters:
        img: images to draw to
        coords: list with bounding box coordinates [(label,x,y,w,h), ...]
        color: color of the bounding boxes in RGB

    Returns:
        image with bounding boxes drawn in
    """
    for coord in coords:
        assert(len(coord) == 5)
        x_center = coord[1] * img.shape[1]
        y_center = coord[2] * img.shape[0]
        x1 = int(x_center - (0.5 * (coord[3] * img.shape[1])))
        y1 = int(y_center - (0.5 * (coord[4] * img.shape[0])))
        x2 = int(x_center + (0.5 * (coord[3] * img.shape[1])))
        y2 = int(y_center + (0.5 * (coord[4] * img.shape[0])))

        # image = cv2.rectangle(image, start_point, end_point, color, thickness)   
        img = cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        
    return img


def compute_iou(boxes_preds, boxes_labels): 
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    Returns:
        tensor: Intersection over union for all examples
    """

    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = K.maximum(box1_x1, box2_x1)
    y1 = K.maximum(box1_y1, box2_y1)
    x2 = K.minimum(box1_x2, box2_x2)
    y2 = K.minimum(box1_y2, box2_y2)

    # clip is for the case when they do not intersect
    intersection = K.clip((x2 - x1), 0, 9999) * K.clip((y2 - y1), 0, 9999)

    box1_area = K.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = K.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def preprocess_label(label_path, grid_len, num_classes, num_boxes):
    """
    Converts labels into yolo format

    Parameters:
        label_path: pathlib Path to a label
        grid_len: length of any side of the square grid
        num_classes: number of classes 
        num_boxes: number of boxes per cell

    Returns:
        numpy array (grid_len, grid_len, num_boxes * 5  + num_classes)
    """
    with label_path.open() as f:  # read label.txt (shape, x, y, w, h)
        labels = [line.replace(" \n", "").split(" ") for line in f if line != '\n']

    truth_matrix = np.zeros([grid_len, grid_len, num_boxes * 5  + num_classes]) 
    
    locs = []
    # translate each bounding box to grid system
    for l in labels:
        # get coordinates
        clss = int(l[0])
        x = float(l[1])
        y = float(l[2])
        w = float(l[3])
        h = float(l[4])

        # scale location to grid
        loc = [grid_len * x, grid_len * y] 
        # truncate (effectively round down) to get grid cell
        loc_i = int(loc[1]) 
        loc_j = int(loc[0])
        # get location relative to grid cell
        y = loc[1] - loc_i 
        x = loc[0] - loc_j

        # to check for duplicates (if duplicates exist maybe higher grid resolution)
        locs.append((loc_i, loc_j))
        
        # enter values in truth matrix
        if truth_matrix[loc_i, loc_j, 0] == 0:
            if num_classes > 0:
                truth_matrix[loc_i, loc_j, clss + num_boxes * 5] = 1
            truth_matrix[loc_i, loc_j, 1:5] = [x, y, w, h]
            truth_matrix[loc_i, loc_j, 0] = 1  # confidence 
    
    duplicates = [elem for elem in locs if locs.count(elem) > 1]
    if len(duplicates) != 0:
        warnings.warn("WARNING: multiple elements in same grid cell, grid resolution might be too small")

    return truth_matrix


def get_best_bb(cell_vec, num_boxes):
    """
    Gets every num_boxes element and return argmax:
    """
    assert(cell_vec.ndim == 1)

    # get every num_boxes element 
    confidences = []
    for bb in range(num_boxes):
        confidences.append(cell_vec[bb * 5])

    assert(len(confidences) == num_boxes)
    assert(confidences[0] == cell_vec[0])
    
    return np.argmax(confidences)
    
def get_bb_coords(label, grid_len, num_classes, num_boxes, confidence_cutoff, print_best_box=False, print_cxywh=False, print_scaled_xy=False, print_coord=False):
    """
    Retrieves bounding box coordinates from an output matrix. 

    Parameters: 
        label: yolo output or preprocessed label
        grid_len: length of any side of the square grid
        num_classes: number of class labels
        num_boxes: number of boxes (5*num_boxes before class labels begin)
        confidence_cutoff: bbs with lower values are ignored
        print_cxywh: boolean to print confidence, x,y,w,h (network output) debugging purposes
        print_scaled_xy: boolean to print x, y (scaled to image) for debugging purposes
        print_coord: boolean to print the label,x,y,w,h as returned by this function for each cell
        print_best_box: boolean to print best_box for debugging purposes
    
    Returns:
        list with bounding box coordinates [(label,x,y,w,h), ...]
    """
    coords = []
    for i in range(grid_len):
        for j in range(grid_len):
            # get the coordinates if confidence above threshold
            if label[i,j,0] >= confidence_cutoff:
                # get index of best bounding box
                best_box = get_best_bb(label[i,j,:], num_boxes)
                assert(best_box >= 0)
                assert(best_box < num_boxes)
                if print_best_box:
                    print(f'\nbest_box: {best_box}')

                # print predicted values
                if print_cxywh:
                    start_idx = best_box * 5
                    print(f'conf, x,y,w,h of cell {i},{j}: {label[i,j, start_idx : start_idx+5]}')

                # scale x,y coordinates (w,h are already in scale)
                x = (label[i,j, 1 + best_box * 5] + j) / grid_len
                y = (label[i,j, 2 + best_box * 5] + i) / grid_len

                # print scaled values
                if print_scaled_xy:
                    print(f'(scaled) x, y of cell {i},{j}: {x}, {y}')

                # get class label
                if num_classes > 0:
                    clss = np.argmax(label[i, j, 5*num_boxes:])
                
                else:
                    clss = 1

                coord = (clss, x, y, label[i,j, (best_box*5) + 3], label[i,j, (best_box*5) + 4])
                if print_coord:
                    print(f'label,x,y,w,h of cell {i},{j}: {coord}')

                coords.append(coord)
    return coords

def get_ltrb_coords(label, grid_len, num_classes, num_boxes, img_size, class_label=1, ground_truth=False, print_best_box=False, print_cxywh=False, print_scaled_xy=False, print_coord=False):
    """
    Retrieves format bounding box coordinates from an output matrix, from which mAP can be calculated. 

    Parameters: 
        label: yolo output or preprocessed label
        grid_len: length of any side of the square grid
        num_classes: number of class labels
        num_boxes: number of boxes (5*num_boxes before class labels begin)
        img_size: tuple of height and width of image
        class_label: label of the class if there is only one class
        ground_truth: if label is ground truth the confidence is not returned
        print_cxywh: boolean to print confidence, x,y,w,h (network output) debugging purposes
        print_scaled_xy: boolean to print x, y (scaled to image) for debugging purposes
        print_coord: boolean to print the label,x,y,w,h as returned by this function for each cell
        print_best_box: boolean to print best_box for debugging purposes

    
    Returns:
        list with bounding box coordinates [[clss, (confidence), left, top, right, bottom],...]
        (values are absolute pixel values)
    """
    img_h = img_size[0]
    img_w = img_size[1]
    
    coords = []
    for i in range(grid_len):
        for j in range(grid_len):
            # get the coordinates and confidence
            
            # get index of best bounding box
            best_box = get_best_bb(label[i,j,:], num_boxes)
            assert(best_box >= 0)
            assert(best_box < num_boxes)
            if print_best_box:
                print(f'\nbest_box: {best_box}')

            # print predicted values
            if print_cxywh:
                start_idx = best_box * 5
                print(f'conf, x,y,w,h of cell {i},{j}: {label[i,j, start_idx : start_idx+5]}')

            # get coordinates relative to image
            x = (label[i,j, 1 + best_box * 5] + j) / grid_len
            y = (label[i,j, 2 + best_box * 5] + i) / grid_len
            w = label[i,j, 3 + best_box * 5]
            h = label[i,j, 4 + best_box * 5]
            confidence = label[i,j, best_box * 5]
            # x,y,w,h are now relative to the image size
            # print scaled values
            if print_scaled_xy:
                print(f'(scaled) x, y of cell {i},{j}: {x}, {y}')
            
            # convert x,y,w,h to absolute pixel values
            x = x * img_w
            y = y * img_h
            w = w * img_w
            h = h * img_h

            # calculate coordinates
            left = int(x - (0.5 * w))
            top = int(y - (0.5 * h))
            right = int(x + (0.5 * w))
            bottom = int(y + (0.5 * h))

            # get class label
            if num_classes > 0:
                clss = np.argmax(label[i, j, 5*num_boxes:])
            
            else:
                clss = class_label
            
            if ground_truth:
                coord = [clss, left, top, right, bottom]
                if confidence == 1:
                    coords.append(coord)

            else:
                coord = [clss, confidence, left, top, right, bottom]
                coords.append(coord)
                
            if print_coord:
                print(f'clss, (conf), left, top, right, bottom of cell {i},{j}: {coord}')

    return coords



def get_boxes_and_scores(label, grid_len, num_boxes, img_size, print_best_box=False, print_cxywh=False, print_scaled_xy=False, print_coord=False):
    """
    Retrieves box coordinates (y,x,y,x) and confidences, can be used for non-max-suppression calculation. 

    Parameters: 
        label: yolo output or preprocessed label
        grid_len: length of any side of the square grid
        num_boxes: number of boxes (5*num_boxes before class labels begin)
        img_size: tuple of height and width of image
        print_cxywh: boolean to print confidence, x,y,w,h (network output) debugging purposes
        print_scaled_xy: boolean to print x, y (scaled to image) for debugging purposes
        print_coord: boolean to print the label,x,y,w,h as returned by this function for each cell
        print_best_box: boolean to print best_box for debugging purposes

    
    Returns:
        boxes: 2D Tensor of shape (number of predicted bounding boxes, 4)
        scores: 1D Tensor of shape (number of predicted bounding boxes)
    """
    img_h = img_size[0]
    img_w = img_size[1]
    
    boxes_list = []
    scores_list = []
    for i in range(grid_len):
        for j in range(grid_len):
            # get the coordinates and confidence
            
            # get index of best bounding box
            best_box = get_best_bb(label[i,j,:], num_boxes)
            assert(best_box >= 0)
            assert(best_box < num_boxes)
            if print_best_box:
                print(f'\nbest_box: {best_box}')

            # print predicted values
            if print_cxywh:
                start_idx = best_box * 5
                print(f'conf, x,y,w,h of cell {i},{j}: {label[i,j, start_idx : start_idx+5]}')

            # get coordinates relative to image
            x = (label[i,j, 1 + best_box * 5] + j) / grid_len
            y = (label[i,j, 2 + best_box * 5] + i) / grid_len
            w = label[i,j, 3 + best_box * 5]
            h = label[i,j, 4 + best_box * 5]
            confidence = label[i,j, best_box * 5]
            # x,y,w,h are now relative to the image size
            # print scaled values
            if print_scaled_xy:
                print(f'(scaled) x, y of cell {i},{j}: {x}, {y}')
            

            # calculate coordinates
            x_min = x - (0.5 * w)
            y_min = y - (0.5 * h)
            x_max = x + (0.5 * w)
            y_max = y + (0.5 * h)

            coord = [y_min, x_min, y_max, x_max]
            boxes_list.append(coord)

            scores_list.append(confidence)
                
            if print_coord:
                print(f'clss, (conf), left, top, right, bottom of cell {i},{j}: {coord}')
    
    boxes = tf.convert_to_tensor(boxes_list, dtype=tf.float32)
    scores = tf.convert_to_tensor(scores_list, dtype=tf.float32)
    # print(boxes.shape)
    # print(scores.shape)
    # print(len(boxes_list))
    # print(len(scores_list))
    return boxes, scores


def boxes_to_img(boxes, img):
    boxes = boxes.numpy()

    num_boxes = boxes.shape[0]

    for box in range(num_boxes):
        x_min = int(boxes[box, 1] * img.shape[1])
        y_min = int(boxes[box, 0] * img.shape[0])
        x_max = int(boxes[box, 3] * img.shape[1])
        y_max = int(boxes[box, 2] * img.shape[0])

        img = cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0,0,255), 2)
    
    return img
