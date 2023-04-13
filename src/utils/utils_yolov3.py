import modules.networks.yolo_v3.models as yolo_model
import numpy as np
import torch
import os
import torch.nn.functional as F


def make_save_dir(output_image_dir):
    if not os.path.isdir(output_image_dir):
        os.makedirs(output_image_dir)


def overlap(a, b):  # returns None if rectangles don't intersect
    from collections import namedtuple
    Rectangle = namedtuple('Rectangle', 'ymin xmin ymax xmax')
    height = float(a[2] - a[0] + 1)
    width = float(a[3] - a[1] + 1)
    a = Rectangle(a[0], a[1], a[2], a[3])
    b = Rectangle(b[0], b[1], b[2], b[3])
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        if dx * dy > 0.50 * height * width:
            return True, dx * dy
        else:
            return False, None
    else:
        return False, None


def find_best_detection(bounding_boxes, detections_gt, scale_factor, h, w):
    good_detections = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections_gt:
        box_w = x2 - x1
        box_h = y2 - y1
        if not np.all(np.array([x1, y1, x2, y2]) > 0):
            continue
        is_overlap, area_overlap = \
            overlap([int(bounding_boxes[2] * scale_factor), int(bounding_boxes[0] * scale_factor),
                     int(bounding_boxes[3] * scale_factor), int(bounding_boxes[1] * scale_factor)],
                    [int(y1), int(x1), int(y2), int(x2)])
        if not is_overlap or (box_h * box_w) < (h * w * 0.01):
            continue
        else:
            good_detections.append([[int(y1), int(x1), int(y2), int(x2), conf, cls_conf, cls_pred], area_overlap])
    if len(good_detections) > 0:
        return max(good_detections, key=lambda x: x[1])[0]
    else:
        return None


def save_image(image, bbox, paths, root):
    import cv2
    import os
    y1, x1, y2, x2 = bbox

    start_point = (x1, y1)
    end_point = (x2, y2)
    color = (0, 0, 0)
    image = cv2.rectangle(cv2.cvtColor(image.permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_RGB2BGR),
                          start_point, end_point, color, 2)
    img_dir = os.path.join(root, paths)
    make_save_dir("/".join(img_dir.split("/")[:-1]))
    cv2.imwrite(img_dir, image)


def compute_detection(image_gt, image_predicted, model_yolo, tracking_gnn, classes, device, index_user_guidance,
                      paths, root_gt, root_pred):
    scale_factor = 1 if image_gt.shape[4] in [256, 320] else 2
    mse_batch = []
    mse_batch_normalized = []
    image_gt = image_gt[:, :, -1, ...]
    image_pred = image_predicted[:, :, -1, ...]
    image_gt = image_gt.to(device)
    image_pred = image_pred.to(device)
    if scale_factor == 2:
        image_gt = F.interpolate(image_gt, scale_factor=scale_factor, mode="nearest")

        image_pred = F.interpolate(image_pred, scale_factor=scale_factor, mode="nearest")
    b, c, h, w = image_gt.shape
    image_gt = \
        F.pad(input=image_gt, pad=(0, 416 - image_gt.shape[-1], 0, 416 - image_gt.shape[-2]), mode="constant", value=0)

    image_pred = F.pad(input=image_pred, pad=(0, 416 - image_pred.shape[-1], 0, 416 - image_pred.shape[-2]),
                       mode="constant", value=0)
    gt_detected_images = []
    pred_detected_images = []
    for idx in index_user_guidance:
        inst_id, batch_id, bounding_boxes, x_n = tracking_gnn.source_frames_nodes_instance_ids[idx][-1].long(),\
                                                 tracking_gnn.batch[idx].long(),\
                                                 tracking_gnn.target_frames_nodes_roi[idx][-1], tracking_gnn.x[idx][-1]
        with torch.no_grad():
            detections_gt = model_yolo(image_gt[batch_id].unsqueeze(0))
            detections_gt = \
                yolo_model.non_max_suppression(detections_gt, 0.50,
                                               0.4)[0]
        # Draw bounding boxes and labels of detections
        gt_y_baricenter = (int(bounding_boxes[2] * scale_factor) + int(bounding_boxes[3] * scale_factor)) / 2
        gt_x_baricenter = (int(bounding_boxes[0] * scale_factor) + int(bounding_boxes[1] * scale_factor)) / 2
        gt_start_y_baricenter, gt_start_x_baricenter = (x_n[:2] + 1) / 2 * torch.LongTensor([h, w]).cuda()
        gt_start_y_baricenter = int(gt_start_y_baricenter)
        gt_start_x_baricenter = int(gt_start_x_baricenter)
        bbox_width = int(bounding_boxes[1] * scale_factor) - int(bounding_boxes[0] * scale_factor)
        bbox_height = int(bounding_boxes[3] * scale_factor) - int(bounding_boxes[2] * scale_factor)
        if bbox_height * bbox_width < 0.005 * w * h:
            continue
        if detections_gt is not None:

            gt_detection = find_best_detection(bounding_boxes, detections_gt, scale_factor, h, w)
            if gt_detection is not None:
                gt_detected_images.append(1)
                y1, x1, y2, x2, conf, cls_conf, cls_pred = gt_detection
                with torch.no_grad():

                    detections_pred = model_yolo(image_pred[batch_id].unsqueeze(0))
                    detections_pred = \
                        yolo_model.non_max_suppression(detections_pred, 0.50,
                                                       0.4)[0]
                    if detections_pred is not None:
                        pred_detection = find_best_detection(bounding_boxes, detections_pred, scale_factor, h, w)
                        if pred_detection is not None:
                            pred_detected_images.append(1)
                            y1_fake, x1_fake, y2_fake, x2_fake, conf_fake, cls_conf_fake, cls_pred_fake = pred_detection
                            print("\t+ Label: %s, Conf: %.5f" % (
                                classes[int(cls_pred)], cls_conf.item()))
                            box_w = x2_fake - x1_fake
                            box_h = y2_fake - y1_fake
                            pred_y_baricenter = (int(y1_fake) + int(y2_fake)) / 2
                            pred_x_baricenter = (int(x1_fake) + int(x2_fake)) / 2
                            mse = np.sqrt((pred_y_baricenter - gt_y_baricenter)**2 +
                                          (pred_x_baricenter - gt_x_baricenter)**2)
                            normalization_factor = np.sqrt((gt_start_y_baricenter - gt_y_baricenter)**2 +
                                                           (gt_start_x_baricenter - gt_x_baricenter)**2)
                            normalization_factor = normalization_factor if normalization_factor > 0 else 1

                            mse_normalized = mse / (normalization_factor + 1e-06)
                            mse_batch.append(mse)
                            mse_batch_normalized.append(mse_normalized)
                            save_image(image_gt[batch_id], [y1, x1, y2, x2],
                                       paths[-1][batch_id][0:-4] + f'_{inst_id}_yolo_detector_gt.png', root_gt)
                            save_image(image_pred[batch_id], [y1_fake, x1_fake, y2_fake, x2_fake],
                                       paths[-1][batch_id][0:-4] + f'_{inst_id}_yolo_detector_pred.png', root_pred)
                            print(f"batch: {batch_id} inst_id: {inst_id}"
                                  f" bouding_box_gt: "
                                  f"{[int(bounding_boxes[2] * scale_factor), int(bounding_boxes[0] * scale_factor), int(bounding_boxes[3] * scale_factor), int(bounding_boxes[1] * scale_factor)]} bounding_box_pred: {[int(y1_fake), int(x1_fake), int(y2_fake), int(x2_fake)]} mse: {mse} , mse_normalized: {mse_normalized}, bb_size:{box_w, box_h}")
                        else:
                            box_w = x2 - x1
                            box_h = y2 - y1

                            save_image(image_gt[batch_id], [y1, x1, y2, x2],
                                       paths[-1][batch_id][0:-4] + f'_{inst_id}_yolo_detector_gt.png', root_gt)

                            print("\t+ Label: %s, Conf: %.5f" % (
                                classes[int(cls_pred)], cls_conf.item()))
                            print(f"NOT FOUND batch: {batch_id} inst_id: {inst_id} bouding_box_gt: {[int(bounding_boxes[2] * scale_factor), int(bounding_boxes[0] * scale_factor), int(bounding_boxes[3] * scale_factor), int(bounding_boxes[1] * scale_factor)]} bb_size:{box_w, box_h}")
                    else:
                        box_w = x2 - x1
                        box_h = y2 - y1

                        save_image(image_gt[batch_id], [y1, x1, y2, x2],
                                   paths[-1][batch_id][0:-4] + f'_{inst_id}_yolo_detector_gt.png', root_gt)

                        print("\t+ Label: %s, Conf: %.5f" % (
                            classes[int(cls_pred)], cls_conf.item()))
                        print(
                            f"NOT FOUND batch: {batch_id} inst_id: {inst_id} bouding_box_gt: {[int(bounding_boxes[2] * scale_factor), int(bounding_boxes[0] * scale_factor), int(bounding_boxes[3] * scale_factor), int(bounding_boxes[1] * scale_factor)]} bb_size:{box_w, box_h}")

                # Create a Rectangle patch
    return mse_batch, mse_batch_normalized, gt_detected_images, pred_detected_images
