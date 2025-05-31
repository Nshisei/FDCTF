import cv2
import numpy as np
import os
import random
from glob import glob
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from setting import *
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
fix_seeds()


class FocalLoss(nn.Module):

    def __init__(self, weight=None,
             gamma=2.5, reduction='mean'):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.reduction = reduction
        self.weight=weight

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction = self.reduction
        )


def make_segment(video_files, train_ratio=0.8):
    train_video_dir = []
    val_video_dir = []
    for dtype in ["slip", "fall"]:
        dirs = [fld for fld in glob(os.path.join(video_files) + "/*") if os.path.isdir(fld) and dtype in os.path.basename(fld)]
        random.shuffle(dirs)
        train_video_dir.extend(dirs[:int(len(dirs) * train_ratio)])
        val_video_dir.extend(dirs[int(len(dirs) * train_ratio):])
    dirs = [fld for fld in glob(os.path.join(video_files) + "/*") if os.path.isdir(fld) and "slip" not in os.path.basename(fld) and "fall" not in os.path.basename(fld)]
    random.shuffle(dirs)
    train_video_dir.extend(dirs[:int(len(dirs) * train_ratio)])
    val_video_dir.extend(dirs[int(len(dirs) * train_ratio):])
    return train_video_dir, val_video_dir

from ultralytics import YOLO
yolov8 = YOLO(BEST0804BR)
# yolov8 = YOLO(BEST0804)
# anno_df = pd.read_csv(ANNTATION5)

########################################################################################################################
### 画像前処理
########################################################################################################################

def mask_img(img, mask, br_alpha=1.4, br_beta=0, bg_alpha=0.8, bg_beta=0):
    """
    imgのうち、maskされた箇所を指定された値分明るく、それ以外の箇所を暗くする
    """
    # マスクされた箇所（人物の箇所）を明るくする
    brightened_img = cv2.bitwise_and(img, mask)
    brightened_img = cv2.convertScaleAbs(brightened_img, alpha=br_alpha, beta=br_beta)

    # マスクされていない箇所（背景）を暗くする
    background_mask = cv2.bitwise_not(mask)
    darkened_img = cv2.bitwise_and(img, background_mask)
    darkened_img = cv2.convertScaleAbs(darkened_img, alpha=bg_alpha, beta=bg_beta)


    # 前景と背景を合成
    result_img = cv2.add(brightened_img, darkened_img)
    result_img = cv2.applyColorMap(result_img, cv2.COLORMAP_INFERNO) 
    
    return result_img


def brighten_bright_areas(img, percentile=92, increment=30):
    """
    imgの明るい箇所上位(percentile)以上を白塗り, それ以外を黒塗りにする
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = np.percentile(gray, percentile)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    brightened_img = img.copy()
    brightened_img[mask == 255] = 255
    brightened_img[mask != 255] = 0
    return brightened_img


def only_yolo_bbox(img, bboxes, class_probs, head_only=False):
    """
    yoloで検出された箇所のみを抽出する
    input:
        img: (120,160)
        bboxes: list([x,y,w,h])
    output:
        img: (120,160)
    """
    masked_img = np.zeros_like(img)  # 画像と同じ形状の0で埋められた配列を作成
    for bbox, c in zip(bboxes, class_probs):
        if head_only and np.argmax(c) != 0:
            continue
        x, y, w, h = bbox
        x = int(x - w/2) 
        y = int(y - h/2)
        masked_img[y:y+int(h), x:x+int(w), :] = 255 
    return masked_img

def remove_top_edge_objects_and_small_areas(img, min_size=150):
    """
    画像の白い箇所のうち、小さすぎるもの、トップに重なっているものを削除する
    img
    min_size=150
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_size or any(point[0][1] == 0 for point in contour):  # Check if any point touches the top edge
            cv2.drawContours(img, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
    return img

def preprocess(img, track_thresh=0.4, iou=0.7, br=True, percentile=89, br_alpha=1.4, br_beta=0, bg_alpha=0.8, bg_beta=0,device="cuda"):
    """
    imgを明るく加工して、yoloで検出を行う
        1. brighten_bright_areas
        2. remove_top_edge_objects_and_small_areas
        3. mask_img
    return result
    """
    if br:
        processed = brighten_bright_areas(img, percentile=percentile)
        processed = remove_top_edge_objects_and_small_areas(processed)
        masked = mask_img(img, processed, br_alpha=br_alpha, br_beta=br_beta, bg_alpha=bg_alpha, bg_beta=bg_beta)
    return masked

def yolo(img, track_thresh=0.4, iou=0.7, br=True, percentile=89, br_alpha=1.4, br_beta=0, bg_alpha=0.8, bg_beta=0,device="cuda"):
    """
    imgを明るく加工して、yoloで検出を行う
        1. brighten_bright_areas
        2. remove_top_edge_objects_and_small_areas
        3. mask_img
    return result
    """
    if br:
        processed = brighten_bright_areas(img, percentile=percentile)
        processed = remove_top_edge_objects_and_small_areas(processed)
        masked = mask_img(img, processed, br_alpha=br_alpha, br_beta=br_beta, bg_alpha=bg_alpha, bg_beta=bg_beta)
        masked = mask_ceil(masked)
    else:
        masked = mask_ceil(img)
    result = yolov8(masked, cls_prob=True, conf=track_thresh, iou=iou, device=device)[0]
    return result

def optical_flow(prev_gray, img, mag_th=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)  # ガウシアンブラーを適用
    if prev_gray is None:
        prev_gray = np.zeros(gray.shape)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 3, 5, 1.5, 0)
    # オプティカルフローの大きさと角度を計算
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 角度を色相に変換し、大きさを明度に変換
    mask = mag > mag_th
    ang = ang[mask] * 180 / np.pi / 2
    mag = mag[mask]
    prev_gray = gray
    return np.mean(ang),np.std(ang) ,np.mean(mag),np.std(mag) 


def increase_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def decrease_color_temperature(img):
    img = img.astype(np.float32)
    img[:, :, 0] = img[:, :, 0] * 1.25
    img[:, :, 2] = img[:, :, 2] * 1.0
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img

def closing(img,kernel_shape=(3, 3), iterations=10):
    """
    kernel sizeの中でiteration回数増幅→iteration回減小を繰り返す
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return dst

def IoU(a,b):
    """
    """
    def xywh2xyxy(cx,cy,w,h):
        return cx - w/2, cy-h/2, cx + w/2, cy+h/2
    x1, y1, x2, y2 = xywh2xyxy(a[0][0], a[0][1], a[1][0], a[1][1])
    x3, y3, x4, y4 = xywh2xyxy(b[0][0], b[0][1], b[1][0], b[1][1])
    
    # bbox1とbbox2の交差領域の座標を計算
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 重なっていない場合

    # 交差領域の幅と高さを計算
    intersection_width = x_right - x_left
    intersection_height = y_bottom - y_top

    # 交差領域の面積と各bboxの面積を計算
    intersection_area = intersection_width * intersection_height
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)

    # IoUを計算
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

########################################################################################################################
### マスク系
########################################################################################################################

def draw_rotated_rect_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    box = None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # バイナリマスクの初期化
    mask = np.zeros_like(gray)

    for contour in contours:
        if len(contour) < 5:
            continue

        # 回転矩形をフィッティング
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 回転矩形を白く塗りつぶす
        cv2.drawContours(mask, [box], 0, (255), thickness=cv2.FILLED)
    
    return mask, box

def calculate_iou(mask1, mask2):
    # AND演算で交差領域を計算
    intersection = cv2.bitwise_and(mask1, mask2)
    intersection_area = np.sum(intersection == 255)
    # OR演算で和集合領域を計算
    union = cv2.bitwise_or(mask1, mask2)
    union_area = np.sum(union == 255)
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def calc_iou_bed(gray, bed_img_file):
    """
    入力されたgray画像とbedとのIoUを計算する
    input:
        gray: (120,160)の人物だけが白くなった画像
        bed_img_file: 参照するbedの画像
    return iou
    """
    rotated_rect_mask, rect = draw_rotated_rect_mask(gray)
    rect_img = cv2.drawContours(gray, [rect], 0, (0, 255, 0), 2)
    if rotated_rect_mask is not None:
        image_path = MASKIMGDIR + bed_img_file + ".png"
        
        bed_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        bed_mask = cv2.resize(bed_mask, (160,120))
        iou = calculate_iou(bed_mask, rotated_rect_mask)
        return iou
    else:
        return 0

ceil = cv2.imread(MASKIMGDIR + "long_movie_image_light_mask.png")
def mask_ceil(img):
    """
    画像中の天井の明かりを取り除く
    """
    mask = ceil > 0
    # mask_normalized = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    img = img * (1 - mask).astype(np.uint8)
    return img

########################################################################################################################
### Motiton history, Optical flow
########################################################################################################################

def optical_flow(prev_gray, img, mag_th=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)  # ガウシアンブラーを適用
    if prev_gray is None:
        prev_gray = np.zeros(gray.shape)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 3, 5, 1.5, 0)
    # オプティカルフローの大きさと角度を計算
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = mag > mag_th
    ang = ang[mask] * 180 / np.pi / 2
    mag = mag[mask]
    mean_ang =  np.mean(ang) if not np.isnan(np.mean(ang)) else 0
    std_ang =  np.std(ang) if not np.isnan(np.std(ang)) else 0
    mean_mag =  np.mean(mag) if not np.isnan(np.mean(mag)) else 0
    std_mag =  np.std(mag) if not np.isnan(np.std(mag)) else 0
    return mean_ang, std_ang, mean_mag, std_mag, gray

def motion_flow(images):
    # 色合いを定義する（16枚の画像に対応）
    colors = [
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (255, 0, 255), # Magenta
        (0, 255, 255), # Cyan
        (128, 0, 0),   # Maroon
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Navy
        (128, 128, 0), # Olive
        (128, 0, 128), # Purple
        (0, 128, 128), # Teal
        (192, 192, 192), # Silver
        (128, 128, 128), # Gray
        (255, 165, 0),   # Orange
        (75, 0, 130) ,   # Indigo
        (255, 0, 0),   # Red
    ]
    # 各画像を処理
    motion_flow = None
    num_images = len(images)
    
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 最大値を255/16に設定
        scaled_image = (gray / 255) * (255 / num_images)
        
        # 色を適用
        r, g, b = colors[i]
        colored_image = np.stack([scaled_image * b, scaled_image * g, scaled_image * r], axis=-1)
        
        # 最初の画像でmotion_flowを初期化
        if motion_flow is None:
            motion_flow = np.zeros_like(colored_image, dtype=np.float32)
        
        # 追加する箇所だけ0埋めしてから上書き
        mask = np.any(colored_image > 0, axis=-1, keepdims=True)
        motion_flow = np.where(mask, 0, motion_flow)  # 追加する箇所を0埋め
        motion_flow = np.where(mask, colored_image, motion_flow)  # その上に新しい色を上書き
    
    # 最終的に加算した画像を255でクリップして、uint8にキャスト
    motion_flow = np.clip(motion_flow, 0, 255).astype(np.uint8)
    return motion_flow

def calculate_value_ratios(motion_flow_image):
    # グレースケールに変換（元々3チャンネルのBGRなので1チャンネルに戻す）
    gray_image = cv2.cvtColor(motion_flow_image, cv2.COLOR_BGR2GRAY)
    # 0を超える値を抽出
    non_zero_values = gray_image[gray_image > 0]
    # 値ごとの出現頻度を計算
    unique, counts = np.unique(non_zero_values, return_counts=True)
    # 総計を計算
    total = counts.sum()
    # 割合を計算
    ratios = counts / total
    # 値とその割合を辞書に格納して返す
    value_ratios = dict(zip(unique, ratios))
    return value_ratios

def calculate_motion_features(head_area, body_area, prev_head_area, prev_body_area, threshold=5):
        # フレームをグレースケールに変換
        if head_area is None or prev_head_area is None:
            head_mhi_rate = None
        else:
            head_area_gray = cv2.cvtColor(head_area, cv2.COLOR_BGR2GRAY)
            prev_head_gray = cv2.cvtColor(prev_head_area, cv2.COLOR_BGR2GRAY)
            head_diff = cv2.bitwise_and(head_area_gray, cv2.bitwise_not(prev_head_gray))
            cv2.imshow("head_diff", head_diff)
            head_same_mask = cv2.bitwise_and(head_area_gray, prev_head_gray)
            cv2.imshow("head_same", head_same_mask)
            head_mhi_rate = np.sum(head_diff) / np.sum(head_same_mask) * 100
        if body_area is None or prev_body_area is None:
            body_mhi_rate = None
        else:
            body_area_gray = cv2.cvtColor(body_area, cv2.COLOR_BGR2GRAY)
            prev_body_gray = cv2.cvtColor(prev_body_area, cv2.COLOR_BGR2GRAY)
            # MHIを計算（差分を取って動きを検出）
            body_diff = cv2.bitwise_and(body_area_gray, cv2.bitwise_not(prev_body_gray))
            body_same_mask = cv2.bitwise_and(body_area_gray, prev_body_gray)
            cv2.imshow("body_diff", body_diff)
            cv2.imshow("body_same", body_same_mask)
            # MHIレートを計算
            body_mhi_rate = np.sum(body_diff) / np.sum(body_same_mask) * 100

        return head_mhi_rate, body_mhi_rate


########################################################################################################################
### 楕円関係
########################################################################################################################

def draw_ellipse_on_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img, []

    result_img = img.copy()
    ellipse_data = []

    for contour in contours:
        if len(contour) < 5:
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
            _, angle = calculate_ellipse_features(ellipse)
            (center, axes, __) = ellipse
            if not isinstance(angle[0], float):
                print(angle[0])
                continue
            ellipse = (center, axes, angle[0])
            cv2.ellipse(result_img, ellipse, (0, 255, 0), thickness=2)
            ellipse_data.append(ellipse)
        except:
            continue
    return result_img, ellipse_data

def calculate_ellipse_features(ellipse):
    """
    楕円の角度を0°に近い角度と180°に近い角度の2つを返す
    input: ellipse = (center, axes, angle) 
    return: [angle_near_0, angle_near_180]
    """
    (center, axes, angle) = ellipse
    aspect_ratio = min(axes) / max(axes)
    if angle < 90:
        angle = [angle, angle + 180]
    elif angle < 180:
        angle = [angle - 180, angle]
    elif angle < 270:
        angle = [angle - 180, angle]
    else:
        angle = [angle - 360, angle - 180]
    return aspect_ratio, angle

def get_ellipse_angle(img, boxes):
    masked_img = only_yolo_bbox(img, boxes)
    processed = brighten_bright_areas(img, percentile=89)
    processed = cv2.bitwise_and(processed, masked_img)
    processed = remove_top_edge_objects_and_small_areas(processed)
    img, curr_ellipses = draw_ellipse_on_contours(processed)
    if len(curr_ellipses) == 3:
        return curr_ellipses[2]
    else:
        return None
    
def get_largest_box_feature(boxes, class_probs):
    br = None
    largest_box = []
    largest_box_area = 0
    largest_box_cls = np.array([None, None,None,None])
    for cls, box in zip(class_probs, boxes):
        x, y, w, h = box
        if np.argmax(cls) != 0 and largest_box_area < w*h:  # If the class is head
            br = w / h
            largest_box_area = w*h
            largest_box = box
            largest_box_cls = cls

    return br, largest_box, largest_box_cls


########################################################################################################################
### Z, camera 
########################################################################################################################
def distance_from_bbox_area(bbox, real_face_area_m2):
    '''
    顔のバウンディングボックスの面積からz座標を推定します。
    :param bbox: 画像内の顔のバウンディングボックス (x, y, width, height)
    :param real_face_area_m2: 実際の顔の面積（平方メートル）
    :return: 推定されたz座標 (カメラからの距離)
    '''
    horizontal_fov_rad = np.deg2rad(HORIZONTAL_FOV_DEG)  # 水平視野角をラジアンに変換
    focal_length_m = (IMAGE_WIDTH * PIXEL_SIZE_M / 2) / np.tan(HORIZONTAL_FOV_DEG / 2)  # 焦点距離の計算

    x, y, width, height = bbox
    bbox_area_px = width * height  # バウンディングボックスの面積（ピクセル単位）
    print("box pixel:", bbox_area_px)
    bbox_area_m2 = bbox_area_px * (PIXEL_SIZE_M ** 2)  # バウンディングボックスの面積をメートル単位に変換
    estimated_z = np.sqrt((real_face_area_m2 * (focal_length_m ** 2)) / bbox_area_m2)  # z座標の推定
    return estimated_z

########################################################################################################################
### tracking
########################################################################################################################
class Args:
    def __init__(self):
        self.byte_track = False
        self.track_thresh = 0.01
        self.track_buffer = 90
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 2.0
        self.min_box_area = 10
        self.mot20 = False

# インスタンスを作成してパラメータを設定
args = Args()

from Byte_tracker import Track
from copy import deepcopy
class TRACKYOLO():
    def __init__(self):

        self.prev_gray = None
        self.args = Args()
        self.track = Track(self.args) 
        self.outputs = None

    def yolo_predict(self, frame, percentile=93, track_thresh=0.01, iou=0.7, br=True, device="cuda"):
        # Run inference on the frame
        result = yolo(frame, percentile=percentile, track_thresh=track_thresh, iou=iou, br=br, device=device)

        # Extract class probabilities
        class_probs = result.boxes.cls_prob.cpu().numpy()
        boxes = result.boxes.xywh.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        class_probs = result.boxes.cls_prob.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        # tracking
        outputs = self.track.update(scores, boxes_xyxy, class_probs)
        track_img = self.track.vis(deepcopy(frame), outputs)
        if outputs is None:
            track_img = np.zeros((120,160,3))
            boxes = np.array([[0,0,0.1,0.1]])
            class_probs = np.array([[0.0,0.0,0.0,0.0]])
        else:
            boxes = outputs[:, 1:5].numpy()
            scores = outputs[:, 5].numpy()
            class_probs = outputs[:, 7:].numpy()
            self.outputs = outputs
        return track_img, boxes, class_probs

########################################################################################################################
### mlflow
########################################################################################################################

import mlflow
def save_log(score_dict):
    """

    """
    mlflow.log_metrics(score_dict)
    mlflow.log_artifact(".hydra/config.yaml")
    mlflow.log_artifact(".hydra/hydra.yaml")
    mlflow.log_artifact(".hydra/overrides.yaml")
    mlflow.log_artifact("features.csv")

import logging
def reduce_mem_usage(df, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    start_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != 'object' and col_type != 'datetime64[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print_('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def read_annotation(annotaiton_csv_path):
    video_anno = defaultdict(list)
    if annotaiton_csv_path:
        anno_df = pd.read_csv(annotaiton_csv_path)
        for idx, d in anno_df.iterrows():
            video_file = d["video_file"]
            label = d["label"]
            video_anno[video_file].append(label)
    return video_anno

from hydra import compose, initialize_config_dir
def get_cnf(config_filename):
    """
    設定値の辞書を取得
    @return
        cnf: OmegaDict
    """
    conf_dir = os.path.join(os.getcwd(), "config")
    if not os.path.isdir(conf_dir):
        print(f"Can not find file: {conf_dir}.")
    with initialize_config_dir(config_dir=conf_dir, version_base="1.1"):
        cnf = compose(config_name=config_filename, return_hydra_config=True)
        return cnf
    

###############################################################################################
#### 損失関数
###############################################################################################
import torch.nn.functional as F
import torch.nn  as nn
class FocalLoss(nn.Module):

    def __init__(self, weight=None,
             gamma=2.5, reduction='mean'):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.reduction = reduction
        self.weight=weight

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction = self.reduction
        )