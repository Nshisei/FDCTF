import cv2
import numpy as np
import os
import random
from glob import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load
import torch
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
from setting import *
from utils import *

def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
fix_seeds()

class ThermalFallDataset(Dataset):
    def __init__(self, video_dirs, annotaiton_csv_path=None, cache_file='cache.npz', window_size=8, transform=None, jpg=0,preprocess_list=[], col=[], image_f=True, table_f=True, preprocess_f=True, feature_csv=None, skip_realtime=True):
        self.video_dirs = video_dirs
        self.image_f = image_f
        self.col = col
        self.skip_realtime = skip_realtime
        self.jpg = jpg
        self.feature_csv = feature_csv
        self.table_f = len(col)>0
        self.preprocess_list = preprocess_list
        self.cache_file = cache_file
        self.window_size = window_size
        self.transform = transform
        self.video_lens = []
        self.preprocess_f = preprocess_f
        self.video_anno = self.read_annotation(annotaiton_csv_path)
        self.frames = self.load_frames()
        self.len_n = np.sum([video - self.window_size + 1 for video in self.video_lens])

        # ラベルの偏りを修正
        self.balanced_indices = self.balance_labels()

    def balance_labels(self):
        indices = []
        for video_idx, video_len in enumerate(self.video_lens):
            for start in range(0, video_len - self.window_size + 1, 1):
                end = start + self.window_size
                label = self.label_annotation(self.video_anno[video_idx][start:end])
                indices.append((video_idx, start, label))

        label_counts = defaultdict(list)
        for i, (video_idx, start, label) in enumerate(indices):
            label_counts[label].append((video_idx, start))

        min_count = float("inf")
        for l, cnt in label_counts.items():
            if l != 0:
                min_count = min(min_count, len(cnt))
        if min_count == 0:
            return indices

        balanced_indices = []
        for label, lst in label_counts.items():
            if label == 0:
                label_0_count = int(min_count * 1.5)
                balanced_indices.extend(random.sample(lst, label_0_count))
            else:
                balanced_indices.extend(lst)
            # balanced_indices.extend(random.sample(label_counts[label], min_count))

        random.shuffle(balanced_indices)
        return balanced_indices

    def load_frames(self, image=True, col=[]):
        frames = []
        annotation = []
        for video_dir in sorted(self.video_dirs):
            video_name = os.path.basename(video_dir)
            if video_name not in self.video_anno.keys():
                continue
            if self.skip_realtime == True and "realtime" in video_name:
                continue
            # df = pd.read_csv(f'./csv_0713/{video_name}features.csv')
            # df = pd.read_csv(f'./csv/{video_name}features.csv')
            df = pd.read_csv(f'./{self.feature_csv}/{video_name}features.csv')
            df = self.make_feature_df(df)

            video_files = sorted(glob(video_dir + "/*.png"))
            annotation.append(self.video_anno[os.path.basename(video_dir)])
            
            video = []
            for video_file in video_files:
                tmp_frame = self.get_image_feature(video_file, df)
                video.append(tmp_frame)
            frames.append(video)
            self.video_lens.append(len(video_files))
        self.video_anno = annotation
        return frames

    def read_annotation(self, annotaiton_csv_path):
        video_anno = defaultdict(list)
        if annotaiton_csv_path:
            anno_df = pd.read_csv(annotaiton_csv_path)
            for idx, d in anno_df.iterrows():
                video_file = d["video_file"]
                label = d["label"]
                label = label if label in [0,1] else 2
                video_anno[video_file].append(label)
        return video_anno

    def label_annotation(self, labels):
        # return labels[-1] # 最終フレームを予測
        x = np.mean(labels[-4:])
        return int((x*2+1)//2)
        # return np.mean(labels)
        
    def preprocess(self, frame):
        """
        前処理等を加えたければここでやる
        """
        def expand_bbox(bbox, scale, frame_width, frame_height):
            x, y, w, h = bbox
            cx, cy = x, y
            new_w = w * scale
            new_h = h * scale
            new_x = max(0, int(cx - new_w / 2))
            new_y = max(0, int(cy - new_h / 2))
            new_x2 = min(frame_width, int(cx + new_w / 2))
            new_y2 = min(frame_height, int(cy + new_h / 2))
            return new_x, new_y, new_x2, new_y2
        
        def crop_and_mask_multiple(frame, bboxes, scale=2):
            processed = brighten_bright_areas(frame, percentile=90)
            processed = remove_top_edge_objects_and_small_areas(processed)
            h, w = frame.shape[:2]
            mask = np.zeros_like(frame)
            for bbox in bboxes:
                x1, y1, x2, y2 = expand_bbox(bbox, scale, w, h)
                mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            return mask

        result = yolo(frame, track_thresh=0.7, iou=0.3, percentile=93)
        bboxes = result.boxes.xywh.cpu().numpy()
        processed2 = crop_and_mask_multiple(frame, bboxes, scale=1.5)

        return processed2
    
    def get_image_feature(self, image_path, feature_df):
        tmp_frame = []
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (160, 120))
        if self.jpg:
            frame = self.convert2jpg(frame)

        
        tmp_frame.append(frame)
        features = feature_df[feature_df['filename'] == os.path.basename(image_path)]
        # if features.shape != (1, len(self.col)):
        #     features = np.array([[0.0 for i in range(len(self.col))]], dtype=float)
        # features = np.nan_to_num(features, nan=0.0)
        tmp_frame.append(features)

        return tmp_frame
    
    def make_feature_df(self, feature_df):
        
        feature_df["body_aspect_change"] = feature_df["body_aspect"] - feature_df["body_aspect"].shift(1)
        feature_df["angle_change"] = abs(feature_df["angle"] - feature_df["angle"].shift(1))
        feature_df["angle_abs_change"] = abs(feature_df["angle"]) - abs(feature_df["angle"].shift(1))
        feature_df["point_world_z_change"] = (feature_df["point_world_z"].round(1) - feature_df["point_world_z"].shift(1).round(1))
        feature_df["angle_90"] = abs(feature_df["angle"])%90
        feature_df["angle_90_change"] = abs(feature_df["angle_90"] - feature_df["angle_90"].shift(1))
        feature_df["z_decrease"] = (feature_df["point_world_z_change"] < 0) 
        feature_df["z_decrease_mean"] = feature_df["z_decrease"].rolling(window=8).mean()
        feature_df["aspect_increase"] = (feature_df["body_aspect_change"] > 0)
        feature_df["angle_abs_increase"] = (feature_df["angle_abs_change"] > 0) 
        feature_df["aspect_change"] = np.where((feature_df['body_aspect'].round(2) - feature_df['body_aspect'].shift(1).round(2)) >= 0.05, 1, 0)
        feature_df["aspect_change_mean"] = feature_df["aspect_change"].rolling(window=8).mean()
        # feature_df["bed_iou"] = feature_df["bed_iou"] >= 0.2
        feature_df["angle_change_2"] = np.where((feature_df['angle'].round(2) - feature_df['angle'].shift(1).round(2)) > 0, 1, 0)
        feature_df["angle_change_mean"] = feature_df["angle_change_2"].rolling(window=8).mean()
        feature_df["head_speed"] = np.sqrt((feature_df["head_center_x"]-feature_df["head_center_x"].shift(1))**2 + (feature_df["head_center_y"]-feature_df["head_center_y"].shift(1))**2)
        feature_df["body_speed"] = np.sqrt((feature_df["body_center_x"]-feature_df["body_center_x"].shift(1))**2 + (feature_df["body_center_y"]-feature_df["body_center_y"].shift(1))**2)

        def categorize(value):
            if value >= 2:
                return 1
            elif value >= 1:
                return 0.5
            elif value > 0:
                return 0.25
            else:
                return 0
            
        # feature_df["opt_mag_avg"] = feature_df["opt_mag_avg"].apply(categorize)
        return feature_df


    def create_windows(self, video_idx, start):
        window = self.frames[video_idx][start:start + self.window_size]
        # z_decrease = (window["point_world_z_change"] < 0) 
        # aspect_increase = (window["body_aspect_change"] > 0)
        # angle_increase = (window["angle_abs_change"] > 0) 
        # window = window['cls_prob_sitting','cls_prob_lying','cls_prob_standing'].values.tolist() + z_decrease + aspect_increase + angle_increase
        label = self.label_annotation(self.video_anno[video_idx][start:start + self.window_size])
        return window, label
    
    def convert2jpg(self, frame):
        # begin png to jpg
        if self.jpg:
            cv2.imwrite('output.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg])
            frame = cv2.imread('output.jpg')
        # end png to jpg
        return frame


    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        video_idx, start = self.balanced_indices[idx]
        window, label = self.create_windows(video_idx, start)
            
        if self.preprocess_f:
            frames = []
            if self.preprocess_f:
                frames = np.array([self.convert2jpg(self.preprocess(win[0])) for win in window], dtype=int)
            else:
                frames = np.array([self.convert2jpg(i[0]) for i in window], dtype=int)
            frames = torch.tensor(frames, dtype=torch.float32)
        else:
            frames = torch.tensor(np.array([self.convert2jpg(i[0]) for i in window], dtype=int), dtype=torch.float32)

        features = []
        for win in window:
            tmp = win[1][self.col].values
            if tmp.shape != (1, len(self.col)):
                tmp = np.zeros((1, len(self.col)))
            tmp = np.nan_to_num(tmp, nan=0.0)
            features.append(tmp)
        features = torch.tensor(np.array(features, dtype=float), dtype=torch.float32)
        return frames, torch.tensor(label, dtype=torch.long), features

    def set_features_to_use(self, feature_col, preprocess_f):
        """使用する特徴量を設定するメソッド"""
        self.col = feature_col
        self.preprocess_f = preprocess_f
        

class EventFallDataset(ThermalFallDataset):
    def __init__(self, cfg, video_dirs, anno_csv, skip_realtime=True):
        self.video_dirs = video_dirs
        self.skip_realtime = skip_realtime
        self.image_f = cfg["parameters"]["image_f"]
        self.col = cfg["features"]
        self.feature_csv = cfg["base"]["feature_csv"]
        self.jpg = cfg["parameters"]["jpg"]
        self.table_f = len(cfg["features"])>0
        self.window_size = cfg["parameters"]["window_size"]
        run_name = cfg["base"]["run_name"]
        self.output_dir = f"~/work/lstm/model_output_img/{run_name}" 
        from dataaugmentation import VideoDataAugmentation
        self.transform = VideoDataAugmentation() if cfg["parameters"]["transform"] else None
        self.video_lens = []
        self.video_anno = self.read_annotation(anno_csv)
        self.anno_df = pd.read_csv(anno_csv)
        self.classes = dict(cfg["train_set"]["classes"])
        self.track = TRACKYOLO()
        self.preprocess_f = cfg["parameters"]["preprocess_f"]

    def inference(self, model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        val_video_dirs = sorted(glob(self.video_dirs))
        frame_id = 0
        track = TRACKYOLO()
        frames = []
        features = []
        labels = []
        model_outputs = []
        softmax = nn.Softmax(dim=1)
        model.eval()
        with torch.no_grad():
            for video_dir in val_video_dirs:
                video_name = os.path.basename(video_dir)
                if video_name not in self.video_anno.keys():
                    continue
                if self.skip_realtime == True and "realtime" in video_name:
                    continue
                feature_df = self.make_feature_df(pd.read_csv(f'./{self.feature_csv}/{video_name}features.csv'))
                for video_file in sorted(glob(video_dir + "/*")):
                    frame_id += 1
                    labels.append(self.anno_df[self.anno_df["filename"] == os.path.basename(video_file).split('.')[0]]["label"].iloc[0])
                    tmp_frames = self.get_image_feature(video_file, feature_df)
                    
                    frame, feature = tmp_frames
                    if self.preprocess_f:
                        frame = self.preprocess(frame)
                    if self.jpg:
                        # begin png to jpg
                        cv2.imwrite('output.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg])
                        frame = cv2.imread('output.jpg')
                        # end png to jpg
                    frames.append(frame)
                    tmp = np.array(feature[self.col].values.tolist())
                    if tmp.shape != (1, len(self.col)):
                        tmp = np.array([[0.0 for i in range(len(self.col))]], dtype=float)
                    tmp = np.nan_to_num(tmp, nan=0.0)
                    features.append(tmp)
                    if self.window_size < len(frames):
                        frames.pop(0)
                        input_frames = torch.tensor(np.array(frames, dtype=float), dtype=torch.float32).unsqueeze(0).to(device)
                        features.pop(0)
                        input_features = torch.tensor(np.array(features, dtype=float), dtype=torch.float32).unsqueeze(0).to(device)
                        outputs = model(input_frames, input_features)
                        output_prob = softmax(outputs).cpu().numpy()[0]
                        model_outputs.append(output_prob)
                        label = np.argmax(output_prob)
                        frame = cv2.putText(frame, f"{self.classes[label]}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow("output", frame)
                    
                    os.makedirs(self.output_dir, exist_ok=True)
                    cv2.imwrite(f"{self.output_dir}/{frame_id}.png", frame)
                    cv2.waitKey(1)

        return model_outputs, labels
    