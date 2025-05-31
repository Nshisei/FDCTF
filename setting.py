import os
import numpy as np
from glob import glob

# path
D = "/mnt/d/"
ANNTATION4 = "/home/shisei/work/lstm/annotation4.csv"
FALLBASICEACH = "/mnt/d/mydatasets/fall_dataset/annotation_csv/fall_basic_each_video.csv"
FALL0713ANNOTATION = "/mnt/d/mydatasets/fall_dataset/annotation_csv/fall_0713_annotation.csv"
FALL0713REAL2ANNOTATION = "/mnt/d/mydatasets/fall_dataset/annotation_csv/fall_0713_annotation.csv"
LONGMOVIEANNOTATION = "/mnt/d/mydatasets/fall_dataset/annotation_csv/long_movie_annotation.csv"
ANNTATION5 = "/mnt/d/mydatasets/fall_dataset/fall0713/annotation5.csv"
ANNTATION6 = "/home/shisei/work/lstm/annotation6.csv"
ALL_VIDEO_DIR_PATH_0713 = "/mnt/d/mydatasets/fall_dataset/fall0713/"
ALL_VIDEO_DIR_PATH = "/mnt/d/mydatasets/fall_dataset/rename_fld/"
LONG_MOVIE = "/mnt/d/mydatasets/fall_dataset/long_movie"
ALL_MP4 = sorted(glob(ALL_VIDEO_DIR_PATH + "*.mp4"))
ALL_MP40713 = sorted(glob(ALL_VIDEO_DIR_PATH_0713 + "*.mp4"))
ALL_VIDEO_FILES = sorted(glob(ALL_VIDEO_DIR_PATH + "/*/*.png"))
BEST0710 = "/mnt/d/trained_model/yolov8n_20240710/weights/best.pt"
BEST0710 = "/mnt/d/trained_model/yolov8n_20240710/weights/best.pt"
BEST0804BR = "/mnt/d/trained_model/yolov8n_bright_0804/weights/best.pt"
BEST0804 = "/mnt/d/trained_model/yolov8n_0804/weights/best.pt"
ANNO_CSV_DIR = "/mnt/d/mydatasets/fall_dataset/annotation_csv/"
MASKIMGDIR = '/mnt/d/mydatasets/fall_dataset/mask_imgs/'

# const
IMAGE_WIDTH = 160  # 画像の幅（ピクセル）
IMAGE_HIGHT = 120  # 画像の高さ（ピクセル）
PIXEL_SIZE_UM = 12  # ピクセルサイズ（マイクロメートル）
PIXEL_SIZE_M = PIXEL_SIZE_UM * 1e-6  # ピクセルサイズ（メートル）
HORIZONTAL_FOV_DEG = 95  # 視野角（度
HEAD_AREA = 0.16*0.35 # 顔の面積(m^2)
YAW = np.radians(48)    # ヨー角 (z軸回り, 首を左右に動かす)
# PITCH = np.radians(-26)  # ピッチ角 (y軸回り, 首を縦に動かす)
PITCH = np.radians(-41.61)  # ピッチ角 (y軸回り, 首を縦に動かす)
ROLL = np.radians(0)    # ロール角（x軸回り, 首をかしげる）
FPS = 8
FALL_CLASSES = {
    0: "usual",
    1: "slipped",
    2: "fell"
}

# params
COL = ['cls_prob_sitting','cls_prob_lying','cls_prob_standing',"body_aspect_change","angle_change","point_world_z_change"]

# camera
from ThermoDetection.Scene import Scene
SCENE = Scene("/home/shisei/work/lstm/ThermoDetection/scene_lab_best3.json")
CAMERA = SCENE.cameras["camera1"]

################################################################
### SETTING CONSTRACT
################################################################
WORKDIR = "/home/shisei/lstm/"
OUTPUTCSV = os.path.join(WORKDIR, "outputs/csvs/")
OUTPUTIMAGE = os.path.join(WORKDIR, "outputs/images/")
