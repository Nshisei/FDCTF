import torch
import torch.nn as nn
import torchvision.models as models
from setting import *
import cv2
import numpy as np
from tqdm import tqdm
# from yolo_lstm import yolo_forward, ConvLSTM
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils import *
# from ultralytics import YOLO
import mlflow
import logging
from Dataset import EventFallDataset
# ログレベルの設定
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# classes = {
#    0: "usual",
#    1: "faling",
#    2: "slipped",
#    3: "falled",
#}


# デバイスの設定
# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
class Eval():
    def __init__(self, cfg, len_loader,logger, eval_type="train",  num_classes=4, classes=[]):
        self.save_dir = os.path.join(WORKDIR,"RESULT", cfg["base"]["run_name"])
        self.logger = logger
        os.makedirs(self.save_dir, exist_ok=True)
        self.eval_type = eval_type
        self.usual_vs_fall = True if cfg["loss_func"] == "hierarchical" else False
        self.total_loss = []
        self.total_accuracy = []
        self.total_precision = []
        self.total_recall = []
        self.total_f1_score = []
        self.total_sensitivity = []
        self.total_specificity = []
        self.len_loader = len_loader
        self.num_classes = len(classes)
        self.class_precision = {i: [] for i in range(self.num_classes)}
        self.class_recall = {i: [] for i in range(self.num_classes)}
        self.class_f1 = {i: [] for i in range(self.num_classes)}
        self.average_precision = []
        self.classes = classes
        self.epoch_eval = defaultdict(list)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        self.usual_vs_fall_result = {"usual": {"precision":[], "recall":[], "f1":[]},
                                     "fall": {"precision":[], "recall":[], "f1":[]}}


    def calculate_metrics(self, outputs, labels, loss):

        softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs)

        _, predicted = torch.max(outputs, 1)

        
        for true, pred in zip(labels.cpu(), predicted.cpu()):
            self.confusion_matrix[true, pred] += 1

        
        metrics = {'TP': np.zeros(self.num_classes), 'FP': np.zeros(self.num_classes),
                   'TN': np.zeros(self.num_classes), 'FN': np.zeros(self.num_classes)}

        for cls in range(self.num_classes):
            metrics['TP'][cls] = ((predicted == cls) & (labels == cls)).sum().item()
            metrics['FP'][cls] = ((predicted == cls) & (labels != cls)).sum().item()
            metrics['TN'][cls] = ((predicted != cls) & (labels != cls)).sum().item()
            metrics['FN'][cls] = ((predicted != cls) & (labels == cls)).sum().item()

            precision = metrics['TP'][cls] / (metrics['TP'][cls] + metrics['FP'][cls]) if (metrics['TP'][cls] + metrics['FP'][cls]) != 0 else 0
            recall = metrics['TP'][cls] / (metrics['TP'][cls] + metrics['FN'][cls]) if (metrics['TP'][cls] + metrics['FN'][cls]) != 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            self.class_precision[cls].append(precision)
            self.class_recall[cls].append(recall)
            self.class_f1[cls].append(f1_score)
  
        sensitivity = np.mean([metrics['TP'][cls] / (metrics['TP'][cls] + metrics['FN'][cls])
                               if (metrics['TP'][cls] + metrics['FN'][cls]) != 0 else 0
                               for cls in range(self.num_classes)])
        specificity = np.mean([metrics['TN'][cls] / (metrics['TN'][cls] + metrics['FP'][cls])
                               if (metrics['TN'][cls] + metrics['FP'][cls]) != 0 else 0
                               for cls in range(self.num_classes)])
        accuracy = np.mean([metrics['TP'][cls] / (metrics['TP'][cls] + metrics['FP'][cls] + metrics['FN'][cls] + metrics['TN'][cls])
                            if (metrics['TP'][cls] + metrics['FP'][cls] + metrics['FN'][cls] + metrics['TN'][cls]) != 0 else 0
                            for cls in range(self.num_classes)])
        precision = np.mean([metrics['TP'][cls] / (metrics['TP'][cls] + metrics['FP'][cls])
                             if (metrics['TP'][cls] + metrics['FP'][cls]) != 0 else 0
                             for cls in range(self.num_classes)])
        recall = sensitivity
        f1_score = np.mean([2 * (precision * recall) / (precision + recall)
                            if (precision + recall) != 0 else 0
                            for cls in range(self.num_classes)])

        
        self.total_loss.append(loss.item())
        self.total_accuracy.append(accuracy)
        self.total_precision.append(precision)
        self.total_recall.append(recall)
        self.total_f1_score.append(f1_score)
        self.total_sensitivity.append(sensitivity)
        self.total_specificity.append(specificity)
        self.calculate_map()

    def calculate_map(self):
        aps = []
        for cls in range(self.num_classes):
            precision = np.array(self.class_precision[cls])
            recall = np.array(self.class_recall[cls])
            sorted_indices = np.argsort(recall)
            precision = precision[sorted_indices]
            recall = recall[sorted_indices]
            ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
            aps.append(ap)
        self.average_precision.append(np.mean(aps))

    def usual_vs_fall_calc(self, conf_matrix):
        # Usual metrics
        true_usual = conf_matrix[0, :]  # 0行目
        pred_usual = conf_matrix[:, 0]  # 0列目
        tp_usual = true_usual[0]  # True Positive for usual
        fp_usual = pred_usual.sum() - tp_usual  # False Positive for usual
        fn_usual = true_usual.sum() - tp_usual  # False Negative for usual

        precision_usual = tp_usual / (tp_usual + fp_usual) if (tp_usual + fp_usual) != 0 else 0
        recall_usual = tp_usual / (tp_usual + fn_usual) if (tp_usual + fn_usual) != 0 else 0
        f1_usual = 2 * (precision_usual * recall_usual) / (precision_usual + recall_usual) if (precision_usual + recall_usual) != 0 else 0

        # Fall metrics
        true_fall = conf_matrix[1:, :].sum(axis=0)  # 1~3行の合計
        pred_fall = conf_matrix[:, 1:].sum(axis=1)  # 1~3列の合計
        tp_fall = true_fall[1:].sum()  # True Positive for fall (sum of 1~3行, except column 0)
        fp_fall = pred_fall.sum() - tp_fall  # False Positive for fall
        fn_fall = true_fall.sum() - tp_fall  # False Negative for fall

        precision_fall = tp_fall / (tp_fall + fp_fall) if (tp_fall + fp_fall) != 0 else 0
        recall_fall = tp_fall / (tp_fall + fn_fall) if (tp_fall + fn_fall) != 0 else 0
        f1_fall = 2 * (precision_fall * recall_fall) / (precision_fall + recall_fall) if (precision_fall + recall_fall) != 0 else 0

        self.usual_vs_fall_result["usual"]["precision"].append(precision_usual)
        self.usual_vs_fall_result["usual"]["recall"].append(recall_usual)
        self.usual_vs_fall_result["usual"]["f1"].append(f1_usual)
        self.usual_vs_fall_result["fall"]["precision"].append(precision_usual)
        self.usual_vs_fall_result["fall"]["recall"].append(recall_usual)
        self.usual_vs_fall_result["fall"]["f1"].append(f1_usual)


    def print_eval(self):
        # self.usual_vs_fall_calc(self.confusion_matrix)
        self.usual_vs_fall_calc(self.confusion_matrix)
        self.plot_confusion_matrix(self.confusion_matrix)
        

        self.epoch_eval["loss"].append(np.mean(self.total_loss))
        self.epoch_eval["Accuracy"].append(np.mean(self.total_accuracy))
        self.epoch_eval["Precision"].append(np.mean(self.total_precision))
        self.epoch_eval["Recall"].append(np.mean(self.total_recall))
        self.epoch_eval["F1_score"].append(np.mean(self.total_f1_score))
        self.epoch_eval["Sensitivity"].append(np.mean(self.total_sensitivity))
        self.epoch_eval["Specificity"].append(np.mean(self.total_specificity))
        self.epoch_eval["mAP"].append(np.mean(self.average_precision))
        self.logger.info(f"- - - - - - - - - {self.eval_type} loss - - - - - - - - - ")
        self.logger.info(f"""
        Accuracy: {(self.epoch_eval["Accuracy"][-1]):0.4f}
        Precision: {(self.epoch_eval["Precision"][-1]):0.4f}
        Recall: {(self.epoch_eval["Recall"][-1]):0.4f}
        F1_score: {(self.epoch_eval["F1_score"][-1]):0.4f}
        Sensitivity: {(self.epoch_eval["Sensitivity"][-1]):0.4f}
        Specificity: {(self.epoch_eval["Specificity"][-1]):0.4f}
        mAP: {(self.epoch_eval["mAP"][-1]):0.4f}
        """)
        if self.usual_vs_fall:
            usual = self.usual_vs_fall_result["usual"]
            fall = self.usual_vs_fall_result["fall"]
            self.logger.info(f"Class usual - Precision: {np.mean(usual['precision']):.4f}, Recall: {np.mean(usual['recall']):.4f}, F1_score: {np.mean(usual['f1']):.4f}")
            self.logger.info(f"Class fall - Precision: {np.mean(fall['precision']):.4f}, Recall: {np.mean(fall['recall']):.4f}, F1_score: {np.mean(fall['f1']):.4f}")
        

        for cls in range(self.num_classes):
            self.logger.info(f"Class {self.classes[cls]} - Precision: {np.mean(self.class_precision[cls]):.4f}, Recall: {np.mean(self.class_recall[cls]):.4f}, F1_score: {np.mean(self.class_f1[cls]):.4f}")

        self.total_loss = []
        self.total_accuracy = []
        self.total_precision = []
        self.total_recall = []
        self.total_f1_score = []
        self.total_sensitivity = []
        self.total_specificity = []
        self.class_precision = {i: [] for i in range(self.num_classes)}
        self.class_recall = {i: [] for i in range(self.num_classes)}
        self.class_f1 = {i: [] for i in range(self.num_classes)}
        self.average_precision = []

    def _plot(self):
        fig, axes = plt.subplots(1,2, figsize=(14,7))

        axes[0].plot(range(len(self.epoch_eval["loss"])), self.epoch_eval["loss"], label=f"{self.eval_type} loss")
        axes[1].plot(range(len(self.epoch_eval["Accuracy"])), self.epoch_eval["Accuracy"], label="Accuracy")
        axes[1].plot(range(len(self.epoch_eval["Precision"])), self.epoch_eval["Precision"], label="Precision")
        axes[1].plot(range(len(self.epoch_eval["Recall"])), self.epoch_eval["Recall"], label="Recall")
        axes[1].plot(range(len(self.epoch_eval["F1_score"])), self.epoch_eval["F1_score"], label="F1_score")
        axes[1].plot(range(len(self.epoch_eval["Sensitivity"])), self.epoch_eval["Sensitivity"], label="Sensitivity")
        axes[1].plot(range(len(self.epoch_eval["Specificity"])), self.epoch_eval["Specificity"], label="Specificity")
        axes[1].plot(range(len(self.epoch_eval["mAP"])), self.epoch_eval["mAP"], label="mAP")
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
        axes[0].set_title(f'{self.eval_type} loss')
        axes[1].set_title("eval")
        fig.tight_layout()
        arg_txt = arg_txt.replace('\n','').replace(' ','')
        os.makedirs(f'./result/{self.save_dir}/train_plots/', exist_ok=True)
        plt.savefig(f'./result/{self.save_dir}/train_plots/plot_{self.eval_type}_{arg_txt}.png')

    def _confutino_matrix(self, predict, labels):
        plt.figure(figsize=(6, 5))
        conf_mat = confusion_matrix(labels, predict, normalize='true')
        sns.heatmap(conf_mat, cmap = 'Blues', annot=True, fmt = '.2f')
        plt.yticks(rotation=0)
        plt.savefig(f"./confution_matrix_{self.eval_type}.png")
        plt.close()

    def plot_confusion_matrix(self, conf_matrix):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf_matrix, cmap="Blues")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))
        ax.set_xticklabels(list(self.classes))
        ax.set_yticklabels(list(self.classes))

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        plt.colorbar(im)
        plt.savefig(f"confusion_matrix_{self.eval_type}.png")
        plt.close()


def train_model(model, train_loader, val_loader, cfg, logger):
    basic_info = cfg["base"] 
    train_set = cfg["train_set"]
    params = cfg["parameters"]
    features = cfg["features"]
    epochs = params["epochs"]
    
    save_dir = os.path.join(WORKDIR, "RESULT", cfg["base"]["experiment_name"], cfg["base"]["run_name"])
    
    mlflow.set_experiment("exp_name")
    with mlflow.start_run(run_name=basic_info["run_name"]) as run:

        # Model, loss function, and optimizer
        best_val_loss = float('inf')
        model = model.to(device)
        mlflow.log_params(params)
        if params["class_weight"] > 0:
            from collections import Counter
            train_label_counts = Counter()
            for _, label, *__ in train_loader:
                train_label_counts.update(label.numpy())
            # Val dataset label counts
            val_label_counts = Counter()
            for _, label, *__ in val_loader:
                val_label_counts.update(label.numpy())
            logger.info("Train label counts:", train_label_counts)
            logger.info("Val label counts:", val_label_counts)
            class_weights = torch.tensor([1/train_label_counts[i]*1000 if train_label_counts[i] != 0 else 0 for i in range(params["class_num"])], device=device)
            # criterion = nn.CrossEntropyLoss(weight=class_weights)
            criterion = FocalLoss(weight=class_weights)
        else:
            # criterion = nn.CrossEntropyLoss()
            criterion = FocalLoss()

        optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=0.0001)

        train_eval = eval(cfg, len(train_loader), eval_type="train", classes=train_set["classes"])
        val_eval = eval(cfg, len(val_loader), eval_type="val", classes=train_set["classes"])
        for epoch in tqdm(range(epochs), desc="Epoch"):
            # トレーニングループ
            model.train()
            train_loss = 0
            for inputs, labels, features in tqdm(train_loader, desc="Iteration", leave=False):

                inputs, labels, features = inputs.to(device), labels.to(device), features.to(device)
                inputs = inputs.float()  # floatにキャスト
                optimizer.zero_grad()
                outputs = model(inputs, features)
                loss = criterion(outputs, labels)
                train_eval.calculate_metrics(outputs, labels, loss)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()


            train_loss /= len(train_loader)
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            for idx, name in train_set["classes"]:
                mlflow.log_metric(f'train_{name}_precision',train_eval.class_precision[idx], step=epoch)
                mlflow.log_metric(f'train_{name}_recall',train_eval.class_recall[idx], step=epoch)
                mlflow.log_metric(f'train_{name}_f1',train_eval.class_f1[idx], step=epoch)
            
            train_eval.self.logger.print_eval()
            logger.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}')

            # 検証ループ
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels, features in val_loader:
                    inputs, labels, features = inputs.to(device), labels.to(device), features.to(device)
                    inputs = inputs.float()  # floatにキャスト
                    outputs = model(inputs, features)
                    loss = criterion(outputs, labels)
                    val_eval.calculate_metrics(outputs, labels, loss)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            mlflow.log_metric('val_loss', train_loss, step=epoch)
            for idx, name in train_set["classes"]:
                mlflow.log_metric(f'val_{name}_precision',val_eval.class_precision[idx], step=epoch)
                mlflow.log_metric(f'val_{name}_recall',val_eval.class_recall[idx], step=epoch)
                mlflow.log_metric(f'val_{name}_f1',val_eval.class_f1[idx], step=epoch)
            val_eval.self.logger.print_eval()
            logger.info(f'Epoch [{epoch + 1}/{epochs}], Val Loss: {val_loss:.4f}')


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info("updata best ckpt")
                save_model_path = f'{save_dir}/best_ckpt.pth'
                torch.save(model.state_dict(), save_model_path)
                mlflow.log_artifact(save_model_path, "best_model")

        logger.info("####################### Training completed.")

        logger.info("####################### Val Event")
        eventdata = EventFallDataset(cfg)
        event_val = os.path.basename(os.path.dirname(train_set["event_video_dir"]))
        model_outputs, labels = eventdata.inference(model)
        event_fall_csv_path = f"{save_dir}/lstm_prob_{event_val}.csv"
        pd.DataFrame(np.concatenate([np.concatenate([[np.zeros(3) for _ in range(params["window_size"])], [o for o in model_outputs]]),
                                     np.expand_dims(labels, axis=1)], 1), columns=["usual", "falling", "fall", "label"]).to_csv(event_fall_csv_path)
        mlflow.log_artifact(event_fall_csv_path, "event_eval_csv")
        data = pd.read_csv(event_fall_csv_path, index_col=0)
        from eval_fall_slip import plot_pr
        plt = plot_pr(data)
        plt.savefig(f"{save_dir}/recall_precision.png")
        mlflow.log_artifact(f"{save_dir}/recall_precision.png", "event_eval_result")

        logger.info("####################### Eval Event completed")

        

class Eval2:
    def __init__(self, cfg, len_loader, logger, eval_type="train", num_classes=4):
        self.save_dir = os.path.join("RESULT", cfg["base"]["run_name"])
        self.logger = logger
        os.makedirs(self.save_dir, exist_ok=True)
        self.eval_type = eval_type
        self.binary_class = ["usual", "fall"]
        self.type_class = ["emergency", "caution", "unknown"]
        self.confusion_matrix_b = np.zeros((len(self.binary_class), len(self.binary_class)), dtype=int)
        self.confusion_matrix_t = np.zeros((len(self.type_class), len(self.type_class)), dtype=int)
        self.loss = []
        self.epoch_eval = defaultdict(list)


    def calculate_metrics(self, output, target, loss):
        # モデルの出力
        fall_output, type_output = output
        # B, T = target.shape  # batch_size, window_size
        # if T > 1:
        #     fall_output = fall_output[:, -1, :]  # 最終フレームのみを対象
        #     target = target[:, -1]  # 最終フレームのみを対象

        # ラベルの処理
        binary_labels = (target > 0).long()  # 転倒判定用ラベル (0: usual, 1: fall)
        mask = target > 0  # タイプ分類が可能なデータのマスク
        subcategory_labels = (target - 1).long()[mask]  # 転倒タイプ分類用ラベル

        # Softmaxで確率に変換
        softmax = nn.Softmax(dim=1)
        fall_output = softmax(fall_output)
        type_output = softmax(type_output)

        # 転倒判定の予測
        _, fall_predicted = torch.max(fall_output, 1)

        # 転倒タイプ分類の予測（マスク適用）
        if torch.sum(mask) > 0:  # マスクされたデータが存在する場合
            filtered_type_output = type_output[mask]
            _, type_predicted = torch.max(filtered_type_output, 1)
        else:
            type_predicted = torch.tensor([], dtype=torch.long)

        # 混同行列の更新（転倒判定用）
        for true, pred in zip(binary_labels.cpu(), fall_predicted.cpu()):
            self.confusion_matrix_b[true, pred] += 1

        # 混同行列の更新（転倒判定用）
        for true, pred in zip(subcategory_labels.cpu(), type_predicted.cpu()):
            self.confusion_matrix_t[true, pred] += 1

        self.loss.append(loss.item())
        

    def print_eval(self):
        # 混同行列の可視化
        self.plot_confusion_matrix(self.confusion_matrix_b, "binary")
        self.plot_confusion_matrix(self.confusion_matrix_t, "type")

         # メトリクス計算用変数の初期化
        binary_metrics = {'TP': np.zeros(2), 'FP': np.zeros(2), 'TN': np.zeros(2), 'FN': np.zeros(2)}

        # 転倒判定用メトリクスの計算
        for cls in range(2):  # binary_labelsは0, 1の2クラスのみ
            binary_metrics['TP'][cls] = self.confusion_matrix_b[cls, cls]
            binary_metrics['FP'][cls] = self.confusion_matrix_b[cls, 1 - cls]
            binary_metrics['TN'][cls] = self.confusion_matrix_b[1 - cls, 1 - cls]
            binary_metrics['FN'][cls] = self.confusion_matrix_b[1 - cls, cls]

        # 転倒タイプ分類用メトリクスの計算
        type_metrics = {'TP': np.zeros(len(self.type_class)), 'FP': np.zeros(len(self.type_class)),
                        'TN': np.zeros(len(self.type_class)), 'FN': np.zeros(len(self.type_class))}
        
        for cls in range(len(self.type_class)):  # 転倒タイプ分類のクラス数
            type_metrics['TP'][cls] = self.confusion_matrix_t[cls, cls]
            type_metrics['FP'][cls] = np.sum(self.confusion_matrix_t[:, cls]) - type_metrics['TP'][cls]
            type_metrics['FN'][cls] = np.sum(self.confusion_matrix_t[cls, :]) - type_metrics['TP'][cls]
            # True Negative (TN): クラス cls ではなく、正しくクラス cls 以外と予測された数
            type_metrics['TN'][cls] = np.sum(self.confusion_matrix_t) - (
                type_metrics['TP'][cls] + type_metrics['FP'][cls] + type_metrics['FN'][cls])

        # メトリクスの集計
        binary_precision_per_class = []
        binary_recall_per_class = []
        binary_f1_per_class = []

        self.logger.info(f"- - - - - - - - -  {self.eval_type} Metrics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        for cls in range(2):  # binary classification classes (usual, fall)
            precision = (binary_metrics['TP'][cls] / (binary_metrics['TP'][cls] + binary_metrics['FP'][cls])
                        if (binary_metrics['TP'][cls] + binary_metrics['FP'][cls]) != 0 else 0)
            recall = (binary_metrics['TP'][cls] / (binary_metrics['TP'][cls] + binary_metrics['FN'][cls])
                    if (binary_metrics['TP'][cls] + binary_metrics['FN'][cls]) != 0 else 0)
            f1_score = (2 * (precision * recall) / (precision + recall)
                        if (precision + recall) != 0 else 0)

            binary_precision_per_class.append(precision)
            binary_recall_per_class.append(recall)
            binary_f1_per_class.append(f1_score)

            self.logger.info(f"Class {self.binary_class[cls]} - Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}, "
                            f"F1_score: {f1_score:.4f}")

        # Binary metrics overall
        binary_precision = np.mean(binary_precision_per_class)
        binary_recall = np.mean(binary_recall_per_class)
        binary_f1_score = np.mean(binary_f1_per_class)

        # Type classification metrics
        type_precision_per_class = []
        type_recall_per_class = []
        type_f1_per_class = []

        for cls in range(len(self.type_class)):  # type classification classes
            precision = (type_metrics['TP'][cls] / (type_metrics['TP'][cls] + type_metrics['FP'][cls])
                        if (type_metrics['TP'][cls] + type_metrics['FP'][cls]) != 0 else 0)
            recall = (type_metrics['TP'][cls] / (type_metrics['TP'][cls] + type_metrics['FN'][cls])
                    if (type_metrics['TP'][cls] + type_metrics['FN'][cls]) != 0 else 0)
            f1_score = (2 * (precision * recall) / (precision + recall)
                        if (precision + recall) != 0 else 0)

            type_precision_per_class.append(precision)
            type_recall_per_class.append(recall)
            type_f1_per_class.append(f1_score)

            self.logger.info(f"Class {self.type_class[cls]} - Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}, "
                            f"F1_score: {f1_score:.4f}")

        # メトリクスの平均値を計算して記録
        Type_Precision = np.mean(type_precision_per_class)
        Type_Recall = np.mean(type_recall_per_class)
        Type_F1_score = np.mean(type_f1_per_class)

        # ログに出力        
        self.logger.info(f"""
        Loss: {np.sum(self.loss):0.4f}
        Binary Precision: {binary_precision:0.4f}
        Binary Recall: {binary_recall:0.4f}
        Binary F1_score: {binary_f1_score:0.4f}
        Type Precision: {Type_Precision:0.4f}
        Type Recall: {Type_Recall:0.4f}
        Type F1_score: {Type_F1_score:0.4f}
        """)

        self.metrics = {
            "Binary Precision": binary_precision,
            "Binary Recall": binary_recall,
            "Binary F1_score": binary_f1_score,
            "Type Precision": Type_Precision,
            "Type Recall": Type_Recall,
            "Type F1_score": Type_F1_score,
        }


    def plot_confusion_matrix(self, conf_matrix, type):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf_matrix, cmap="Blues")
        if type == "binary":
            cls_name = ["usual", "fall"]
        else:
            cls_name = ["emergency", "caution", "unknown"]

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(cls_name)))
        ax.set_yticks(np.arange(len(cls_name)))
        ax.set_xticklabels(cls_name)
        ax.set_yticklabels(cls_name)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(cls_name)):
            for j in range(len(cls_name)):
                ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        plt.colorbar(im)
        plt.savefig(f"confusion_matrix_{self.eval_type}_{type}.png")
        plt.close()


class Eval3:
    def __init__(self, cfg, len_loader, logger, eval_type="train", num_classes=4):
        self.save_dir = os.path.join("RESULT", cfg["base"]["run_name"])
        self.logger = logger
        os.makedirs(self.save_dir, exist_ok=True)
        self.eval_type = eval_type
        self.binary_class = ["usual", "fall"]
        self.type_class = ["emergency", "caution", "unknown"]
        self.confusion_matrix_b = np.zeros((len(self.binary_class), len(self.binary_class)), dtype=int)
        self.confusion_matrix_t = np.zeros((len(self.type_class), len(self.type_class)), dtype=int)
        self.loss = []
        self.epoch_eval = defaultdict(list)


    def calculate_metrics(self, output, target, loss):
        # モデルの出力
        fall_output, type_output = output
        B, T = target.shape  # batch_size, window_size
        if T > 1:
            fall_output = fall_output[:, -1, :]  # 最終フレームのみを対象
            target = target[:, -1]  # 最終フレームのみを対象

        # ラベルの処理
        binary_labels = (target > 0).long()  # 転倒判定用ラベル (0: usual, 1: fall)
        mask = target > 0  # タイプ分類が可能なデータのマスク
        subcategory_labels = (target - 1).long()[mask]  # 転倒タイプ分類用ラベル

        # Softmaxで確率に変換
        softmax = nn.Softmax(dim=1)
        fall_output = softmax(fall_output)
        type_output = softmax(type_output)

        # 転倒判定の予測
        _, fall_predicted = torch.max(fall_output, 1)

        # 転倒タイプ分類の予測（マスク適用）
        if torch.sum(mask) > 0:  # マスクされたデータが存在する場合
            filtered_type_output = type_output[mask]
            _, type_predicted = torch.max(filtered_type_output, 1)
        else:
            type_predicted = torch.tensor([], dtype=torch.long)

        # 混同行列の更新（転倒判定用）
        for true, pred in zip(binary_labels.cpu(), fall_predicted.cpu()):
            self.confusion_matrix_b[true, pred] += 1

        # 混同行列の更新（転倒判定用）
        for true, pred in zip(subcategory_labels.cpu(), type_predicted.cpu()):
            self.confusion_matrix_t[true, pred] += 1

        self.loss.append(loss.item())
        

    def print_eval(self):
        # 混同行列の可視化
        self.plot_confusion_matrix(self.confusion_matrix_b, "binary")
        self.plot_confusion_matrix(self.confusion_matrix_t, "type")

         # メトリクス計算用変数の初期化
        binary_metrics = {'TP': np.zeros(2), 'FP': np.zeros(2), 'TN': np.zeros(2), 'FN': np.zeros(2)}

        # 転倒判定用メトリクスの計算
        for cls in range(2):  # binary_labelsは0, 1の2クラスのみ
            binary_metrics['TP'][cls] = self.confusion_matrix_b[cls, cls]
            binary_metrics['FP'][cls] = self.confusion_matrix_b[cls, 1 - cls]
            binary_metrics['TN'][cls] = self.confusion_matrix_b[1 - cls, 1 - cls]
            binary_metrics['FN'][cls] = self.confusion_matrix_b[1 - cls, cls]

        # 転倒タイプ分類用メトリクスの計算
        type_metrics = {'TP': np.zeros(len(self.type_class)), 'FP': np.zeros(len(self.type_class)),
                        'TN': np.zeros(len(self.type_class)), 'FN': np.zeros(len(self.type_class))}
        
        for cls in range(len(self.type_class)):  # 転倒タイプ分類のクラス数
            type_metrics['TP'][cls] = self.confusion_matrix_t[cls, cls]
            type_metrics['FP'][cls] = np.sum(self.confusion_matrix_t[:, cls]) - type_metrics['TP'][cls]
            type_metrics['FN'][cls] = np.sum(self.confusion_matrix_t[cls, :]) - type_metrics['TP'][cls]
            # True Negative (TN): クラス cls ではなく、正しくクラス cls 以外と予測された数
            type_metrics['TN'][cls] = np.sum(self.confusion_matrix_t) - (
                type_metrics['TP'][cls] + type_metrics['FP'][cls] + type_metrics['FN'][cls])

        # メトリクスの集計
        binary_precision_per_class = []
        binary_recall_per_class = []
        binary_f1_per_class = []

        self.logger.info(f"- - - - - - - - -  {self.eval_type} Metrics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        for cls in range(2):  # binary classification classes (usual, fall)
            precision = (binary_metrics['TP'][cls] / (binary_metrics['TP'][cls] + binary_metrics['FP'][cls])
                        if (binary_metrics['TP'][cls] + binary_metrics['FP'][cls]) != 0 else 0)
            recall = (binary_metrics['TP'][cls] / (binary_metrics['TP'][cls] + binary_metrics['FN'][cls])
                    if (binary_metrics['TP'][cls] + binary_metrics['FN'][cls]) != 0 else 0)
            f1_score = (2 * (precision * recall) / (precision + recall)
                        if (precision + recall) != 0 else 0)

            binary_precision_per_class.append(precision)
            binary_recall_per_class.append(recall)
            binary_f1_per_class.append(f1_score)

            self.logger.info(f"Class {self.binary_class[cls]} - Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}, "
                            f"F1_score: {f1_score:.4f}")

        # Binary metrics overall
        binary_precision = np.mean(binary_precision_per_class)
        binary_recall = np.mean(binary_recall_per_class)
        binary_f1_score = np.mean(binary_f1_per_class)

        # Type classification metrics
        type_precision_per_class = []
        type_recall_per_class = []
        type_f1_per_class = []

        for cls in range(len(self.type_class)):  # type classification classes
            precision = (type_metrics['TP'][cls] / (type_metrics['TP'][cls] + type_metrics['FP'][cls])
                        if (type_metrics['TP'][cls] + type_metrics['FP'][cls]) != 0 else 0)
            recall = (type_metrics['TP'][cls] / (type_metrics['TP'][cls] + type_metrics['FN'][cls])
                    if (type_metrics['TP'][cls] + type_metrics['FN'][cls]) != 0 else 0)
            f1_score = (2 * (precision * recall) / (precision + recall)
                        if (precision + recall) != 0 else 0)

            type_precision_per_class.append(precision)
            type_recall_per_class.append(recall)
            type_f1_per_class.append(f1_score)

            self.logger.info(f"Class {self.type_class[cls]} - Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}, "
                            f"F1_score: {f1_score:.4f}")

        # メトリクスの平均値を計算して記録
        Type_Precision = np.mean(type_precision_per_class)
        Type_Recall = np.mean(type_recall_per_class)
        Type_F1_score = np.mean(type_f1_per_class)

        # ログに出力        
        self.logger.info(f"""
        Loss: {np.sum(self.loss):0.4f}
        Binary Precision: {binary_precision:0.4f}
        Binary Recall: {binary_recall:0.4f}
        Binary F1_score: {binary_f1_score:0.4f}
        Type Precision: {Type_Precision:0.4f}
        Type Recall: {Type_Recall:0.4f}
        Type F1_score: {Type_F1_score:0.4f}
        """)

        self.metrics = {
            "Binary Precision": binary_precision,
            "Binary Recall": binary_recall,
            "Binary F1_score": binary_f1_score,
            "Type Precision": Type_Precision,
            "Type Recall": Type_Recall,
            "Type F1_score": Type_F1_score,
        }


    def plot_confusion_matrix(self, conf_matrix, type):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf_matrix, cmap="Blues")
        if type == "binary":
            cls_name = ["usual", "fall"]
        else:
            cls_name = ["emergency", "caution", "unknown"]

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(cls_name)))
        ax.set_yticks(np.arange(len(cls_name)))
        ax.set_xticklabels(cls_name)
        ax.set_yticklabels(cls_name)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(cls_name)):
            for j in range(len(cls_name)):
                ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        plt.colorbar(im)
        plt.savefig(f"confusion_matrix_{self.eval_type}_{type}.png")
        plt.close()


import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict

class Eval4:
    def __init__(self, cfg, len_loader, logger, eval_type="train", num_classes=3):
        self.save_dir = os.path.join("RESULT", cfg["base"]["run_name"])
        self.logger = logger
        os.makedirs(self.save_dir, exist_ok=True)
        self.eval_type = eval_type
        self.type_class = ["emergency", "caution", "unknown"]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        self.loss = []
        self.epoch_eval = defaultdict(list)

    def calculate_metrics(self, main_probs, main_true_label_type, loss):
        """
        main_probs: (batch_size, 3) - emergency, caution, unknown の確率
        main_true_label_type: (batch_size) - 正解ラベル (0: emergency, 1: caution, 2: unknown)
        loss: 現在のバッチの損失
        """
        # Softmaxがまだ適用されていないなら、適用
        if not torch.is_tensor(main_probs):
            main_probs = torch.tensor(main_probs)
        if not torch.is_tensor(main_true_label_type):
            main_true_label_type = torch.tensor(main_true_label_type)

        # 最も確率の高いクラスを予測ラベルとする
        _, predicted = torch.max(main_probs, 1)

        # 混同行列の更新
        for true, pred in zip(main_true_label_type.cpu(), predicted.cpu()):
            self.confusion_matrix[true, pred] += 1

        # ロスを保存
        self.loss.append(loss.item())

    def print_eval(self):
        # 混同行列の可視化
        self.plot_confusion_matrix(self.confusion_matrix, "type")

        # メトリクス計算用変数の初期化
        metrics = {'TP': np.zeros(len(self.type_class)), 'FP': np.zeros(len(self.type_class)),
                   'TN': np.zeros(len(self.type_class)), 'FN': np.zeros(len(self.type_class))}

        # メトリクスの計算
        for cls in range(len(self.type_class)):
            metrics['TP'][cls] = self.confusion_matrix[cls, cls]
            metrics['FP'][cls] = np.sum(self.confusion_matrix[:, cls]) - metrics['TP'][cls]
            metrics['FN'][cls] = np.sum(self.confusion_matrix[cls, :]) - metrics['TP'][cls]
            # True Negative (TN): クラス cls ではなく、正しくクラス cls 以外と予測された数
            metrics['TN'][cls] = np.sum(self.confusion_matrix) - (
                metrics['TP'][cls] + metrics['FP'][cls] + metrics['FN'][cls])

        # メトリクスの集計
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        self.logger.info(f"- - - - - - - - -  {self.eval_type} Metrics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        for cls in range(len(self.type_class)):
            precision = (metrics['TP'][cls] / (metrics['TP'][cls] + metrics['FP'][cls])
                         if (metrics['TP'][cls] + metrics['FP'][cls]) != 0 else 0)
            recall = (metrics['TP'][cls] / (metrics['TP'][cls] + metrics['FN'][cls])
                      if (metrics['TP'][cls] + metrics['FN'][cls]) != 0 else 0)
            f1_score = (2 * (precision * recall) / (precision + recall)
                        if (precision + recall) != 0 else 0)

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1_score)

            self.logger.info(f"Class {self.type_class[cls]} - Precision: {precision:.4f}, "
                             f"Recall: {recall:.4f}, "
                             f"F1_score: {f1_score:.4f}")

        # 全体のメトリクス
        Precision = np.mean(precision_per_class)
        Recall = np.mean(recall_per_class)
        F1_score = np.mean(f1_per_class)

        # ログに出力        
        self.logger.info(f"""
        Loss: {np.sum(self.loss):0.4f}
        Precision: {Precision:0.4f}
        Recall: {Recall:0.4f}
        F1_score: {F1_score:0.4f}
        """)

        self.metrics = {
            "Precision": Precision,
            "Recall": Recall,
            "F1_score": F1_score,
        }

    def plot_confusion_matrix(self, conf_matrix, type):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf_matrix, cmap="Blues")
        cls_name = ["emergency", "caution", "unknown"]

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(cls_name)))
        ax.set_yticks(np.arange(len(cls_name)))
        ax.set_xticklabels(cls_name)
        ax.set_yticklabels(cls_name)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(cls_name)):
            for j in range(len(cls_name)):
                ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        plt.colorbar(im)
        plt.savefig(f"confusion_matrix_{self.eval_type}_{type}.png")
        plt.close()
