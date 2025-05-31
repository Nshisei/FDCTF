import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def concat_pred_event(pred_event):
    """

    """

    new_event = []
    i = 0
    while i < len(pred_event):
        current_event = pred_event[i]

        # 次のイベントと結合できるかどうかをチェック
        while i < len(pred_event) - 1 and abs(pred_event[i][1] - pred_event[i + 1][0]) < 8:
            current_event = [current_event[0], pred_event[i + 1][1], current_event[2]]
            i += 1

        new_event.append(current_event)
        i += 1

    return new_event

def make_event(ground_truth, label=[2, 3]):
    ground_event = []
    event = None
    start_idx = None

    for idx, i in enumerate(ground_truth):
        if i in label:
            if event is None:  # 新しいイベントが始まった
                event = i
                start_idx = idx
        else:
            if event is not None:  # イベントが終わった
                ground_event.append([start_idx, idx - 1, event])
                event = None
                start_idx = None

    # ループが終了した時に最後のイベントを追加
    if event is not None:
        ground_event.append([start_idx, len(ground_truth) - 1, event])

    return ground_event

def plot_pr(data, save_dir):
    # 定数
    Z_max = 4
    
    # Recall, Precisionを格納するリスト
    
    D_max_values = range(1, 40)  # D_maxを1から30まで変化させる
    ground_truth = np.array(data["label"])
    ground_event = make_event(ground_truth, label=[2,3])
    result_ = []
    max_f1 = {"f1":0, "th": 0, "D_max": 0}
    max_f1_TP = []
    plt.figure(figsize=(10, 6))
    for th in ["argmax", 0.5, 0.6]:
        if str(th) == "argmax":
            long_predictions = np.array([int(np.argmax(out[["usual", "falling", "fall"]])) for i, out in data.iterrows()])
        else:
            long_predictions = []
            for i, out in data.iterrows():
                long_predictions.append(int(np.argmax([out[["usual", "falling", "fall"]] >= th])))
            long_predictions = np.array(long_predictions)
        pred_event =  make_event(long_predictions, label=[2])
        pred_event = concat_pred_event(pred_event)
        pred_event = [p for p in pred_event if abs(p[0]-p[1]) >= Z_max]
    
        recalls = []
        precisions = []
    
        
        for D_max in D_max_values:
            # カウントの初期化
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            FP_events = []
            FN_events = []
            TP_events = []
        
            # マッチング済みのインデックスを保持するセット
            matched_gt_indices = set()
            matched_pred_indices = set()
        
            # ground_truthとpred_eventをループしてチェック
            for g_idx, (g_start, g_end, ge) in enumerate(ground_event):
                matched = False
                for p_idx, (p_start, p_end, pe) in enumerate(pred_event):
                    # D_max以内で一致するかチェックし、すでにマッチング済みでないか確認
                    if p_idx not in matched_pred_indices and g_idx not in matched_gt_indices:
                        if abs(g_start - p_start) <= D_max and abs(g_end - p_end) <= (5 * D_max):
                            TP += 1
                            matched_gt_indices.add(g_idx)
                            matched_pred_indices.add(p_idx)
                            matched = True
                            TP_events.append([p_start, p_end, ge])
                            break
                
                if not matched:
                    FN += 1
                    FN_events.append([g_start, g_end, ge])
        
            # 予測イベントがどの正解イベントとも一致しない場合はFPとカウント
            for p_idx, (p_start, p_end, pe) in enumerate(pred_event):
                if p_idx not in matched_pred_indices:  # すでにマッチング済みでないか確認
                    matched = False
                    for g_idx, (g_start, g_end, ge) in enumerate(ground_event):
                        # D_max以内で一致するかチェックし、すでにマッチング済みでないか確認
                        if g_idx not in matched_gt_indices:
                            if abs(g_start - p_start) <= D_max and abs(g_end - p_end) <= (5 * D_max):
                                matched_gt_indices.add(g_idx)
                                matched_pred_indices.add(p_idx)
                                matched = True
                                break
        
                    if not matched:
                        FP += 1
                        FP_events.append([p_start, p_end, pe])
        
            # Recall, Precisionの計算
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            if D_max%8 == 0 and D_max != 0:
                # 結果表示
                print("Threshold", th)
                print("D max", D_max)
                print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
                recall =  TP/(TP+FN) if (TP+FN) != 0 else 0
                precision = TP/(TP+FP) if (TP+FP) != 0 else 0
                f1 = 2*recall*precision/(recall+precision) if (recall+precision) != 0 else 0
                print(f"recall: {recall:.3f}")
                print(f"precision: {precision:.3f}")
                print(f"f1: {f1:.3f}")
            
            recalls.append(recall)
            precisions.append(precision)
            f1 = 2*recall*precision/(recall + precision) if (recall + precision) != 0 else 0
            result_.append([th, D_max, recall, precision, f1])
            if max_f1["f1"] < f1:
                max_f1["f1"] = f1
                max_f1["D_max"] = D_max
                max_f1["th"] = str(th)
                max_f1_TP = TP_events
            
        
        # グラフ描画
        plt.plot(D_max_values, recalls, label=f'Recall threshold={th}', marker='o')
        plt.plot(D_max_values, precisions, label=f'Precision threshold={th}', marker='x')
    plt.xlabel('D_max')
    plt.ylabel('Score')
    plt.title('Recall and Precision vs D_max')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/recall_precision.png")
    plt.show()
    result_df = pd.DataFrame(result_, columns=['Threshold', 'D_max', 'Recall', 'Precision', 'f1'])
    result_df.to_csv(f"{save_dir}/recall_precision.csv")
    return max_f1_TP

def plot_histogram_of_offsets(data, save_dir):
    Z_max = 4
    D_max_values = range(1, 40)
    
    ground_truth = np.array(data["label"])
    ground_event = make_event(ground_truth, label=[2, 3])
    
    long_predictions = np.array([int(np.argmax(out[["usual", "falling", "fall"]])) for i, out in data.iterrows()])
    pred_event = make_event(long_predictions, label=[2])
    pred_event = concat_pred_event(pred_event)
    pred_event = [p for p in pred_event if abs(p[0] - p[1]) >= Z_max]

    offsets_in_seconds = []
    matched_gt_indices = set()
    matched_pred_indices = set()
    TP_events = []
    
    for D_max in D_max_values:
        for g_idx, (g_start, g_end, ge) in enumerate(ground_event):
            for p_idx, (p_start, p_end, pe) in enumerate(pred_event):
                if p_idx not in matched_pred_indices and g_idx not in matched_gt_indices:
                    if abs(g_start - p_start) <= D_max and abs(g_end - p_end) <= (5 * D_max):
                        matched_gt_indices.add(g_idx)
                        matched_pred_indices.add(p_idx)
                        TP_events.append([p_start, p_end, ge])
                        offset_in_seconds = (p_start - g_start) / 8  # Convert offset from frames to seconds
                        offsets_in_seconds.append(offset_in_seconds)

    # Set bins to 1-second increments, with x-axis range from -5 to 5
    bins = np.arange(-5, 6, 1)

    fig, ax1 = plt.subplots()

    # Plot the histogram (left y-axis)
    counts, bin_edges, _ = ax1.hist(offsets_in_seconds, bins=bins, edgecolor='black', align='mid', label='TP Event Count')
    ax1.set_xlabel('Error time with correct event (seconds)')
    ax1.set_ylabel('TP Event Count')
    ax1.set_xticks(np.arange(-5, 6, 1))  # Set x-ticks to 1-second increments
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([0, 8])
    ax1.grid(True, which='both', axis='x', linestyle='--', linewidth=0.7)

    # Plot cumulative frequency (right y-axis)
    ax2 = ax1.twinx()
    cumulative_counts = np.cumsum(counts)
    cumulative_freq = cumulative_counts / len(ground_event)  # Normalize to total correct events
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.plot(bin_centers, cumulative_freq, color='blue', marker='o', label='Cumulative Frequency')
    ax2.set_ylabel('Cumulative Frequency')
    ax2.set_ylim([0, 1.1])

    # Add gridlines and labels
    # fig.suptitle(f'Error time with correct event \n and Cumulative Frequency of Detected Events (Total: {len(TP_events)} Events)')

    # Add legends for both y-axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    combined_lines = lines_1 + lines_2
    combined_labels = labels_1 + labels_2
    ax1.legend(combined_lines, combined_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True, ncol=len(combined_labels))
    # ax1.legend(ncols=1)


    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f"{save_dir}/histogram_seconds.png")
    plt.show()

def plot_histogram_of_offsets2(data, save_dir):
    # plt.rcParams['font.family'] = 'Meiryo UI'
    Z_max = 4
    D_max_values = range(1, 40)
    
    ground_truth = np.array(data["label"])
    ground_event = make_event(ground_truth, label=[2, 3])
    
    long_predictions = np.array([int(np.argmax(out[["usual", "falling", "fall"]])) for i, out in data.iterrows()])
    pred_event = make_event(long_predictions, label=[2])
    pred_event = concat_pred_event(pred_event)
    pred_event = [p for p in pred_event if abs(p[0] - p[1]) >= Z_max]

    offsets_label_2 = []
    offsets_label_3 = []
    matched_gt_indices = set()
    matched_pred_indices = set()
    TP_events = []
    
    for D_max in D_max_values:
        for g_idx, (g_start, g_end, ge) in enumerate(ground_event):
            for p_idx, (p_start, p_end, pe) in enumerate(pred_event):
                if p_idx not in matched_pred_indices and g_idx not in matched_gt_indices:
                    if abs(g_start - p_start) <= D_max and abs(g_end - p_end) <= (5 * D_max):
                        matched_gt_indices.add(g_idx)
                        matched_pred_indices.add(p_idx)
                        TP_events.append([p_start, p_end, ge])
                        offset_in_seconds = (p_start - g_start) / 8  # Convert offset from frames to seconds
                        if ge == 2:
                            offsets_label_2.append(offset_in_seconds)
                        elif ge == 3:
                            offsets_label_3.append(offset_in_seconds)

    # Set bins to 1-second increments, with x-axis range from -5 to 5
    bins = np.arange(-5, 6, 1)

    fig, ax1 = plt.subplots()

    # Plot the stacked histogram (left y-axis)
    counts_3, _, _ = ax1.hist(offsets_label_3, bins=bins, edgecolor='black', color='seagreen', align='mid', label='Emergency', alpha=0.7)
    counts_2, _, _ = ax1.hist(offsets_label_2, bins=bins, edgecolor='black', color='royalblue', align='mid', label='Caution', alpha=0.7, bottom=counts_3)
    ax1.set_xlabel('Error time with correct event (seconds)', fontsize=16)
    ax1.set_ylabel('TP Event Count', fontsize=16)
    ax1.set_xticks(np.arange(-5, 6, 1))  # Set x-ticks to 1-second increments
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([0, 8])
    ax1.grid(True, which='both', axis='x', linestyle='--', linewidth=0.7)

    # Plot cumulative frequency (right y-axis)
    ax2 = ax1.twinx()
    cumulative_counts = np.cumsum(counts_2 + counts_3)
    cumulative_freq = cumulative_counts / len(ground_event)  # Normalize to total correct events
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax2.plot(bin_centers, cumulative_freq, color='navy', marker='o', label='Cumulative Frequency')
    ax2.set_ylabel('Cumulative Frequency', fontsize=16)
    ax2.set_ylim([0, 1.1])

    ax1.set_xticklabels([f'{x:.0f}' for x in ax1.get_xticks()], fontsize=14)
    ax1.set_yticklabels([f'{y:.0f}' for y in ax1.get_yticks()], fontsize=14)
    ax2.set_yticklabels([f'{y:.1f}' for y in ax2.get_yticks()], fontsize=14)



    # Add legends for both y-axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    combined_lines = lines_1 + lines_2
    combined_labels = labels_1 + labels_2
    ax1.legend(combined_lines, combined_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True, ncol=len(combined_labels), fontsize=14)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f"{save_dir}/stacked_histogram_seconds.png")
    plt.show()



def fall_slip_matrix(data, TP_events, save_dir):
    # Step 2: Initialize sequences and labels
    sequence = data.iloc[:, :3].values
    sequence_label = np.argmax(sequence, axis=1)
    
    thresholds = [8, 16, 24, 32]
    cm_list = []

    for threshold in thresholds:
        sequences = []
        labels = []

        # Step 3: Process each TP_event to determine slip or fall
        for ps, pg, true_label in TP_events:
            # Check the sequence of the past 40 frames before ps
            start_idx = max(0, ps - 40)
            past_sequence = sequence_label[start_idx:ps]

            # Count the number of frames where 'falling' (2nd column) > 0.4
            falling_count = np.sum([1 for i in past_sequence if i == 1])


            # Determine slip or fall based on the falling count
            if falling_count >= threshold:
                pred_label = 2  # Slip
            else:
                pred_label = 3  # Fall

            sequences.append(sequence[ps:pg+1])
            labels.append((true_label, pred_label))

        # Step 4: Create confusion matrix
        true_labels = [label[0] for label in labels]
        pred_labels = [label[1] for label in labels]

        cm = confusion_matrix(true_labels, pred_labels, labels=[2, 3])
        cm_list.append(cm)

    # Step 5: Plot all confusion matrices in one figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    label_names = ['slip', 'fall']

    for idx, cm in enumerate(cm_list):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, cbar=False, ax=axes[idx])
        axes[idx].set_title(f'Threshold: {thresholds[idx]}')
        axes[idx].set_xlabel('Predicted Labels')
        axes[idx].set_ylabel('True Labels')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrices.png")
    plt.show()

    # Calculate accuracy for slip and fall for each threshold
    for idx, cm in enumerate(cm_list):
        slip_correct = cm[0, 0]  # True Positive for slip
        slip_total = cm[0, 0] + cm[0, 1]  # Total actual slip events
        fall_correct = cm[1, 1]  # True Positive for fall
        fall_total = cm[1, 0] + cm[1, 1]  # Total actual fall events

        slip_accuracy = slip_correct / slip_total if slip_total > 0 else 0
        fall_accuracy = fall_correct / fall_total if fall_total > 0 else 0

        print(f'Threshold: {thresholds[idx]}')
        print(f'  Slip Accuracy: {slip_accuracy:.2f}')
        print(f'  Fall Accuracy: {fall_accuracy:.2f}')

    return cm_list


def fall_slip_matrix2(data, TP_events, save_dir, falling_th=0.4):
    # Step 2: Initialize sequences and labels
    sequence = data.iloc[:, :3].values
    sequence_label = np.argmax(sequence, axis=1)
    
    thresholds = [8, 16, 24, 32]
    cm_list = []

    for threshold in thresholds:
        sequences = []
        labels = []

        # Step 3: Process each TP_event to determine slip or fall
        for ps, pg, true_label in TP_events:
            # Check the sequence of the past 40 frames before ps
            start_idx = max(0, ps - 40)
            past_sequence = sequence[start_idx:ps]

            # Count the number of frames where 'falling' (2nd column) > 0.4
            falling_count = np.sum(past_sequence[:, 1] > falling_th)

            # Determine slip or fall based on the falling count
            if falling_count >= threshold:
                pred_label = 2  # Slip
            else:
                pred_label = 3  # Fall

            sequences.append(sequence[ps:pg+1])
            labels.append((true_label, pred_label))

        # Step 4: Create confusion matrix
        true_labels = [label[0] for label in labels]
        pred_labels = [label[1] for label in labels]

        cm = confusion_matrix(true_labels, pred_labels, labels=[2, 3])
        cm_list.append(cm)

    # Step 5: Plot all confusion matrices in one figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    label_names = ['slip', 'fall']

    for idx, cm in enumerate(cm_list):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, cbar=False, ax=axes[idx])
        axes[idx].set_title(f'Threshold: {thresholds[idx]}')
        axes[idx].set_xlabel('Predicted Labels')
        axes[idx].set_ylabel('True Labels')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrices2.png")
    plt.show()

    # Calculate accuracy for slip and fall for each threshold
    for idx, cm in enumerate(cm_list):
        slip_correct = cm[0, 0]  # True Positive for slip
        slip_total = cm[0, 0] + cm[0, 1]  # Total actual slip events
        fall_correct = cm[1, 1]  # True Positive for fall
        fall_total = cm[1, 0] + cm[1, 1]  # Total actual fall events

        slip_accuracy = slip_correct / slip_total if slip_total > 0 else 0
        fall_accuracy = fall_correct / fall_total if fall_total > 0 else 0

        print(f'Threshold: {thresholds[idx]}')
        print(f'  Slip Accuracy: {slip_accuracy:.2f}')
        print(f'  Fall Accuracy: {fall_accuracy:.2f}')

    return cm_list


import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, confusion_matrix
def create_dataset_from_events(data, TP_event):
    X = []
    y = []

    for event in TP_event:
        start_frame, end_frame, true_label = event

        # 開始フレームから40フレーム分のデータを抽出
        if 0 <= start_frame - 40:
            sample = data.iloc[start_frame-40:start_frame][['usual', 'falling', 'fall']].values
            X.append(sample)
            y.append(true_label)
    
    return np.array(X), np.array(y)

# HMMの学習
def train_hmm(X, n_components=2):
    if len(X) == 0:  # Xが空のリストの場合
        return None

    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
    X_concat = np.concatenate(X)  # 全ての時系列データを連結
    lengths = [x.shape[0] for x in X]  # 各サンプルの長さを保持
    model.fit(X_concat, lengths)
    return model

# HMMを使用して予測を行う
def predict_hmm(fall_model, slip_model, X):
    y_pred = []
    for x in X:
        if fall_model is None:
            fall_prob = float('-inf')
        else:
            fall_prob = fall_model.score(x)
        
        if slip_model is None:
            slip_prob = float('-inf')
        else:
            slip_prob = slip_model.score(x)
        
        y_pred.append(3 if fall_prob > slip_prob else 2)  # fall: 3, slip: 2 としてラベル付け
    return np.array(y_pred)

def eval_HMM(train_data, TP_event_train, val_data, TP_event_val, save_dir):
    # トレーニングデータセットの作成
    train_X, train_y = create_dataset_from_events(train_data, TP_event_train)
    # 評価データセットの作成
    val_X, val_y = create_dataset_from_events(val_data, TP_event_val)

    # データの形状を確認
    print("Train X shape:", train_X.shape)
    print("Train y shape:", train_y.shape)
    print("Val X shape:", val_X.shape)
    print("Val y shape:", val_y.shape)

    # トレーニングデータを使ってHMMのトレーニング
    fall_X = [x for x, y in zip(train_X, train_y) if y == 3]
    slip_X = [x for x, y in zip(train_X, train_y) if y == 2]

    fall_hmm = train_hmm(fall_X, n_components=2)
    slip_hmm = train_hmm(slip_X, n_components=4)

    # トレーニングデータに対する予測と精度計算
    train_y_pred = predict_hmm(fall_hmm, slip_hmm, train_X)
    train_accuracy = accuracy_score(train_y, train_y_pred)
    print(f"Train Accuracy: {train_accuracy:.2f}")

    # 検証データに対する予測と精度計算
    val_y_pred = predict_hmm(fall_hmm, slip_hmm, val_X)
    val_accuracy = accuracy_score(val_y, val_y_pred)
    print(f"Validation Accuracy: {val_accuracy:.2f}")

    # 混同行列の作成
    train_cm = confusion_matrix(train_y, train_y_pred, labels=[2, 3])
    val_cm = confusion_matrix(val_y, val_y_pred, labels=[2, 3])

    # 混同行列のリスト
    cm_list = [train_cm, val_cm]
    thresholds = ['Train Data', 'Validation Data']

    # Step 5: 混同行列を一つの図としてプロット
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.ravel()
    label_names = ['slip', 'fall']

    for idx, cm in enumerate(cm_list):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, cbar=False, ax=axes[idx])
        axes[idx].set_title(f'{thresholds[idx]}')
        axes[idx].set_xlabel('Predicted Labels')
        axes[idx].set_ylabel('True Labels')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/HMM.png")
    plt.show()