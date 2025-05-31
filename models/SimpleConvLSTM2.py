import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvLSTM2(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=100, num_layers=1,
                 feature_col=7, cnn_output_dim=4, sub_task_name_dict={}):
        super().__init__()
        # CNNを3層に変更
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # CNN出力の次元を計算
        dummy_input = torch.zeros(1, 3, 120, 160)  # 入力のサンプルサイズを幅160高さ120に変更
        with torch.no_grad():
            cnn_output_size = self.cnn(dummy_input).shape[1]

        self.fc_cnn = nn.Sequential(
            nn.Linear(cnn_output_size, cnn_output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.feature_col = feature_col
        lstm_input_size = cnn_output_dim
        if feature_col:
            lstm_input_size += feature_col

        # LSTMの1層目（共通）
        self.lstm_shared = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        # LSTMの2層目（タスクごと）
        self.lstm_task_specific = nn.ModuleList([
            nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
            for _ in sub_task_name_dict
        ])

        # メインタスクのFC層
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 補助タスクのFC層（2層に変更）
        self.sub_task_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, len(values))
            ) for name, values in sub_task_name_dict.items()
        ])

    def forward(self, x, features):
        batch_size, seq_length, h, w, c = x.size()
        c_in = x.view(batch_size * seq_length, c, h, w)  # (B*T, C, H, W)

        # CNNによる特徴抽出
        cnn_out = self.cnn(c_in)
        cnn_out = self.fc_cnn(cnn_out)
        cnn_out = cnn_out.view(batch_size, seq_length, -1)

        if self.feature_col:
            features = features.squeeze(2)
            combined_features = torch.cat([features, cnn_out], dim=2)
        else:
            combined_features = cnn_out

        # LSTMの1層目（共通）
        r_out, (h_n, c_n) = self.lstm_shared(combined_features)

        # メインタスクの分類
        main_output = self.fc(r_out[:, -1, :])

        # 補助タスクの出力
        if len(self.sub_task_fcs) > 0:
            sub_outputs = []
            for i, (lstm, fc) in enumerate(zip(self.lstm_task_specific, self.sub_task_fcs)):
                task_r_out, _ = lstm(r_out)
                sub_outputs.append(fc(task_r_out[:, -1, :]))
            return main_output, sub_outputs
        else:
            return main_output, None
