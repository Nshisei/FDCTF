import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvLSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=100, num_layers=2,
                 feature_col=7, cnn_output_dim=4, sub_task_name_dict={}):
        super().__init__()
        # シンプルなCNNの定義
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
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 3 * 5, cnn_output_dim),  # Adjusted for 5 pooling layers
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.feature_col = feature_col
        lstm_input_size = cnn_output_dim
        if feature_col:
            lstm_input_size += feature_col
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        # Fully connected layers for auxiliary tasks
        self.sub_task_fcs = nn.ModuleList([nn.Linear(hidden_dim, len(values)) for name, values in sub_task_name_dict.items()])


    def forward(self, x, features):
        batch_size, seq_length, h, w, c = x.size()
        c_in = x.view(batch_size * seq_length, c, h, w)#.permute(0, 3, 1, 2)  # (B*T, C, H, W)に変換

        # CNNによる特徴抽出
        cnn_out = self.cnn(c_in)
        cnn_out = cnn_out.view(batch_size, seq_length, -1)

        if self.feature_col:
            features = features.squeeze(2)
            combined_features = torch.cat([features, cnn_out], dim=2)
        else:
            combined_features = cnn_out

        # LSTM
        r_out, (h_n, c_n) = self.lstm(combined_features)

        # メインタスクの分類
        output = self.fc(r_out[:, -1, :])

        # 補助タスク
        if len(self.sub_task_fcs) > 0:
            sub_outputs = [fc(r_out[:, -1, :]) for fc in self.sub_task_fcs]
            return output, sub_outputs
        else:
            return output, None