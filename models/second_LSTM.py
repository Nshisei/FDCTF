import torch
import torch.nn as nn

class SecondLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, output_size=3, dropout=0.3):
        super(SecondLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, main_prob, sub_tasks_list):
        # main_prob: (batch_size, window_size, 2)  # 転倒確率の2値出力 (各タイムステップでの予測)
        # sub_tasks_list: リスト [(batch_size, window_size, n1), (batch_size, window_size, n2), ...]  # 補助タスクの出力

        # 🔄 空チェックを追加: sub_tasks_listが空ならダミーのゼロテンソルを追加
        if len(sub_tasks_list) > 0:
            sub_tasks_concat = torch.cat(sub_tasks_list, dim=-1)  # (batch_size, window_size, total_sub_task_dim)
            print(f"sub_tasks_concat.size(): {sub_tasks_concat.size()}")
            combined_input = torch.cat([main_prob, sub_tasks_concat], dim=-1)  # (batch_size, window_size, input_size)
        else:
            combined_input = main_prob  # サブタスクがない場合は main_prob だけ

        # LSTMへの入力サイズを確認
        print(f"combined_input.size(): {combined_input.size()}")

        # LSTMの処理
        lstm_out, _ = self.lstm(combined_input)  # (batch_size, window_size, hidden_size)
        output = self.fc(lstm_out[:, -1, :])     # (batch_size, output_size)
        return output
