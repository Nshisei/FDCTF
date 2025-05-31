import torch
import torch.nn as nn
import torchvision.models as models
device = "cuda" if torch.cuda.is_available() else "cpu"
class ConvLSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=100, num_layers=2,
                 feature_col=7, resnet_output_dim=4, sub_task_name_dict={}, resnet_fix_params=0):
        super(ConvLSTM, self).__init__()
        # Load pre-trained ResNet model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(
                *(list(self.resnet.children())[:-2]),  # Exclude the avgpool and fc layers
                nn.AdaptiveAvgPool2d((1, 1)),  # Add AdaptiveAvgPool to get (512, 1, 1) output
                nn.Flatten(),  # Flatten to get (512,)
                nn.Linear(512, resnet_output_dim),  # Reduce dimensions
                nn.ReLU(),
                nn.Dropout(0.5)
            ).to(device)
        self.feature_col = feature_col
        lstm_input_size = resnet_output_dim
        if feature_col:
            lstm_input_size += feature_col
        self.lstm = torch.nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim,
                                  num_layers=num_layers, batch_first=True).to(device)
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        # 補助タスク用のfc layers

        self.sub_task_fcs = nn.ModuleList([torch.nn.Linear(hidden_dim, len(values)) for name, values in sub_task_name_dict.items()])

        # 全層のrequires_gradをFalseに設定
        if resnet_fix_params > 0:
            for param in self.resnet.parameters():
                param.requires_grad = False

            # 最後から3つの層だけを更新対象に設定
            for layer in list(self.resnet.children())[-resnet_fix_params:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x, features):

        batch_size, seq_length, h, w, c = x.size()
        c_in = x.view(batch_size * seq_length, c, h, w) # torch.Size([256,3,120,160])

        # Extract features using ResNet
        c_out = self.resnet(c_in)
        resnet_out = c_out.view(batch_size, seq_length, -1)

        if self.feature_col:
            # featuresをリストに追加
            features = features.squeeze(2)
            combined_features = torch.cat([features, resnet_out], dim=2)
        else:
            combined_features = resnet_out
 
        # LSTM
        r_out, (h_n, c_n) = self.lstm(combined_features)

        # Main task classification
        output = self.fc(r_out[:, -1, :])

        # 補助タスク
        if len(self.sub_task_fcs)>0:
            sub_outputs = [fc(r_out[:, -1, :]) for fc in self.sub_task_fcs]
            return output, sub_outputs
        else:
            return output, None