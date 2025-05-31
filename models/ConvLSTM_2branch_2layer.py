import torch
import torch.nn as nn
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

class ConvLSTM_2layer(nn.Module):
    def __init__(self, num_classes_fall=2, num_classes_type=3, hidden_dim=100, num_layers=2,
                 feature_col=7, resnet_output_dim=4, sub_task_name_dict={}, resnet_fix_params=0):
        super(ConvLSTM_2layer, self).__init__()
        
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

        # First LSTM: Frame-wise fall detection and auxiliary tasks
        self.lstm1 = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim,
                             num_layers=num_layers, batch_first=True).to(device)
        
        self.fc_fall = nn.Linear(hidden_dim, num_classes_fall)  # 2-class fall detection
        
        self.sub_task_fcs = nn.ModuleList([nn.Linear(hidden_dim, len(values)) for name, values in sub_task_name_dict.items()])
        
        # Second LSTM: Sequence classification (emergency, caution, unknown)
        self.lstm2 = nn.LSTM(input_size=hidden_dim + len(self.sub_task_fcs), hidden_size=hidden_dim,
                             num_layers=1, batch_first=True).to(device)
        
        self.fc_type = nn.Linear(hidden_dim, num_classes_type)  # 3-class sequence classification
        
        # Freeze ResNet layers if required
        if resnet_fix_params > 0:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for layer in list(self.resnet.children())[-resnet_fix_params:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x, features):
        batch_size, seq_length, h, w, c = x.size()
        c_in = x.view(batch_size * seq_length, c, h, w)  # Reshape for CNN
        
        # Feature extraction
        c_out = self.resnet(c_in)
        resnet_out = c_out.view(batch_size, seq_length, -1)

        if self.feature_col:
            features = features.squeeze(2)
            combined_features = torch.cat([features, resnet_out], dim=2)
        else:
            combined_features = resnet_out
        
        # First LSTM for frame-wise fall detection
        r_out1, _ = self.lstm1(combined_features)
        fall_output = self.fc_fall(r_out1)  # Output for each frame
        
        # Auxiliary tasks
        sub_outputs = [fc(r_out1) for fc in self.sub_task_fcs] if self.sub_task_fcs else []
        
        # Concatenate fall_output and sub_outputs for second LSTM
        second_lstm_input = torch.cat([r_out1] + sub_outputs, dim=2)  # Keep frame-wise outputs
        
        # Second LSTM for sequence classification
        r_out2, _ = self.lstm2(second_lstm_input)
        type_output = self.fc_type(r_out2[:, -1, :])  # Only last frame output
        
        return (fall_output, type_output), sub_outputs
