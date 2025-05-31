import torch
import torch.nn as nn
import torchvision.models as models
device = "cuda" if torch.cuda.is_available() else "cpu"
class ConvLSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=100, num_layers=2,
                 feature_col=7, resnet_output_dim=4, motion_f=False, motion_ratio_dim=0, image_f=True):
        super(ConvLSTM, self).__init__()
        # Load pre-trained ResNet model
        self.image_f = image_f
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(
                *(list(self.resnet.children())[:-2]),  # Exclude the avgpool and fc layers
                nn.AdaptiveAvgPool2d((1, 1)),  # Add AdaptiveAvgPool to get (512, 1, 1) output
                nn.Flatten(),  # Flatten to get (512,)
                nn.Linear(512, resnet_output_dim),  # Reduce dimensions
                nn.ReLU(),
                nn.Dropout(0.5)
            ).to(device)
        if motion_f:
            self.resnet2 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.resnet2 = torch.nn.Sequential(
                *(list(self.resnet2.children())[:-2]),  # Exclude the avgpool and fc layers
                nn.AdaptiveAvgPool2d((1, 1)),  # Add AdaptiveAvgPool to get (512, 1, 1) output
                nn.Flatten(),  # Flatten to get (512,)
                nn.Linear(512, resnet_output_dim),  # Reduce dimensions
                nn.ReLU(),
                nn.Dropout(0.5)
            ).to(device)
        self.motion_ratio_dim = motion_ratio_dim
        # LSTM
        self.motion_f = motion_f
        self.feature_col = feature_col
        lstm_input_size = 0
        if image_f:
            lstm_input_size += resnet_output_dim
        if feature_col:
            lstm_input_size += feature_col
        self.lstm = torch.nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim,
                                  num_layers=num_layers, batch_first=True).to(device)
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim + int(motion_f) * resnet_output_dim + motion_ratio_dim, num_classes)

    def forward(self, x, features):
        if self.image_f:
            batch_size, seq_length, h, w, c = x.size()
            c_in = x.view(batch_size * seq_length, c, h, w) # torch.Size([256,3,120,160])

            # Extract features using ResNet
            c_out = self.resnet(c_in)
            resnet_out = c_out.view(batch_size, seq_length, -1)

        if self.feature_col and self.image_f:
            # featuresをリストに追加
            features = features.squeeze(2)
            combined_features = torch.cat([features, resnet_out], dim=2)
        elif self.image_f:
            combined_features = resnet_out
        elif self.feature_col:
            features = features.squeeze(2)
            combined_features = features

        # LSTM
        r_out, (h_n, c_n) = self.lstm(combined_features)

        if self.motion_f:
            # GPU上でモーションフロー画像を生成
            motion_flow_out = self.generate_motion_flow_rgb(x)
            # motion_flow_out = motion_flow_out.unsqueeze(1).repeat(1, 3, 1, 1)  # [batch_size, 3, h, w]
            # motion_flow_out = self.resnet2(motion_flow_out)
            # LSTMの出力とmotion flowからの出力を結合して全結合層に入力
            combined_output = torch.cat([r_out[:, -1, :], motion_flow_out], dim=1)
            output = self.fc(combined_output)
            return output

        if self.motion_ratio_dim:
            motion_flow = self.generate_motion_flow(x)
            motion_ratio = self.calculate_value_ratios(motion_flow, dim=self.motion_ratio_dim)
            combined_output = torch.cat([r_out[:, -1, :], motion_ratio], dim=1)
            output = self.fc(combined_output)
            return output

        # Classification
        output = self.fc(r_out[:, -1, :])

        return output

    def generate_motion_flow(self, x):
        # x: [batch_size, seq_length, c, h, w]
        grayscale_images = x[:, :, 0, :, :]  # Extract the first channel, [batch_size, seq_length, h, w]
        grayscale_images = grayscale_images.float() / 255.0
        scaled_images = grayscale_images / 16.0
        motion_flow = scaled_images.sum(dim=1)  # Sum along the sequence dimension, [batch_size, h, w]
        motion_flow = torch.clamp(motion_flow, 0, 1.0) * 255.0
        motion_flow = motion_flow.to(torch.float32)
        return motion_flow

    def calculate_value_ratios(self, motion_flow_image, dim=4):
        # motion_flow_image: [batch_size, h, w]
        batch_size, h, w = motion_flow_image.size()

        # Flatten the image to process pixel values
        motion_flow_image = motion_flow_image.view(batch_size, -1)

        # Prepare bins for the range [0, 255]
        bin_ranges = torch.linspace(0, 255, steps=dim + 1, device=motion_flow_image.device)

        # Initialize binned counts
        binned_counts = torch.zeros(batch_size, dim, device=motion_flow_image.device)

        # Calculate counts for each bin
        for i in range(dim):
            binned_counts[:, i] = torch.sum((motion_flow_image >= bin_ranges[i]) & 
                                            (motion_flow_image < bin_ranges[i + 1]), dim=1)

        # Normalize the counts
        total_counts = torch.sum(binned_counts, dim=1, keepdim=True)
        ratio_list = binned_counts / total_counts

        return ratio_list

    def generate_motion_flow_rgb(self, x):    
        batch_size, seq_length, c, h, w = x.size()
        
        grayscale_images = x[:, :, 0, :, :].to(device)  # Extract the first channel, [batch_size, seq_length, h, w]
        scaled_images = grayscale_images.float() / 16.0  # Normalize and scale images
    
        motion_flow = torch.zeros(batch_size, 3, h, w, device=x.device, dtype=torch.float32)
    
        for i in range(seq_length):
            r, g, b = colors[i]
            colored_image = torch.stack([scaled_images[:, i] * b, scaled_images[:, i] * g, scaled_images[:, i] * r], dim=1)  # [batch_size, 3, h, w]
            
            # Create a mask where the current image is non-zero
            mask = colored_image > 0
            mask = mask.any(dim=1, keepdim=True)  # Combine across the color channels
            
            # Apply the mask: set the masked regions to 0 in motion_flow and overwrite them with colored_image
            motion_flow = torch.where(mask, colored_image, motion_flow)
    
        motion_flow = torch.clamp(motion_flow, 0, 255)  # Ensure values are in the correct range
        # # Convert motion_flow to match r_out dimensions
        motion_flow = self.resnet2(motion_flow)
        motion_flow = torch.flatten(motion_flow, start_dim=1)  # Flatten to [batch_size, -1]

        return motion_flow
