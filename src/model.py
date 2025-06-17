import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    """
    Deep Q-Network optimized for Tetris Gymnasium environments
    Supports both image observations and feature vector observations
    """

    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.n_actions = action_space.n

        # Determine input type based on observation space
        if len(obs_space.shape) == 3:  # Image observation (H, W, C)
            self._init_conv_network(obs_space)
        elif len(obs_space.shape) == 1:  # Feature vector observation
            self._init_fc_network(obs_space)
        else:
            raise ValueError(
                f"Unsupported observation space shape: {obs_space.shape}")

    def _init_conv_network(self, obs_space):
        """Initialize convolutional network for image observations"""
        self.network_type = "conv"
        h, w, c = obs_space.shape

        # Convolutional layers
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(obs_space.shape)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.n_actions)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        print(
            f"Initialized CNN-DQN: {conv_out_size} -> 512 -> 256 -> {self.n_actions}")

    def _init_fc_network(self, obs_space):
        """Initialize fully connected network for feature vector observations"""
        self.network_type = "fc"
        input_size = obs_space.shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.n_actions)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        print(
            f"Initialized FC-DQN: {input_size} -> 512 -> 256 -> 128 -> {self.n_actions}")

    def _get_conv_output_size(self, shape):
        """Calculate the output size of convolutional layers"""
        h, w, c = shape
        # Simulate forward pass through conv layers
        x = torch.zeros(1, c, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.numel()

    def forward(self, x):
        """Forward pass through the network"""
        if self.network_type == "conv":
            return self._forward_conv(x)
        else:
            return self._forward_fc(x)

    def _forward_conv(self, x):
        """Forward pass for convolutional network"""
        # Ensure correct input format (batch_size, channels, height, width)
        if len(x.shape) == 3:  # Add batch dimension
            x = x.unsqueeze(0)

        # If input is (batch, height, width, channels), transpose to (batch, channels, height, width)
        if x.shape[-1] <= 4:  # Assuming channels is the smallest dimension
            x = x.permute(0, 3, 1, 2)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten for fully connected layers (ensure contiguous memory)
        x = x.contiguous().view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def _forward_fc(self, x):
        """Forward pass for fully connected network"""
        # Ensure batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture for improved value estimation
    Particularly useful for Tetris where not all actions may be equally important
    """

    def __init__(self, obs_space, action_space):
        super(DuelingDQN, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.n_actions = action_space.n

        if len(obs_space.shape) == 3:  # Image observation
            self._init_conv_features(obs_space)
        elif len(obs_space.shape) == 1:  # Feature vector observation
            self._init_fc_features(obs_space)
        else:
            raise ValueError(
                f"Unsupported observation space shape: {obs_space.shape}")

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )

        print(
            f"Initialized Dueling DQN with {self.feature_size} features -> Value + {self.n_actions} Advantages")

    def _init_conv_features(self, obs_space):
        """Initialize convolutional feature extractor"""
        self.network_type = "conv"
        h, w, c = obs_space.shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Adaptive pooling for consistent output size
            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.feature_size = 64 * 7 * 7

    def _init_fc_features(self, obs_space):
        """Initialize fully connected feature extractor"""
        self.network_type = "fc"
        input_size = obs_space.shape[0]

        self.features = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.feature_size = 256

    def forward(self, x):
        """Forward pass through dueling architecture"""
        # Handle input format
        if self.network_type == "conv":
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            if x.shape[-1] <= 4:  # Channels last to channels first
                x = x.permute(0, 3, 1, 2)
        else:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)

        # Extract features
        features = self.features(x)
        # After permute/features, tensor may be non-contiguous → use reshape
        features = features.reshape(features.size(0), -1)

        # Compute value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage (dueling architecture)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


def create_model(obs_space, action_space, model_type="dqn"):
    """
    Factory function to create appropriate model based on requirements
    
    Args:
        obs_space: Environment observation space
        action_space: Environment action space
        model_type: "dqn" or "dueling_dqn"
    
    Returns:
        Initialized model
    """
    if model_type.lower() == "dueling_dqn":
        return DuelingDQN(obs_space, action_space)
    else:
        return DQN(obs_space, action_space)


# Model testing function
def test_model():
    """Test model creation and forward pass"""
    import gymnasium as gym

    print("Testing model architectures...")

    # Test with dummy spaces
    obs_spaces = [
        gym.spaces.Box(low=0, high=255, shape=(
            84, 84, 4), dtype=np.uint8),  # Image
        gym.spaces.Box(low=-1, high=1, shape=(200,),
                       dtype=np.float32),      # Features
    ]

    action_space = gym.spaces.Discrete(7)  # Typical Tetris action space

    for i, obs_space in enumerate(obs_spaces):
        print(f"\nTest {i+1}: Observation space {obs_space.shape}")

        # Test both model types
        for model_type in ["dqn", "dueling_dqn"]:
            model = create_model(obs_space, action_space, model_type)

            # Create dummy input
            dummy_input = torch.randn((1,) + obs_space.shape)

            # Forward pass
            output = model(dummy_input)
            print(
                f"  {model_type}: Input {dummy_input.shape} -> Output {output.shape}")
            assert output.shape == (
                1, action_space.n), f"Expected {(1, action_space.n)}, got {output.shape}"

    print("✅ All model tests passed!")


if __name__ == "__main__":
    test_model()
