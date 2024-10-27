import torch
import torch.nn as nn
from flowery import flowery

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    @flowery(verbose=True)
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.fc1(x)
        print(f"After fc1: {x.shape}")
        x = torch.relu(x)
        x = self.fc2(x)
        print(f"After fc2: {x.shape}")
        return x

# Example usage
if __name__ == '__main__':
    model = SimpleMLP(10, 5, 2)
    input_tensor = torch.randn(1, 10)
    output = model(input_tensor)
    print(f"Output: {output}")

