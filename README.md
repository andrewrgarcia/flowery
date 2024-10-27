# flowery 🌸

A quick way to hide print statements during code runs. Perfect for debugging machine learning models or any project with too much output.

## Features
- Suppress or allow print statements with a simple decorator.
- Helpful for ML models and complex algorithms where selective logging is necessary.
- No dependencies – just one Python file to include in your project.

## Usage

1. **Download the [`flowery/flower_code.py`](flowery/flower_code.py) file** and place it in your project directory.

2. **Example 1: Debugging in PyTorch Models**
```python
import torch
import torch.nn as nn
from flower_code import flowery

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
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

model = NeuralNetwork(10, 5, 2)
input_tensor = torch.randn(1, 10)
output = model(input_tensor)
print(f"Output: {output}")
```

3. **Example 2: Debugging Complex Algorithms**
```python
from flower_code import flowery

@flowery(verbose=True)
def fibonacci(n, memo={}):
    print(f"Calculating fibonacci({n})")
    if n in memo:
        print(f"Returning memoized value for {n}")
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    print(f"Computed fibonacci({n}): {memo[n]}")
    return memo[n]

print(f"Result: {fibonacci(5)}")
```

## Recommendations

If you'd like to download the complete repository (including examples and tests):
```bash
git clone https://github.com/your-username/flowery.git
cd flowery
```

## Testing

You can run the tests with:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License.

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

## Author

Developed by Andrew R. Garcia.

