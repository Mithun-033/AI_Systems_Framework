AI Systems Framework
=====================

The framework currently provides:

- Custom Triton-backed matrix multiplication and linear layers in `src/utils`
- Three Triton-compatible optimizers: SGD, Adam, and AdamW
- Three Triton losses: MSE, CrossEntropy, and BCEWithLogits

```python
import torch

from optimizers import TritonAdamW
from losses import TritonCrossEntropyLoss

model = torch.nn.Linear(16, 10)
optimizer = TritonAdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = TritonCrossEntropyLoss()
```