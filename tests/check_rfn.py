import numpy as np
from timeomni_vl.bitsi import RobustFidelityNormalizer

x = np.random.randn(100, 3).astype(np.float32) * 50 + 100
rfn = RobustFidelityNormalizer()
x_norm, stats = rfn.fit_transform(x)
x_hat = rfn.inverse_transform(x_norm, stats)
print("max error:", np.max(np.abs(x - x_hat)))
print("mean error:", np.mean(np.abs(x - x_hat)))
print("x range:", x.min(), x.max())
print("x_norm range:", x_norm.min(), x_norm.max())
