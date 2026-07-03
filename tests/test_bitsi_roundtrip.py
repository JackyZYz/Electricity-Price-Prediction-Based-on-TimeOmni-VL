import numpy as np

from timeomni_vl.bitsi import BiTSIValidator, I2TSConverter, RobustFidelityNormalizer, TS2IConverter

np.random.seed(42)
x = np.random.randn(96 * 7, 3).astype(np.float32) * 100 + 300

rfn = RobustFidelityNormalizer()
x_norm, stats = rfn.fit_transform(x)
print(f"x_norm range: [{x_norm.min():.3f}, {x_norm.max():.3f}]")

ts2i = TS2IConverter(frequency=96, image_size=448)
image = ts2i.convert(x_norm)
print(f"TS-image size: {image.size}")
image.save("outputs/test_ts_image.png")

i2ts = I2TSConverter(frequency=96, image_size=448)
x_hat_norm = i2ts.convert(image, n_vars=3, target_length=x.shape[0], stats=stats)
x_hat = rfn.inverse_transform(x_hat_norm, stats)
print(f"Round-trip MAE: {np.mean(np.abs(x - x_hat)):.4f}")

validator = BiTSIValidator(rfn, ts2i, i2ts)
metrics = validator.validate(x)
print(f"Validator metrics: {metrics}")
