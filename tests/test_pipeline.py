from timeomni_vl.config import ConfigManager
from timeomni_vl.data.pipeline import build_data_pipeline
from timeomni_vl.bitsi import RobustFidelityNormalizer, TS2IConverter
from timeomni_vl.tasks.understanding import UnderstandingTaskGenerator
from timeomni_vl.tasks.generation import GenerationTaskGenerator
import numpy as np

cfg = ConfigManager("configs/data.yaml").get_data_config()
pipeline = build_data_pipeline(cfg)
samples = pipeline.run()
print(f"Generated {len(samples['train'])} train samples")

sample = samples['train'][0]
print(f"Sample context shape: {sample.context.shape}, target shape: {sample.target.shape}")

rfn = RobustFidelityNormalizer()
ctx_norm, stats = rfn.fit_transform(sample.context)
ts2i = TS2IConverter(frequency=cfg.frequency, image_size=448)
image = ts2i.convert(ctx_norm, variable_names=None)
print(f"TS-image size: {image.size}")
image.save("outputs/test_train_ts_image.png")

qa_gen = UnderstandingTaskGenerator(frequency=cfg.frequency, image_size=448)
qas = qa_gen.generate_all(sample)
print(f"Generated {len(qas)} QA tasks")
for qa in qas:
    print(f"  {qa['task_id']}: {qa['question']}")

gen_gen = GenerationTaskGenerator(frequency=cfg.frequency, context_days=cfg.context_days, forecast_days=cfg.forecast_days)
cot = gen_gen.generate_cot(qas)
print(f"CoT preview:\n{cot[:200]}...")
