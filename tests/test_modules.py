import numpy as np

from timeomni_vl.bitsi import BiTSIValidator, I2TSConverter, RobustFidelityNormalizer, TS2IConverter
from timeomni_vl.data.sampler import RollingWindowSampler, Sample
from timeomni_vl.evaluation.generation_metrics import GenerationMetrics
from timeomni_vl.models import build_adapter
from timeomni_vl.tasks.generation import GenerationEvaluator
from timeomni_vl.tasks.understanding import UnderstandingEvaluator, UnderstandingTaskGenerator
from timeomni_vl.utils.text import extract_between_tags


def test_rfn_roundtrip():
    x = np.random.randn(100, 3).astype(np.float32) * 50 + 100
    rfn = RobustFidelityNormalizer()
    x_norm, stats = rfn.fit_transform(x)
    x_hat = rfn.inverse_transform(x_norm, stats)
    assert x_hat.shape == x.shape
    assert np.mean(np.abs(x - x_hat)) < np.std(x) * 2


def test_ts2i_shape():
    x = np.random.randn(96 * 7, 2).astype(np.float32)
    ts2i = TS2IConverter(frequency=96, image_size=448)
    image = ts2i.convert(x)
    assert image.width == 448
    assert image.height > 0


def test_i2ts_shape():
    ts2i = TS2IConverter(frequency=96, image_size=448)
    i2ts = I2TSConverter(frequency=96, image_size=448)
    x = np.random.randn(96 * 7, 2).astype(np.float32)
    image = ts2i.convert(x)
    x_hat = i2ts.convert(image, n_vars=2, target_length=x.shape[0], stats={"mu": np.zeros(2), "sigma": np.ones(2)})
    assert x_hat.shape == x.shape


def test_bitsi_validator():
    x = np.random.randn(96 * 3, 2).astype(np.float32) * 100 + 200
    rfn = RobustFidelityNormalizer()
    ts2i = TS2IConverter(frequency=96, image_size=448)
    i2ts = I2TSConverter(frequency=96, image_size=448)
    validator = BiTSIValidator(rfn, ts2i, i2ts)
    metrics = validator.validate(x)
    assert "mae" in metrics
    assert "rmse" in metrics


def test_mock_adapter():
    adapter = build_adapter("mock")
    adapter.load(None, "cpu")
    assert adapter.loaded
    from PIL import Image
    image = Image.new("RGB", (448, 448))
    answer = adapter.understand(image, "test")
    assert "<think>" in answer
    generated = adapter.generate(image, "test")
    assert generated.size == (896, 896)


def test_understanding_qa_generation():
    sample = Sample(
        context=np.random.randn(96 * 7, 3).astype(np.float32),
        target=np.random.randn(96, 3).astype(np.float32),
        metadata={"target_var_idx": 0},
        task="forecasting",
    )
    gen = UnderstandingTaskGenerator(frequency=96, image_size=448)
    qas = gen.generate_all(sample)
    assert len(qas) >= 4
    assert qas[0]["task_id"] == "qa1"


def test_generation_evaluator():
    pred = np.random.randn(100)
    ref = pred + np.random.randn(100) * 0.1
    evaluator = GenerationEvaluator()
    result = evaluator.evaluate(pred, ref)
    assert "mae" in result
    assert "rmse" in result


def test_text_extraction():
    text = "<think>Reasoning</think> Answer"
    assert extract_between_tags(text) == "Reasoning"


def test_generation_metrics():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    ref = np.array([1.1, 2.1, 3.1, 4.1])
    mae = GenerationMetrics.mae(pred, ref)
    assert mae > 0
    direction = GenerationMetrics.direction_accuracy(pred, ref)
    assert 0 <= direction <= 1


def test_understanding_evaluator():
    pred = {"task_id": "qa1", "answer": 3}
    ref = {"task_id": "qa1", "answer": 3}
    evaluator = UnderstandingEvaluator()
    score = evaluator.evaluate([pred], [ref])
    assert score["qa1"] == 1.0
