from tests.test_modules import (
    test_bitsi_validator,
    test_generation_evaluator,
    test_generation_metrics,
    test_i2ts_shape,
    test_mock_adapter,
    test_rfn_roundtrip,
    test_text_extraction,
    test_ts2i_shape,
    test_understanding_evaluator,
    test_understanding_qa_generation,
)

test_rfn_roundtrip()
print("rfn ok")
test_ts2i_shape()
print("ts2i ok")
test_i2ts_shape()
print("i2ts ok")
test_bitsi_validator()
print("validator ok")
test_mock_adapter()
print("mock adapter ok")
test_understanding_qa_generation()
print("understanding qa ok")
test_generation_evaluator()
print("generation evaluator ok")
test_text_extraction()
print("text extraction ok")
test_generation_metrics()
print("generation metrics ok")
test_understanding_evaluator()
print("understanding evaluator ok")
print("all module tests passed")
