from timeomni_vl.data.loader import CSVLoader
from timeomni_vl.data.aligner import DateAligner
from timeomni_vl.data.cleaner import DataCleaner

loader = CSVLoader("Dataset")
data_dict = loader.load_all()
print(f"Loaded {len(data_dict)} files")

aligner = DateAligner(frequency=96)
aligned = aligner.align(data_dict, "统一结算点电价临时结果")
print(f"Aligned shape: {aligned.shape}")
print(f"Columns: {list(aligned.columns)[:5]} ... {list(aligned.columns)[-5:]}")
print(f"Date range: {aligned.index.min()} to {aligned.index.max()}")
print(f"Missing ratio per column:\n{aligned.isna().mean().sort_values(ascending=False).head()}")

cleaner = DataCleaner(missing_threshold=0.5)
cleaned = cleaner.clean(aligned)
print(f"Cleaned shape: {cleaned.shape}")
print(f"Remaining columns: {list(cleaned.columns)}")
