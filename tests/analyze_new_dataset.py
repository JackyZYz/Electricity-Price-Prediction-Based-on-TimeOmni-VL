from timeomni_vl.data.loader import CSVLoader
from timeomni_vl.data.aligner import DateAligner
from timeomni_vl.data.cleaner import DataCleaner

loader = CSVLoader("Dataset")
data_dict = loader.load_all()
print(f"Loaded {len(data_dict)} files")

aligner = DateAligner(frequency=96)
aligned = aligner.align(data_dict, "统一结算点电价临时结果")
print(f"Aligned shape: {aligned.shape}")

print("\nMissing ratios (top 15):")
print(aligned.isna().mean().sort_values(ascending=False).head(15))

cleaner = DataCleaner(missing_threshold=0.5)
cleaned = cleaner.clean(aligned)
print(f"\nCleaned shape: {cleaned.shape}")
print(f"Remaining columns ({len(cleaned.columns)}):")
for c in cleaned.columns:
    print(f"  - {c}")
