import pandas as pd
from vllm import LLM, SamplingParams

df = pd.read_parquet('/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_nemotron_combined.parquet')
