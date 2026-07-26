import sys
sys.path.append('/home/ubuntu/AcquisitionSynthesis')
from argparse import ArgumentParser
import os
from config import *
# from rewards.confidence import compute_confidence_batch
from rewards.combined_batch import compute_combined_batch
# from rewards.gradient import compute_gradient
# from rewards.diversity import compute_diversity
# from rewards.answer_variance import compute_answer_variance_batch
import requests
from tqdm import tqdm
import pickle

def normalize(nums):
    nums = np.array(nums)
    nums = (nums - nums.min()) / (nums.max() - nums.min() + 1e-4)
    return nums


def filter(train_data: pd.DataFrame, size, model_name, dataset_name, dir_name):
    acquisition_rewards = []
    if not os.path.exists(f'filtering_metadata/{dir_name}/combined.pkl'):
        data = []
        for i in tqdm(range(len(train_data))):
            data.append(train_data.iloc[i]['question'])

        acquisition_rewards.append(compute_combined_batch(data, model_name))
        with open(f'filtering_metadata/{dir_name}/combined.pkl', 'wb+') as f:
            pickle.dump(acquisition_rewards[-1], f)
    
    rankings = np.zeros((len(train_data)))
    for j in range(len(acquisition_rewards)):
        rankings += normalize(acquisition_rewards[j])
    
    train_data['rank'] = rankings
    train_data = train_data.sort_values('rank', ascending=False)
    return train_data[:size]


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--data", type=str, default="nemotron")
    argparser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    argparser.add_argument("--dir_name", type=str, default="llama8b_nemotron")
    argparser.add_argument("--output_file", type=str, default="filtered.parquet")
    argparser.add_argument("--size", type=int, default=5000)
    args = argparser.parse_args()

    train_data = load_data(f"/home/ubuntu/AcquisitionSynthesis/data/{args.data}/valid.parquet")
    filtered = filter(train_data, args.size, args.model, f"/home/ubuntu/AcquisitionSynthesis/data/{args.data}/valid.parquet", args.dir_name)

    def format(col):
        filtered[col] = filtered[col].apply(lambda x: x.replace("\n", ""))
    
    format('question')
    format('answer')
    format('reasoning')
    file_name = args.output_file
    filtered.to_parquet(file_name)