import sys
sys.path.append('/home/ubuntu/AcquisitionSynthesis/rewards/')
from format import parse
import requests
import torch
import os
SERVER_IP = os.environ['SERVER_IP']

def compute_combined(data):
    SERVER_A = f"http://{SERVER_IP}:5145/combined"
    payload = {
        "data": data,
    }

    for _ in range(3):
        r = requests.post(
            SERVER_A,
            json=payload,
            headers={"X-API-Key": ""},
            timeout=1000,
        )
        if r.ok: break
    if not r.ok: return float(0.0)
    print("status:", r.status_code)
    print("body:", r.text)
    r.raise_for_status()

    if r.json()["acquisition_reward"] is not None:
        combined_reward = r.json()["acquisition_reward"]
    else:
        return float(0.0)
    return combined_reward

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    data, xml_reward = parse(solution_str)
    if data is None: return float(0.0)

    combined_reward = compute_combined(data)
    return combined_reward + xml_reward