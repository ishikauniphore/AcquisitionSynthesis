import sys
sys.path.append('/home/ubuntu/AcquisitionSynthesis/rewards/')
from format import parse
import requests
import torch
import os
SERVER_IP = os.environ['SERVER_IP']

def compute_combined_lsi(data):
    SERVER_A = f"http://{SERVER_IP}:5145/combined_lsi"
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
        combined_lsi_reward = r.json()["acquisition_reward"]
    else:
        return float(0.0)
    return combined_lsi_reward

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    data, xml_reward = parse(solution_str)
    if data is None: return float(0.0)

    combined_lsi_reward = compute_combined_lsi(data)
    return combined_lsi_reward + xml_reward
