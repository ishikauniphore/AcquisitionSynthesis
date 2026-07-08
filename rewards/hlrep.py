import sys
sys.path.append('/home/ubuntu/AcquisitionSynthesis/rewards/')
from format import parse
import requests
import torch
import os
SERVER_IP = os.environ['SERVER_IP']

def compute_hlrep(data):
    SERVER_A = f"http://{SERVER_IP}:5145/hlrep"
    payload = {
        "data": data,
    }

    for _ in range(3):
        r = requests.post(
            SERVER_A,
            json=payload,
            headers={"X-API-Key": ""},
            timeout=600,
        )
        if r.ok: break
    if not r.ok: return float(0.0)
    print("status:", r.status_code)
    print("body:", r.text)
    r.raise_for_status()

    if r.json()["acquisition_reward"] is not None:
        hlrep_reward = min(max(r.json()["acquisition_reward"], 0.0), 1.0)
    else:
        return float(0.0)
    return hlrep_reward

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    data, xml_reward = parse(solution_str)
    if data is None: return float(0.0)

    hlrep_reward = compute_hlrep(data)
    return hlrep_reward + xml_reward