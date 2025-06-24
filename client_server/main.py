from fastapi import FastAPI

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from argparse import ArgumentParser
from pathlib import Path

from models.train_pytorch_lstm import AttentionLSTMModel
from models.train_ratio_predictor import TransformerRatioPredictor
from models.utils import (
    make_sequence_data_enhanced, 
    make_ratio_sequence_data_with_padding
    )
from models.budget_utils import (
    rebalance_budget,
    calc_daily_limits,
    simulate_with_user_budget,
    calculate_last_month_budget,
)
from models.config import *

from models.client_combined_user_specific import post_inference,FLCombinedClient


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test", tags=["test"])
async def test():
    return {"message": "This is a test endpoint"}


@app.get("/predict", tags=["predict"])
async def predict():
    # parser = ArgumentParser()
    # parser.add_argument("--user_id", type=str, required=True, help="ユーザーID (例: U001)")
    # args = parser.parse_args()

    client = FLCombinedClient("U001")
    fl.client.start_numpy_client(server_address="flower_server:8080", client=client)
    print(f"✅ FL完了。ユーザー U001 の推論と提案を生成中...")
    post_inference("U001", client.model_total, client.model_ratio, client.X_total, client.scaler_total, client.X_ratio, client.cat_cols)

    return {"message": "This is a test endpoint"}

