from typing import List
from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel
import dill
import numpy as np
from service.api.exceptions import UserNotFoundError
from service.log import app_logger
from collections import Counter
from pathlib import Path
from typing import Dict
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender, TFIDFRecommender, BM25Recommender
from service.models import recommend_popular, recommend_knn, UserKnn
import pandas as pd
import os
import implicit


# get popular recommendations
interactions = pd.read_csv('data/interactions.csv')
interactions['last_watch_dt'] = pd.to_datetime(interactions['last_watch_dt'])
interactions.rename(
    columns={
        'last_watch_dt': 'datetime',
        'total_dur': 'weight',
            },
    inplace=True,
    )
popular_recs = recommend_popular(interactions)
popular_recs_30 = recommend_popular(interactions, days = 30)

interactions.drop("weight", axis = 1, inplace = True)
interactions.rename(
    columns={
        'watched_pct': 'weight',
            },
    inplace=True,
    )


# get dic with users types - hot/cold
with open("models/user_categories.dill", "rb") as f:
        user_type_dict = dill.load(f)


#load userknn model
with open("models/userknn_hot_users.dill", "rb") as f:
        userknn_hot_users = dill.load(f)
print('Model is loaded and ready to predict')


# load predictions of listwise lgbm model
listwise_preds = pd.read_csv("data/preds_listwise.csv")
listwise_preds.recs = listwise_preds.recs.apply(lambda x: [int(i) for i in x[1:-1].split(", ")])


# load predictions of dssm model
dssm_preds = pd.read_csv("data/dssm_predictions.csv")
dssm_preds.item_id = dssm_preds.item_id.apply(lambda x: [int(i) for i in x[1:-1].split(", ")])


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")
    if model_name == "popular":
        recs_list = popular_recs
    elif model_name == "knn_popular_hybrid":
        try:
            if user_type_dict[user_id] == 'cold':
                recs_list = popular_recs
            elif user_type_dict[user_id] == 'hot':
                user_interactions = interactions[interactions["user_id"] == user_id][
                        ["user_id", "item_id", "weight"]
                    ]
                recs_list = recommend_knn(userknn_hot_users, user_interactions, popular_recs)
        except:
            recs_list = popular_recs
    elif model_name == "listwiseLGBM":
        try:
            recs_list = listwise_preds[listwise_preds.user_id == user_id].recs.values[0]
            recs_list = list(dict.fromkeys(recs_list))
            if len(recs_list) != 10:
                diff_with_pop = [x for x in popular_recs if x not in recs_list]
                recs_list = recs_list + diff_with_pop[:(10-len(recs_list))]
        except:
            recs_list = popular_recs

    elif model_name == "DSSM":
        try:
            recs_list = dssm_preds[dssm_preds.user_id == user_id].item_id.values[0]
        except:
            recs_list = popular_recs_30


    return RecoResponse(user_id=user_id, items=recs_list)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
