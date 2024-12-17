import pandas as pd
from perfume_recommender.perfume_recommendation.perfume_recommendation import (
    PerfumeRecommender,
)


def test_build_main_accord_trees():
    df = pd.DataFrame(
        {"mainaccord1": ["floral", "woody"], "mainaccord2": ["citrus", "spicy"]}
    )
    recommender = PerfumeRecommender(df)
    recommender.build_main_accord_trees()

    assert (
        "mainaccord2" in recommender.main_accord_trees
    ), "Main accord trees should be built"


def test_recommend_main_accord():
    df = pd.DataFrame(
        {"mainaccord1": ["floral", "floral"], "mainaccord2": ["citrus", "spicy"]}
    )
    recommender = PerfumeRecommender(df)
    recommender.build_main_accord_trees()

    recommendations = recommender.recommend_main_accord({"mainaccord1": "floral"})
    assert recommendations == [
        "citrus",
        "spicy",
    ], "Next main accord recommendations should match"
