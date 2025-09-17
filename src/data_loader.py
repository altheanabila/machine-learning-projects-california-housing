from sklearn.datasets import fetch_california_housing
import pandas as pd





def load_california_housing(as_frame: bool = True):

    housing = fetch_california_housing(as_frame=as_frame)
    if as_frame:
        df = housing.frame
        X = df.drop(columns=["MedHouseVal"])
        y = df["MedHouseVal"]
        return X, y, housing
    else:
        return housing.data, housing.target, housing