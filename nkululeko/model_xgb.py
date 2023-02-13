# xgbmodel.py

from xgboost import XGBClassifier
from nkululeko.model import Model

class XGB_model(Model):
    """An XGBoost model"""
    clf = XGBClassifier(use_label_encoder=False) # set up the classifier
