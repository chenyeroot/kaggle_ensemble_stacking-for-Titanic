#导入相应的工具库
import pandas as pd  
import xgboost as xgb  
  
from xgboost.sklearn import XGBClassifier  
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV,cross_val_score ,RandomizedSearchCV 
from scipy.stats import uniform  #均匀分布
from sklearn import metrics  
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import sklearn.preprocessing as preprocessing

from sklearn.ensemble import (RandomForestClassifier, 
                              ExtraTreesClassifier,
                              AdaBoostClassifier, 
                              GradientBoostingClassifier,
                              RandomForestRegressor)
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
