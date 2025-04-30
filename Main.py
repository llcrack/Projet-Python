import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_object as go

from dash import dcc, html
from dash.dependecies import Input, Output

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier