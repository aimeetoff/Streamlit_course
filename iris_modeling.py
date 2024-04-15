### 生物學案例 - 鳶尾花預測 ###
#%% (I) 情境介紹與前處理 - 匯入套件與資料
# shift + enter

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.preprocessing import scale, LabelEncoder # for scaling the data
import sklearn.metrics as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.io as pio
# pio.renderers.default = 'browser'

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
df = pd.read_csv("iris.csv", encoding="utf=8-sig")

#%% (I) 情境介紹與前處理 - 資料探勘
# F9
# 前五筆
print(df.head())

# 花種項目
print(df.iloc[:, -1].unique())

# 資料統計
description = df.describe()

#%% (I) 情境介紹與前處理 - 資料標準化
# shift + enter

# 定義X變數 與目標變數y
X = df.iloc[:, :-1]
y = df.iloc[:, -1] # define the target 

X_scaled = scale(X) # scale the iris data

le = LabelEncoder()
y_encoded = le.fit_transform(y)

#%% (II) 模型訓練與視覺化結果 - 模型建制與訓練
# shift + enter

clustering = KMeans(n_clusters=3, random_state=42)
clustering.fit(X_scaled) #fit the dataset


#%% (II) 模型訓練與視覺化結果 - 預測結果
# F9

# 預測結果
X['prediction'] = clustering.labels_
X

#%% (II) 模型訓練與視覺化結果 - 視覺化呈現: 預測
# shift + enter

colors = np.array(["Red","Green","Blue"])
fig_predict = make_subplots(rows=1, cols=2, shared_yaxes=True)

fig_predict.add_trace(
    go.Scatter(
        x=X["petal_length"], y=X["petal_width"],
        name = "petal",
        mode='markers',
        marker_color=colors[X.prediction]
    ),
    row=1,
    col=1,
)

fig_predict.add_trace(
    go.Scatter(
        x=X["sepal_length"], y=X["sepal_width"],
        mode='markers',
        name = "sepal",
        marker_color=colors[X.prediction]
    ),
    row=1,
    col=2,
)

fig_predict.update_layout(
    title_text="<b>預測</b>",
    title_x=0.1,
    width=900,
    height=600,
    font=dict(
        family="Courier New, monospace",
        size=15,
    ),
)

fig_predict.show()
plot(fig_predict, filename="".join([fig_predict['layout']['title']
        ['text'].strip('</b>'), '.html']), auto_open=False)

#%% (II) 模型訓練與視覺化結果 - 視覺化呈現: 實際
# shift + enter

colors = np.array(["Red","Green","Blue"])
fig_actual = make_subplots(rows=1, cols=2, shared_yaxes=True)

fig_actual.add_trace(
    go.Scatter(
        x=X["petal_length"], y=X["petal_width"],
        name = "petal",
        mode='markers',
        marker_color=colors[y_encoded]
    ),
    row=1,
    col=1
)

fig_actual.add_trace(
    go.Scatter(
        x=X["sepal_length"], y=X["sepal_width"],
        mode='markers',
        name = "sepal",
        marker_color=colors[y_encoded]
    ),
    row=1,
    col=2
)

fig_actual.update_layout(
    title_text = "<b>實際</b>",
    title_x = 0.1,
    width=900,
    height=600,
    font=dict(
        family="Courier New, monospace",
        size=15,
    ),
)

fig_actual.show()
plot(fig_actual, filename="".join([fig_actual['layout']['title']
        ['text'].strip('</b>'), '.html']), auto_open=False)





# %%
