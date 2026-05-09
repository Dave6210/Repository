import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_and_compare(df):
    #1.构造特征
    #预测任务:预测特定小时、星期、区域的订单需求量
    model_data = df.groupby(['hour', 'day_of_week', 'PULocationID']).size().reset_index(name='demand')
    X = model_data[['hour', 'day_of_week', 'PULocationID']].values.astype(np.float32)
    y = model_data['demand'].values.astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #2.随机森林
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train.ravel())
    rf_pred = rf.predict(X_test)

    #3.PyTorch 神经网络
    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)

    #简单网络结构
    model = nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []
    print("神经网络训练中...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    #绘制 Loss 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Training Loss')
    plt.title('M3 神经网络训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('outputs/loss_curve.png')
    plt.close()

    #评估
    nn_pred = model(torch.tensor(X_test)).detach().numpy()
    mae = mean_absolute_error(y_test, nn_pred)
    rmse = np.sqrt(mean_squared_error(y_test, nn_pred))

    print(f"训练完成！指标 -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return model, rf