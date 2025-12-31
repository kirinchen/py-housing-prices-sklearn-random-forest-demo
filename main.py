import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# === 修正 1: 設定中文字型 (針對 Windows) ===
# 必須在畫圖之前設定，告訴 Matplotlib 使用微軟正黑體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 讓負號正常顯示

# 1. 創造模擬資料 (假裝我們有 1000 筆成交紀錄)
np.random.seed(42)
n_samples = 1000

# 變數 A：坪數 (10 ~ 50坪)
size = np.random.randint(10, 50, n_samples)

# 變數 B：屋齡 (0 ~ 40年) - 這是我們要測試的「V型」變數
age = np.random.randint(0, 40, n_samples)

# 變數 C：幸運數字 (1 ~ 9) - 這是完全沒用的「垃圾變數」
lucky_num = np.random.randint(1, 10, n_samples)


# --- 定義房價公式 (這是上帝視角，模型不知道) ---
# 房價 = 坪數 * 50萬 + 屋齡因素 + 隨機波動
# 屋齡因素邏輯：(屋齡 - 20) 的平方 -> 讓 0歲和40歲都很貴，20歲最便宜

def _main():
    price = (size * 50) + ((age - 20) ** 2 * 10) + np.random.normal(0, 200, n_samples)

    # 整理成表格
    df = pd.DataFrame({
        '坪數 (Size)': size,
        '屋齡 (Age)': age,
        '幸運數字 (Lucky)': lucky_num,
        '房價 (Price)': price
    })

    # 2. 丟進 sklearn 分析 (完全不做預處理)
    X = df.drop(columns=['房價 (Price)'])
    y = df['房價 (Price)']

    # 建立隨機森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 3. 抓出特徵重要性 (Feature Importance)
    importance = model.feature_importances_
    feature_names = X.columns

    # 4. 畫圖驗證
    plt.figure(figsize=(10, 5))

    # === 修正 2: 更新 Seaborn 語法 (消除 FutureWarning) ===
    # 舊寫法: sns.barplot(x=importance, y=feature_names, palette='viridis')
    # 新寫法: 指定 hue 並關閉 legend
    sns.barplot(
        x=importance,
        y=feature_names,
        hue=feature_names,
        palette='viridis',
        legend=False
    )
    # ===================================================

    plt.title('房價預測：哪個變數最重要？')
    plt.xlabel('重要性分數 (越高越好)')
    plt.show()

    # 5. 印出分數
    print("變數重要性排行：")
    for name, score in zip(feature_names, importance):
        print(f"{name}: {score:.4f}")


if __name__ == '__main__':
    _main()