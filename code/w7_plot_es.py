import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ✅ 中文字型設定
rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Heiti TC', 'Noto Sans CJK TC']
rcParams['axes.unicode_minus'] = False

# 模擬 epochs
x = np.linspace(0.1, 10, 200)

# 訓練誤差（持續下降）
train_error = 0.8 / (x + 1) + 0.02

# 泛化誤差（先下降後上升，有明顯谷底）
gen_error = 0.6 / (x + 1) + 0.03 * (x - 4)**2 / 10 + 0.05

# 畫圖
plt.figure(figsize=(8, 4))
plt.plot(x, train_error, 'b-', label='Training loss', linewidth=2)
plt.plot(x, gen_error, 'g-', label='Validation loss', linewidth=2)

# 最佳 epoch（EarlyStopping 點）
optimal_x = 4

# 軸標、圖例、標題
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(frameon=True, fontsize=10)
plt.title('Early Stopping Illustration', fontsize=13)
plt.tight_layout()
plt.savefig('early_stopping.png', bbox_inches='tight', dpi=300)
plt.show()
