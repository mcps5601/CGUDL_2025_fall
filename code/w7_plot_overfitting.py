import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ✅ 設定中文字型（Windows 用微軟正黑體、Mac 用蘋方或黑體、Linux 用 Noto）
rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Heiti TC', 'Noto Sans CJK TC']
rcParams['axes.unicode_minus'] = False  # 避免負號顯示錯誤

# 模擬 epochs
x = np.linspace(0.1, 10, 200)

# 平滑的訓練與泛化誤差曲線
train_error = 1 / (x + 1)
gen_error = 1 / (x + 1) + 0.02 * x + 0.05

# 畫圖
plt.figure(figsize=(8, 4))
plt.plot(x, train_error, 'b-', label='Training error', linewidth=2)
plt.plot(x, gen_error, 'g-', label='Generalization error', linewidth=2)

# 最適容量
optimal_x = 3
plt.axvline(optimal_x, color='r', linewidth=2)

# 標註 Underfitting / Overfitting 區域
plt.text(0.6, 0.7, 'Underfitting zone', fontsize=11)
plt.text(optimal_x + 0.4, 0.7, 'Overfitting zone', fontsize=11)

# 軸標、圖例、標題
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('誤差值', fontsize=12)
plt.legend(frameon=True, fontsize=10)
plt.title('Underfitting vs Overfitting', fontsize=13)
plt.tight_layout()
plt.savefig('overfitting.png', bbox_inches='tight', dpi=300)
plt.show()
