import numpy as np
import matplotlib.pyplot as plt

# 定義 p 的範圍，避開 0 避免 log(0)
p = np.linspace(0.001, 1, 1000)

# 定義兩種 Loss 函數
cross_entropy = -np.log(p)
squared_residual = (1 - p)**2

# 畫圖
plt.figure(figsize=(10, 6))
plt.plot(p, cross_entropy, label='CE = -log(p)', color='dodgerblue', linewidth=2)
plt.plot(p, squared_residual, label='Squared Error = (1 - p)²', color='darkorange', linewidth=2)

# 標註
plt.text(0.075, 5.5, 'Bad\nPrediction!!!', ha='center', va='center', fontsize=10)
plt.text(0.93, 0.35, 'Good Prediction!!!', ha='center', va='center', fontsize=10)

# 軸與標籤
plt.xlabel('p', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Cross-Entropy vs Squared Error', fontsize=14)
plt.legend()
plt.grid(True)
plt.ylim(0, 7)  # 限制 y 軸避免太高

# 顯示圖形
plt.savefig('loss_comparison.png', bbox_inches='tight')
