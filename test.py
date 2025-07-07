import numpy as np
import matplotlib.pyplot as plt

min_timescale = 1.0
max_timescale = 10000.0
num_timescales = 6

log_increment = np.log(max_timescale / min_timescale) / (num_timescales - 1)
inv_timescales = min_timescale * np.exp(-np.arange(num_timescales) * log_increment)

positions = np.arange(0, 100)

plt.figure(figsize=(10, 5))
for i, inv_ts in enumerate(inv_timescales):
    # 画出 sin 的波形
    signal = np.sin(positions * inv_ts)
    plt.plot(positions, signal, label=f"timescale {i}\n(inv={inv_ts:.4f})")

plt.title("不同 timescale 下的位置编码正弦波")
plt.xlabel("position")
plt.ylabel("sin(position * inv_timescale)")
plt.legend()
plt.grid()
plt.show()