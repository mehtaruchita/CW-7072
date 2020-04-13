#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from log_reg_cost_function import  sigmoid
#testing data
nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
fig.show()
