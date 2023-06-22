import numpy as np

from keras.datasets import boston_housing

from matplotlib import pyplot as plt


(train_X, train_y), (test_X, test_y) = boston_housing.load_data()

y = list(train_y) + list(test_y)

print(min(y))
print(max(y))

bins = np.arange(min(y), max(y), 5) # fixed bin size

plt.xlim([min(y)-5, max(y)+5])

plt.hist(y, bins=bins, alpha=0.5)
plt.title('Repartition of data in Boston Housing DataSet')
plt.xlabel('Price')
plt.ylabel('Count')

plt.show()