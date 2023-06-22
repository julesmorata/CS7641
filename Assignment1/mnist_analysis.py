import numpy as np

from keras.datasets import mnist

from matplotlib import pyplot as plt


(train_X, train_y), (test_X, test_y) = mnist.load_data()

y = list(train_y) + list(test_y)

counts = []
for i in range(10):
    counts.append(y.count(i))

plt.bar(range(10), counts)
plt.title('Repartition of data in MNIST DataSet')
plt.xlabel('Label')
plt.xticks(np.arange(0, 10, 1))
plt.ylabel('Count')

plt.show()