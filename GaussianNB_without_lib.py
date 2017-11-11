import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

### generate training data
x_class_0 = np.arange(2,6,0.2)
y_class_0 = 1 + (np.random.normal(0,1,len(x_class_0)))*2

x_class_1 = np.arange(14,18,0.2)
y_class_1 = 1 + (np.random.normal(0,1,len(x_class_1)))*2

x_class_2 = np.arange(7,12,0.2)
y_class_2 = 14 + (np.random.normal(0,1,len(x_class_2)))*2

x = np.concatenate([x_class_0, x_class_1, x_class_2])
y = np.concatenate([y_class_0, y_class_1, y_class_2])

classes_0 = np.zeros(len(x_class_0))
classes_1 = np.zeros(len(x_class_1)) + 1
classes_2 = np.zeros(len(x_class_2)) + 2

classes = np.concatenate([classes_0, classes_1, classes_2])

train_data = np.column_stack((x,y,classes))

colors = ['r', 'g', 'b']
f = lambda x: colors[int(x)]
colors_train = list(map(f,classes))

my_dpi = 96
plt.figure(figsize =(800/my_dpi, 800/my_dpi), dpi = my_dpi)

plt.scatter(x,y,color = colors_train)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training Data\n')

plt.savefig("gausNB_training_data.png")
