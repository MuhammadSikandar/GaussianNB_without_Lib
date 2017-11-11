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


### naive bayes classfier

### start with finding prior P(class)
### p(Y=k) = count(data pts in class k) / total # of data pts

def prior(data,K):
    p = np.ones((K))
    nrow = data.shape[0]
    for k in np.arange(K):
        p[k] = float(np.sum(data[:,-1] == k)) / float(nrow)
    return p

prior_ = prior(train_data, 3)


## for likelihood i need to first implement gaussian and for gaussian I need mean and std dev

J = 2 # feature vector, x1, x2 or here in this code x, y

def learn_gaussians(train_data, J, K):
    mean = np.zeros(J*K).reshape((J,K))
    variance = np.zeros(J*K).reshape((J,K))
    classes = train_data[:,-1]
    for j in np.arange(J):
        for k in np.arange(K):
            mean[j,k] = float(np.sum(train_data[classes==k,j])) / float(np.sum(classes == k))
            variance[j,k] = np.sum((train_data[classes==k, j] - mean[j,k])**2) /(np.sum(classes ==k) - 1)
    return mean, variance


gaussian = learn_gaussians(train_data,2,3)
gauss_mean = gaussian[0]
print gauss_mean
gauss_variance = gaussian[1]
print gauss_variance

## likelihood
## xi = test data pt
## K is class k1, k2 k3
## J is attribute x and y

def likelihood(xi, k, gauss_mean, gauss_variance):
    product_of_gaussians = 1
    for j in np.arange(J):
        gaussian_ = scipy.stats.norm.pdf(xi[j], gauss_mean[j,k], np.sqrt(gauss_variance[j,k]))
        product_of_gaussians = product_of_gaussians * gaussian_
    return product_of_gaussians

## posterior probAblity is the output result
## posterior = likelihood * prior

def posterior (xi, K, gauss_mean, gauss_variance):
    post = np.zeros(K)
    for k in np.arange(K):
        post[k] = prior_[k] * likelihood(xi, k, gauss_mean, gauss_variance)
    return  post

### generating test data

x_test = np.arange(0,20,0.2)
y_test = 3 + (np.random.normal(0,1,len(x_test))) * 12
data_test = np.column_stack((x_test, y_test))

#classify with naive bayes
classes_test = np.ones((len(x_test)))

for i in np.arange((len(x_test))):
    post = posterior(data_test[i,:], 3, gauss_mean, gauss_variance)
    classes_test[i] = (np.where(post ==np.max(post)))[0][0]

colors_test = list(map(f, classes_test))

plt.figure (figsize= (800/my_dpi, 800/my_dpi), dpi= my_dpi)
plt.scatter (x_test, y_test, color = colors_test)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Data\n')

plt.savefig("GaussNB_testresults-png")