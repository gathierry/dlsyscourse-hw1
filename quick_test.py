import numpy as np


def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].size):
            args[i].flat[j] += eps
            f1 = float(f(*args, **kwargs).sum())
            args[i].flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).sum())
            args[i].flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    return numerical_grads
    
def add(x, y):
    return x + y
    
# a = np.random.uniform(size=[5, 2])
# b = np.random.uniform(size=[5, 2])
# print(gradient_check(add, a, b))

def reshape(x):
    return x.reshape(3, 4)
    
# a = np.random.uniform(size=[6, 2])
# print(gradient_check(reshape, a))
    
def broadcast(x):
    return np.broadcast_to(x, [4, 6, 3])

# a = np.random.uniform(size=[3,])
# print(np.broadcast_to(a, (2,3)))
# print(np.broadcast_to(a, [5, 4]).shape)
# print(gradient_check(broadcast, a))

    
def summation(x):
    return np.sum(x, axis=1)
    
def matmul(x, y):
    return x @ y
    
# a = np.random.uniform(size=[5,5,2,3])
# b = np.random.uniform(size=[5,5,3,4])
# # print(a)
# # print(b)
# c = matmul(a, b)  # 5,5,2,4
# out_grad = np.ones(c.shape)
# a_grad = out_grad @ b.transpose(0,1,3,2)
# b_grad = a.transpose(0,1,3,2) @ out_grad
# grad = gradient_check(matmul, a, b)
# print([x.shape for x in grad], a_grad.shape, b_grad.shape)


def log(x):
    return np.log(x)
    
a = np.random.uniform(size=[2,3])
grad = gradient_check(log, a)
print(a, 1/a, grad)


