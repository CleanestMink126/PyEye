import numpy as np


def analyzePerson(iris_list):
    mean_iris = np.nanmean(iris_list, axis = 0)
    list_sums = [compareImages(mean_iris,iris) for iris in iris_list]
    list_sums_mean = np.mean(list_sums)
    list_sums_std = np.std(list_sums)

    return mean_iris, list_sums_mean, list_sums_std

def compareImages(iris1, iris2):
    not_count = np.logical_or(np.isnan(iris1) ,np.isnan(iris2))
    diff = np.nansum((-iris1,iris2),axis = 0)
    diff[not_count] = 0
    diff = np.absolute(diff)
    if np.max(not_count): # if has nans
        # print(diff)
        diff[not_count] = np.nan
    return np.nanmean(diff)

def rollImage(compared, rolled, scope = 5):
    values = [compareImages(compared,  np.roll(rolled,i, axis = 0)) for i in range(-scope,scope+1)]
    shift = np.argmin(values) - scope
    return shift

if __name__ == "__main__":
    # a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # b = np.array([[2,3,4],[5,6,7],[8,9,10]])
    # print(compareImages(2*a,2*b))
    # c = np.random.normal(loc = 2, scale = 1, size = (100, 3, 3))
    # print(analyzePerson(c))
    # c = np.random.normal(loc = 2, scale = 3, size = (100, 3, 3))
    # print(analyzePerson(c))
    a = np.array([[1,2,3],[1,1,1],[1,1,1],[1,1,1],[4,5,6],[7,8,9]])
    b = np.array([[7,8,9],[1,2,3],[1,1,1],[1,1,1],[1,1,1],[4,5,6]])
    print(rollImage(a, b, scope = 2))
    # c = np.random.normal(loc = 4, scale = 3, size = (40, 3, 3))
    # print(analyzePerson(c))
