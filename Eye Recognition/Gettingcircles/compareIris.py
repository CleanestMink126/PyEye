import numpy as np
import pickle
import polarTransform
import pupilDetection
import os


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
    return np.argmin(values) - scope



class indiv_iris:
    def __init__(self, info,iris_list):
        self.iris_list = iris_list
        self.info = info
        self.mean_iris, self.list_sums_mean, self.list_sums_std = analyzePerson(self.iris_list)

    def analyzeImage(self,iris):
        diff = compareImages(self.mean_iris, iris)
        z_score = (diff - self.list_sums_mean)/self.list_sums_std
        return z_score

class iris_db:
    def __init__(self, filename):
        self.person_list = []
        self.filename = filename

    def add_person(self,info, iris_list):
        self.person_list.append(indiv_iris(info, iris_list))

    def save(self):
        pickle.dump(self, self.filename)



if __name__ == "__main__":

    directory  = '../EyePictures/'

    numEyes = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            myImg = pupilDetection.getCircles(directory+filename)
            new_img = polarTransform.polarToCart(path = directory+filename, center_x =myImg.center[1]
            ,center_y=myImg.center[0], radius = myImg.irisRad)
            # plt.imshow(new_img, cmap='gray')
            # plt.show()
            numEyes +=1
        if numEyes > 10: break
    pupilDetection.getCircles()
