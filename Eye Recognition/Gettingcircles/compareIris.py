import numpy as np
import pickle
import polarTransform
import pupilDetection
import os
import pupilDeps


def analyzeList(mean_iris,iris_list, roll):
    iris_list = [rollImage(mean_iris,iris, scope=roll) for iris in iris_list]
    list_sums = [compareImages(mean_iris,iris) for iris in iris_list]
    list_sums_mean = np.mean(list_sums)
    list_sums_std = np.std(list_sums)
    return list_sums_mean, list_sums_std


def compareImages(iris1, iris2):
    not_count = np.logical_or(np.isnan(iris1) ,np.isnan(iris2))
    diff = np.nansum((-iris1,iris2),axis = 0)
    diff[not_count] = 0
    diff = np.square(diff)
    if np.max(not_count): # if has nans
        # print(diff)
        diff[not_count] = np.nan
    return np.nanmean(diff)

def rollImage(compared, rolled, scope = 5):
    values = [compareImages(compared,  np.roll(rolled,i, axis = 0)) for i in range(-scope,scope+1)]
    return np.roll(rolled,np.argmin(values) - scope, axis = 0)


class info_class:
    def __init__(self,name):
        self.name = name

class indiv_iris:
    def __init__(self, info,iris_list, roll = 5):
        self.iris_list = iris_list
        self.info = info
        self.mean_iris = np.nanmean(iris_list, axis = 0)
        self.my_mean, self.my_std = analyzeList(self.mean_iris, self.iris_list, roll)

    def setComparison(self,iris_list, roll):
        self.other_mean, self.other_std = analyzeList(self.mean_iris, iris_list, roll)

    def analyzeImage(self,iris):
        diff = compareImages(self.mean_iris, iris)
        z_score_me = (diff - self.my_mean)/self.my_std
        z_score_other = (diff - self.other_mean)/self.other_std
        return z_score_me, z_score_other

class iris_db:
    def __init__(self, filename):
        self.person_list = []
        self.filename = filename
        self.iris_list_tot = None
        self.iris_list_tot_indices = None
        self.numPeople = 0
        self.roll = 5

    def add_person(self,info, iris_list):
        self.numPeople += 1
        info.number = self.numPeople
        self.person_list.append(indiv_iris(info, iris_list, roll = self.roll))
        if self.iris_list_tot is not None:
            self.iris_list_tot = np.concatenate((self.iris_list_tot,iris_list), axis = 0)
            self.iris_list_tot_indices = np.concatenate((self.iris_list_tot_indices,[info.number]*len(iris_list)), axis = 0)
        else:
            self.iris_list_tot = iris_list
            self.iris_list_tot_indices =[info.number]*len(iris_list)

    def setComparison(self):
        for person in self.person_list:
            iris_list = self.iris_list_tot[self.iris_list_tot_indices != person.info.number]
            person.setComparison(iris_list, self.roll)

    def save(self):
        pickle.dump(self, self.filename)

    def findMostLikely(self,irisImg):
        listValues = [(person.analyzeImage(irisImg),person.info.name) for person in self.person_list]
        print(listValues)




if __name__ == "__main__":
    # db = iris_db("testFile")
    # a = np.random.normal(loc= 1,scale = 1, size = (40, 5, 5))
    # b = np.random.normal(loc= -1,scale = 1, size = (40, 5, 5))
    # c = np.random.normal(loc= 3,scale = 1, size = (40, 5, 5))
    # d = np.random.normal(loc= -3,scale = 1, size = (40, 5, 5))
    #
    # db.add_person(info_class("a"),a)
    # db.add_person(info_class("b"),b)
    # db.add_person(info_class("c"),c)
    # db.add_person(info_class("d"),d)
    # db.setComparison()
    # print('A2')
    # a2 = np.random.normal(loc= 1,scale = 1, size = (5, 5))
    # db.findMostLikely(a2)
    #
    # print('B2')
    # a2 = np.random.normal(loc= -1,scale = 1, size = (5, 5))
    # db.findMostLikely(a2)
    #
    # print('C2')
    # a2 = np.random.normal(loc= 3,scale = 1, size = (5, 5))
    # db.findMostLikely(a2)
    #
    # print('D2')
    # a2 = np.random.normal(loc= -3,scale = 1, size = (5, 5))
    # db.findMostLikely(a2)

    directory  = '../EyePictures/Yichen/'
    subfolder = '1/'
    numEyes = 0
    for i in range(3):
        myImg = pupilDetection.displayImg(directory + str(i+1)+'.jpg')
        new_img = polarTransform.polarToCart(gray_img = myImg.img, center_x =myImg.center[1]
        ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad))
        new_img = polarTransform.polarToCart(gray_img = myImg.likelihood, center_x =myImg.center[1]
        ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad))
        new_img = polarTransform.polarToCart(gray_img = myImg.threshold, center_x =myImg.center[1]
        ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad))
    #
    # for filename in os.listdir(directory):
    #     try:
    #         personIndex = int(filename)
    #         print(personIndex)
    #         curr_subfolder = directory + filename+'/'+subfolder
    #         for img_name in os.listdir(curr_subfolder):
    #             try:
    #                 print(img_name)
    #                 if img_name.endswith(".bmp"):
    #
    #                     myImg = pupilDetection.getCircles(curr_subfolder + img_name)
    #                     new_img = polarTransform.polarToCart(gray_img = myImg.img, center_x =myImg.center[1]
    #                     ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad))
    #                     new_img = polarTransform.polarToCart(gray_img = myImg.likelihood, center_x =myImg.center[1]
    #                     ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad))
    #                     new_img = polarTransform.polarToCart(gray_img = myImg.diff, center_x =myImg.center[1]
    #                     ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad))
    #
    #                     # plt.imshow(new_img, cmap='gray')
    #                     # plt.show()
    #                     numEyes +=1
    #             except pupilDetection.BaselineError:
    #                 print('Baseline Error')
    #             if numEyes > 10: break
    #         if numEyes > 10: break
    #     except ValueError:
    #         continue

    # pupilDetection.getCircles()
