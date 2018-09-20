import numpy as np
import pickle
import polarTransform
import pupilDetection
import os
import pupilDeps
import compareDeps
import cv2

height = 80
width = 120

def analyzeList(mean_iris,iris_list, roll):
    '''Find mean and deviation within a given list'''
    iris_list = [rollImage(mean_iris,iris, scope=roll) for iris in iris_list]
    list_sums = [compareImages(mean_iris,iris) for iris in iris_list]
    list_sums_mean = np.mean(list_sums)
    list_sums_std = np.std(list_sums)
    return list_sums_mean, list_sums_std


def compareImages(iris1, iris2):
    '''Find the square difference between two images'''
    not_count = np.logical_or(np.isnan(iris1) ,np.isnan(iris2))
    diff = np.nansum((-iris1,iris2),axis = 0)
    diff[not_count] = 0
    diff = np.square(diff)
    if np.max(not_count): # if has nans
        # print(diff)
        diff[not_count] = np.nan
    return np.nanmean(diff)

def rollImage(compared, rolled, scope = 5):
    '''Check is there is some shifting going on in the iris'''
    values = [compareImages(compared,  np.roll(rolled,i, axis = 0)) for i in range(-scope,scope+1)]
    return np.roll(rolled,np.argmin(values) - scope, axis = 0)


class info_class:
    '''Just a container for info about the iris'''
    def __init__(self,name):
        self.name = name

class indiv_iris:
    '''Info about an individual'''
    def __init__(self, info,iris_list, roll = 5):
        self.iris_list = iris_list
        self.info = info
        self.mean_iris = np.nanmean(iris_list, axis = 0)
        self.my_mean, self.my_std = analyzeList(self.mean_iris, self.iris_list, roll)

    def setComparison(self,iris_list, roll):
        self.other_mean, self.other_std = analyzeList(self.mean_iris, iris_list, roll)

    def analyzeImage(self,iris):
        '''See how well an image fits into this person'''
        diff = compareImages(self.mean_iris, iris)
        z_score_me = (diff - self.my_mean)/self.my_std
        z_score_other = (diff - self.other_mean)/self.other_std
        return z_score_me, z_score_other

class iris_db:
    height = height
    widht = width

    def __init__(self, filename):
        self.person_list = []
        self.filename = filename
        self.iris_list_tot = None
        self.iris_list_tot_indices = None
        self.numPeople = 0
        self.roll = 5

    def add_person(self,info, iris_list):
        '''Add another person to the list'''
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
        '''For each person in the list set their within means and comparative mean'''
        for person in self.person_list:
            iris_list = self.iris_list_tot[self.iris_list_tot_indices != person.info.number]
            person.setComparison(iris_list, self.roll)

    def save(self):
        '''save your data'''
        pickle.dump(self, self.filename)

    def findMostLikely(self,irisImg):
        '''predict if the eye belongs to each person'''
        comparison_thres = -1
        listValues = [(person.analyzeImage(irisImg),person.info.name) for person in self.person_list]
        calculations = np.zeros(len(listValues))
        indexes = []
        for s,d in listValues:
            calculations[i] = int((s[0] - s[1]) < comparison_thres)
            indexes.append(d)
        return calculations, indexes


    def addIris(self, curr_subfolder, ending, name):
        iris_list = None
        '''create and save a new peron'''
        for img_name in os.listdir(curr_subfolder):
            try:
                print(img_name)
                if img_name.endswith(ending):
                    # cv2.imshow('detected circles',iris/255)
                    # cv2.waitKey(0)
                    print(curr_subfolder + img_name)
                    iris = getIrisInfo(curr_subfolder + img_name)

                    if iris_list is not None:
                        iris_list = np.concatenate((iris_list,iris[np.newaxis,:]), axis = 0)
                    else:
                        iris_list = iris[np.newaxis,:]
                    # plt.imshow(new_img, cmap='gray')
                    # plt.show()
                    # numEyes +=1
            except pupilDetection.BaselineError:
                print('Baseline Error')
        person_info = info_class(name)
        self.add_person(person_info,iris_list)

def getIrisInfo(filepath):
    '''Do the whole shebang given a iris filepath'''
    myImg = pupilDetection.getCircles(filepath)
    iris = polarTransform.polarToCart(gray_img = myImg.img, center_x =myImg.center[1]
    ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad), filterImg = True)
    threshold = polarTransform.polarToCart(gray_img = myImg.viablePixels, center_x =myImg.center[1]
    ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad))

    iris[threshold<=0.1] = np.nan
    return iris



if __name__ == "__main__":

    # directory  = '../EyePictures/CASIA'
    # subfolder = '1/'
    # numEyes = 0
    # for i in range(3):
    #     myImg = pupilDetection.getCircles(directory + str(i+1)+'.jpg')
    #     new_img = polarTransform.polarToCart(myImg.img, center_x =myImg.center[1]
    #     ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad), filterImg = True)
    #     new_img = polarTransform.polarToCart(myImg.likelihood, center_x =myImg.center[1]
    #     ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad))
    #     # new_img = polarTransform.polarToCart(gray_img = myImg.threshold, center_x =myImg.center[1]
        # ,center_y=myImg.center[0], radius = (myImg.pupilRad,myImg.irisRad))
    directory  = '../EyePictures/CASIA/'
    subfolder_train = '1/'
    subfolder_test = '2/'
    db = iris_db('test.pkl')
    for filename in os.listdir(directory):
        try:
            personIndex = int(filename)
            # if personIndex>30:
            #     continue
            print(personIndex)

            curr_subfolder = directory + filename+'/'+subfolder_train
            db.addIris(curr_subfolder, '.bmp', filename)
            # if numEyes > 10: break
        except:
            continue

    db.setComparison()
    truePositive = []
    falsePositive = []

    for filename in os.listdir(directory):
        try:
            personIndex = int(filename)
            # if personIndex>30:
            #     continue
            print(personIndex)
            curr_subfolder = directory + filename+'/'+subfolder_test
            for img_name in os.listdir(curr_subfolder):
                try:
                    print(img_name)
                    if img_name.endswith('.bmp'):
                        # cv2.imshow('detected circles',iris/255)
                        # cv2.waitKey(0)
                        print(curr_subfolder + img_name)
                        iris = getIrisInfo(curr_subfolder + img_name)
                        calculations, indexes =db.findMostLikely(iris)
                        trueIndex = indexes.index(filename)
                        truePositive.append(calculations[trueIndex])
                        # print(calculations[np.arange(len(calculations))!=trueIndex])
                        # print(falsePositive)
                        falsePositive =np.concatenate( [falsePositive,calculations[np.arange(len(calculations))!=trueIndex]])
                        # plt.imshow(new_img, cmap='gray')
                        # plt.show()
                        # numEyes +=1
                except pupilDetection.BaselineError:
                    print('Baseline Error')
            # if numEyes > 10: break
        except:
            continue
    truePositive = np.mean(truePositive)
    trueNegative = 1 - np.mean(falsePositive)
    print('True Positive:', truePositive)
    print('True Negative:',trueNegative)
