import numpy as np
import pupilDetection
import os
import compareDeps
import cv2

original_directory  = '/Data/EyePictures/CASIA/'
seg_directory = '/Data/EyePictures/Segmented/'
subfolder_train = '1/'
subfolder_test = '2/'
def convert_images():
    for filename in os.listdir(original_directory):
        try:
            personIndex = int(filename)
            print(personIndex)
            curr_subfolder = original_directory + filename+'/'+subfolder_test
            for img_name in os.listdir(curr_subfolder):
                try:
                    print(img_name)
                    if img_name.endswith('.bmp'):
                        print(curr_subfolder + img_name)
                        iris = compareDeps.getIrisInfo(curr_subfolder + img_name)
                        save_path = seg_directory+filename+'/'+subfolder_train + img_name
                        cv2.imwrite(save_path,iris)
                        save_path = seg_directory+img_name
                        cv2.imwrite(save_path,iris)
                except pupilDetection.BaselineError:
                    print('Baseline Error')
            # if numEyes > 10: break
        except:
            continue

def test_accuracy_classic():
    db = compareDeps.iris_db('test.pkl')
    for filename in os.listdir(original_directory):
        try:
            personIndex = int(filename)
            # if personIndex>30:
            #     continue
            print(personIndex)

            curr_subfolder = original_directory + filename+'/'+subfolder_train
            db.addIris(curr_subfolder, '.bmp', filename)
            # if numEyes > 10: break
        except:
            continue
    db.setComparison()
    truePositive = []
    falsePositive = []
    for filename in os.listdir(original_directory):
        try:
            personIndex = int(filename)
            print(personIndex)
            curr_subfolder = original_directory + filename+'/'+subfolder_test
            for img_name in os.listdir(curr_subfolder):
                try:
                    print(img_name)
                    if img_name.endswith('.bmp'):
                        # cv2.imshow('detected circles',iris/255)
                        # cv2.waitKey(0)
                        print(curr_subfolder + img_name)
                        iris = compareDeps.getIrisInfo(curr_subfolder + img_name)
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

if __name__ == "__main__":
    convert_images()
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
