import numpy as np
import glob
import itertools
import cv2
import os
def write_data(train, train_file_name, test, test_file_name, max_size = 10000):
    i = 0
    while i < len(test):
        append_binary_file(train_file_name,train[i:i+max_size].tobytes())
        append_binary_file(test_file_name,test[i:i+max_size].tobytes())
        i += max_size

def append_binary_file(file_name, bytes_):
    with open(file_name,"ab") as f:
        f.write(bytes_)

output_size = 360,80,1
FILEPATH = '/Data/EyePictures/Segmented/'
TRAIN_LABEL_SAVE = FILEPATH + 'train_labels_fine'
TRAIN_INPUT_SAVE = FILEPATH + 'train_images_fine'
TEST_LABEL_SAVE = FILEPATH + 'test_labels_fine'
TEST_INPUT_SAVE = FILEPATH + 'test_images_fine'
subfolder = '1/'

def enumerate_files(filepaths):
    list_people = [None] * len(filepaths)
    for personIndex,filename in enumerate(filepaths):
        list_people[personIndex] = []
        curr_subfolder = filename+'/'+subfolder
        for img_name in os.listdir(curr_subfolder):
            print(img_name)
            if img_name.endswith('.bmp'):
                print(curr_subfolder + img_name)
                list_people[personIndex].append(curr_subfolder + img_name)
    return list_people

def convert_to_bin(filepaths):
    list_people = enumerate_files(filepaths)
    list_data_final = []
    list_labels_final = []
    for p_images in list_people:
        list_data = []
        list_labels = []
        all_combinations = itertools.combinations(p_images,2)
        for i_1, i_2 in all_combinations:
            image_1 = cv2.imread(i_1,0)/255.0
            image_2 = cv2.imread(i_2,0)/255.0
            # print(i_1,i_2)
            x = np.concatenate([np.expand_dims(image_1, axis=-1), np.expand_dims(image_2, axis=-1)], axis = 2)
            list_data.append(x)
            list_labels.append(1)
        for _ in range(len(list_labels)):
            i_1 = p_images[np.random.randint(len(p_images))]
            comparison_person = list_people[np.random.randint(len(p_images))]
            if p_images[0] == comparison_person[0]:
                continue
            i_2 = comparison_person[np.random.randint(len(comparison_person))]
            # print(i_1,i_2)
            image_1 = cv2.imread(i_1,0)/255.0
            image_2 = cv2.imread(i_2,0)/255.0
            x = np.concatenate([np.expand_dims(image_1, axis=-1), np.expand_dims(image_2, axis=-1)], axis = 2)
            list_data.append(x)
            list_labels.append(0)
        list_data_final += list_data
        list_labels_final += list_labels
    return np.array(list_data_final, dtype=np.float32), np.array(list_labels_final, dtype=np.uint8)

if __name__ == '__main__':
    num = 108
    split = .8
    cutoff = int(num * split)
    test_paths = []
    train_paths = []
    for filename in os.listdir(FILEPATH):
        try:
            personIndex = int(filename)
        except:
            continue
        if personIndex <= cutoff:
            train_paths.append(FILEPATH + filename)
        else:
            test_paths.append(FILEPATH + filename)
    train_data, train_labels = convert_to_bin(train_paths)
    test_data, test_labels = convert_to_bin(test_paths)
    print(train_data.dtype)
    print(train_labels.dtype)
    append_binary_file(TRAIN_LABEL_SAVE, train_labels.tobytes())
    append_binary_file(TRAIN_INPUT_SAVE, train_data.tobytes())
    append_binary_file(TEST_LABEL_SAVE, test_labels.tobytes())
    append_binary_file(TEST_INPUT_SAVE, test_data.tobytes())
