import glob, os
from random import shuffle

def common_elements(list1, list2):
    common = []
    for check in list1:
        if check in list2:
            common.append(check)
            print("Common title: {}".format(check))
    return common

# Current directory
current_dir = "/media/ayush/DATA1/Data/stanford_campus_dataset/converted_data/YOLO"

print(current_dir)

# Percentage of images to be used for the test set
percentage_test = 20;

# Create and/or truncate train.txt and test.txt
file_train = open('../../TinyYolo/train.txt', 'w+')  
file_test = open('../../TinyYolo/test.txt', 'w+')

# Populate train.txt and test.txt
counter = 0  
index_test = round(100 / percentage_test)  
print(index_test)

txt_glob = glob.iglob(os.path.join(current_dir, "*.txt"))
jpg_glob = glob.iglob(os.path.join(current_dir, "*.jpg"))
txt_titles = []
jpg_titles = []

for pathAndFilename in txt_glob:
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    txt_titles.append(title)
    print("File title: {}.txt".format(title))

for pathAndFilename in jpg_glob:
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))	
    jpg_titles.append(title)
    print("File title: {}.jpg".format(title))

print(txt_titles)
print(jpg_titles)

common_titles = common_elements(txt_titles, jpg_titles)
shuffle(common_titles)
print("Length of common_titles list: {}".format(len(common_titles)))

for title in common_titles:
    if counter == index_test:
        file_test.write(current_dir + "/" + title + '.jpg' + "\n")
        counter = 0
    else:
        file_train.write(current_dir + "/" + title + '.jpg' + "\n")
    counter = counter + 1

print("Complete")