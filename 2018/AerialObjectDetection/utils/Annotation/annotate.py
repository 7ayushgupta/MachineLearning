import os
import subprocess
import numpy as np
import random
import pickle as cPickle
import cv2
import math
import xml.etree.cElementTree as ET


def assert_path(path, error_message):
    assert os.path.exists(path), error_message


def count_files(path, filename_starts_with=''):
    files = [f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))
                     and f.startswith(filename_starts_with)]
    return len(files)


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def init_directories():
    # Setup the directory structure.
    if not os.path.exists(destination_path):
        os.makedirs(os.path.join(destination_path, 'images_total'))
        os.makedirs(os.path.join(destination_path, 'annotations'))
        os.makedirs(os.path.join(destination_path, 'pickle_store'))

def split_video(video_file, image_name_prefix):
    return subprocess.check_output('ffmpeg -i ' + os.path.abspath(video_file) + ' ' + '-r 1' + ' '+ image_name_prefix +'%d.jpg', shell=True, cwd=os.path.join(destination_path, 'images_total'))


def log(message, level='info'):
    formatters = {
        'GREEN': '\033[92m',
        'END': '\033[0m',
    }
    print ('{GREEN}<'+level+'>{END}\t' + message).format(**formatters)


def write_to_file(filename, content):
    f = open(filename, 'a')
    f.write(content+'\n')

def annotate_frames(sdd_annotation_file, dest_path, filename_prefix, number_of_frames):

    # Pickle the actual SDD annotation
    pickle_file = os.path.join(destination_path, 'pickle_store', filename_prefix + 'annotation.pkl')
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as fid:
            sdd_annotation = cPickle.load(fid)
    else:
        sdd_annotation = np.genfromtxt(sdd_annotation_file, delimiter=' ', dtype=np.str)
        with open(pickle_file, 'wb') as fid:
            cPickle.dump(sdd_annotation, fid)

    # Create VOC style annotation.
    first_image_path = os.path.join(destination_path, 'images_total', filename_prefix+'1.jpg')
    assert_path(first_image_path, 'Cannot find the images. Trying to access: ' + first_image_path)
    first_image = cv2.imread(first_image_path)
    height, width, depth = first_image.shape

    for frame_number in range(1, number_of_frames+1):
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = destination_folder_name
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = 'Stanford Drone Dataset'
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)
        ET.SubElement(annotation, "segmented").text = '0'
        ET.SubElement(annotation, "filename").text = filename_prefix + str(frame_number)

        annotations_in_frame = sdd_annotation[sdd_annotation[:, 5] == str(frame_number)]

        for annotation_data in annotations_in_frame:
            object = ET.SubElement(annotation, "object")
            ET.SubElement(object, "name").text = annotation_data[9].replace('"','')
            ET.SubElement(object, "pose").text = 'Unspecified'
            ET.SubElement(object, "truncated").text = annotation_data[7] # occluded
            ET.SubElement(object, "difficult").text = '0'
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = annotation_data[1]
            ET.SubElement(bndbox, "ymin").text = annotation_data[2]
            ET.SubElement(bndbox, "xmax").text = annotation_data[3]
            ET.SubElement(bndbox, "ymax").text = annotation_data[4]

        xml_annotation = ET.ElementTree(annotation)
        xml_annotation.write(os.path.join(dest_path, filename_prefix + str(frame_number) + ".xml"))

def split():
    init_directories()
    
    for scene in videos_to_be_processed:
        path = os.path.join(dataset_path, 'videos', scene)
        assert_path(path, path + ' not found.')

        videos = videos_to_be_processed.get(scene)

        if len(videos) > 0:
            for video_index in videos.keys():
                video_path = os.path.join(path, 'video' + str(video_index))
                assert_path(video_path, video_path + ' not found.')
                assert count_files(video_path) == 1, video_path+' should contain one file.'

                # Split video into frames
                # Check whether the video has already been made into frames
                jpeg_image_path = os.path.join(destination_path, 'images_total')
                image_name_prefix = scene + '_video' + str(video_index) + '_'
                video_file = os.path.join(video_path, 'video.mov')
                if count_files(jpeg_image_path, image_name_prefix) == 0:
                    # Split Video
                    log('Splitting ' + video_file)
                    split_video(video_file, image_name_prefix)
                    log('Splitting ' + video_file + ' complete.')

                    # Annotate
                    log('Annotating frames from ' + video_file)
                    sdd_annotation_file = os.path.join(dataset_path, 'annotations', scene,
                                                       'video' + str(video_index), 'annotations.txt')
                    assert_path(sdd_annotation_file, 'Annotation file not found. '
                                                     'Trying to access ' + sdd_annotation_file)
                    dest_path = os.path.join(destination_path, 'annotations')
                    number_of_frames = count_files(jpeg_image_path, image_name_prefix)
                    annotate_frames(sdd_annotation_file, dest_path, image_name_prefix, number_of_frames)
                    log('Annotation Complete.')

                else:
                    log(video_file + ' is already split into frames. Skipping...')

    log('Done.')


if __name__ == '__main__':

    # --------------------------------------------------------
    # videos_to_be_processed is a dictionary.
    # Keys in this dictionary should match the 'scenes' in Stanford Drone Dataset.
    # Value for each key is also a dictionary.
    #   - The number of items in the dictionary, can atmost be the number of videos each 'scene'
    #   - Each item in the dictionary is of the form {video_number:fraction_of_images_to_be_split_into_train_val_test_set}
    #   - eg: {2:(.7, .2, .1)} means 0.7 fraction of the images from Video2, should be put into training set,
    #                                0.2 fraction to validation set and
    #                                0.1 fraction to test set.
    #                                Also, training and validation images are merged into trainVal set.
    # --------------------------------------------------------

    # Uniform Sub Sampling : Split should contain only 0 / 1
    videos_to_be_processed = {'bookstore': {1: (1, 0, 0)},
                              'coupa': {0: (1, 0, 0)}}
                            #   'deathCircle': {0: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}}

    # videos_to_be_processed = {'bookstore': {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)},
    #                           'coupa': {0: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)},
    #                           'deathCircle': {0: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}}
                            #   'gates': {0: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)},
                            #   'hyang': {0: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)},
                            #   'little': {0: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)},
                            #   'nexus': {0: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)},
                            #   'quad': {0: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}}

    dataset_path = '/media/ayush/DATA1/Data/stanford_campus_dataset'
    destination_folder_name = 'converted_data'
    destination_path = os.path.join(dataset_path, destination_folder_name)

    # split_and_annotate()
    split()