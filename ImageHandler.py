# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from PIL import Image

writer = tf.python_io.TFRecordWriter("train.tfrecords")

images_path = "./snow/"

classes = {'snow'}

for index, name in enumerate(classes):
    for img_name in os.listdir(images_path):
        img_path = images_path + img_name
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())

writer.close()

for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value

    print("image is : %s" % image)
    print("label is : %s" % label)