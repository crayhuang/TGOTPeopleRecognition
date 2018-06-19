from skimage import io, transform
import tensorflow as tf
import numpy as np

people_dict = {2: 'snow'}

path1 = "./snow/309.png"

w=100
h=100
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img, (w, h))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data.append(data1)

    saver = tf.train.import_meta_graph('./model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")
    classification_result = sess.run(logits, feed_dict)

    print(classification_result)
    print(tf.argmax(classification_result, 1).eval())

    output = []
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        print(output[i])
        # print("图片", i+1, "预测:" + people_dict[output[i]])