"""Originally from Tensorflow Github, slightly modified by Mark"""
import os
import pdb
import functools
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3

class Cifar10DataSet:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.is_training = tf.placeholder(tf.bool)

    def make_batch(self):
        """Read the images and labels from 'filenames'."""
        dataset = tf.data.TFRecordDataset(self.get_filenames())
        dataset = dataset.map(self.parser, num_parallel_calls=self.batch_size)

        min_queue_examples = int(
            Cifar10DataSet.num_examples_per_epoch(True) * 0.4)
        # Ensure that the capacity is sufficiently large to provide good random shuffling.
        dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * self.batch_size)
        dataset = dataset.prefetch(buffer_size=2 * self.batch_size)

        # Batch it up.
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        self.iterator = dataset.make_initializable_iterator()
        self.image_batch, self.label_batch = self.iterator.get_next()

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }
        )
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        image = tf.cond(self.is_training,
            true_fn=functools.partial(self.preprocess, image=image),
            false_fn=lambda: tf.identity(image))

        return image, label

    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        # Pad 4 pixels on each dimension of feature map, done in mini-batch
        image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
        image = tf.image.random_flip_left_right(image)
        return image

    def get_filenames(self, data_dir='./data/cifar-10-data'):
        train_records = tf.constant(os.path.join(data_dir, 'train.tfrecords'), dtype=tf.string)
        eval_records = tf.constant(os.path.join(data_dir, 'eval.tfrecords'), dtype=tf.string)
        return tf.cond(self.is_training, lambda: train_records, lambda: eval_records)

    @staticmethod
    def num_examples_per_epoch(is_training=True):
        if is_training:
            return 50000
        else:
            return 10000
