import time
import argparse
import numpy as np
import tensorflow as tf
from loss import dice_coefficient, loss


parser = argparse.ArgumentParser(description='arguments for tensorflow model')
parser.add_argument('--input_size', default=512,
                    help='input size', type=int)
parser.add_argument('--gpu_list', default='0', type=str,
                    help='number of gpus available')
parser.add_argument('--checkpoint_path', default='/tmp/east_icdar2015_resnet_v1_50_rbox/',
                    type=str)
parser.add_argument('--batch_size_per_gpu', default=10, type=int)
parser.add_argument('--num_readers', default=16, type=int)
parser.add_argument('--learning_rate', default=.0001, type=float)
parser.add_argument('--max_steps', default=100000, type=int)
parser.add_argument('--moving_average_decay', default=.997, type=float)
parser.add_argument('--restore', default=False, type=bool)
parser.add_argument('--save_checkpoint_steps', default=1000, type=int)
parser.add_argument('--save_summary_steps', default=100, type=int)
parser.add_argument('--pretrained_model_path', default=None)

FLAGS = parser.parse_args()

from resnet_blocks import EAST
import icdar

gpus = list(range(len(FLAGS.gpu_list.split(','))))

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not os.path.exists(FLAGS.checkpoint_path):
        os.mkdir(FLAGS.checkpoint_path)
    # else:
    #     if not FLAGS.restore:
    #         os.remove(FLAGS.checkpoint_path)
    #         os.mkdir(FLAGS.checkpoint_path)

    global_step = FLAGS.max_steps

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                   epsilon=1e-07, amsgrad=False, name='Adam')

    if FLAGS.pretrained_model_path is not None:
        pass
    if FLAGS.restore:
        print('continue training from previous checkpoint')
        pass
    else:
        if FLAGS.pretrained_model_path is not None:
            pass
    east_model = EAST((FLAGS.input_size, FLAGS.input_size, 3), 
                       batch=FLAGS.batch_size_per_gpu)
    data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                     input_size=FLAGS.input_size,
                                     batch_size=FLAGS.batch_size_per_gpu * len(gpus))

    start = time.time()
    for step in range(FLAGS.max_steps):
        data = next(data_generator)
        x_batch = tf.convert_to_tensor(data[0])
        y_batch_score = tf.convert_to_tensor(data[2])
        y_batch_geo = tf.convert_to_tensor(data[3])
        y_batch_mask = tf.convert_to_tensor(data[4])

        with tf.GradientTape() as tape:
          f_score, f_geometry = east_model(x_batch, training=True)  # Logits for this minibatch
          # Compute the loss value for this minibatch.
          loss_value = loss(y_batch_score, f_score, y_batch_geo, 
                            f_geometry, y_batch_mask)
        grads = tape.gradient(loss_value, east_model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        opt.apply_gradients(zip(grads, east_model.trainable_weights))
          
        if step == 10:
          print('test run complete bud...')
          break

if __name__ == '__main__':
    main()