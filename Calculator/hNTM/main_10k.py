import tensorflow as tf
import numpy as np
import sys, os, time, math
# from model import TopicModel
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/util')
from sklearn.utils import shuffle
import shutil

# path = "/media/vol/imyiyang/hNTM/"
# os.chdir(path)
# sys.path.append(path)
file_base = "../../10kdata"

sys.path.insert(0, '/media/vol/imyiyang/hNTM/Calculator/hNTM')
sys.path.insert(0, '/media/vol/imyiyang/hNTM/Calculator/hNTM/utils')
from model import TopicModel
from dataset import Dataset
from util import save_configuration


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    tf.flags.DEFINE_boolean("restore", False, "restore from the last check point")
    tf.flags.DEFINE_integer("extreme", 2000, "filter_extreme in vocabulary")
    tf.flags.DEFINE_string("full", "full", "full tokens of 10K report or not")
    tf.flags.DEFINE_string("dir_logs", "./out/10k_32_16_8/", "out directory")
    tf.flags.DEFINE_string("num_topics", "32 16 8", "number of topics (separated by space)")
    tf.flags.DEFINE_string("layer_sizes", "512 256 128", "size of all latent layers (separated by space)")
    tf.flags.DEFINE_string("embedding_sizes", "100 50 20",
                           "size of embeddings in the topic-word distribution matrices (separated by space)")
    tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
    tf.flags.DEFINE_integer("year", 2014, "Year of 10K reports")
    tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs")
    tf.flags.DEFINE_integer("use_kl", 0, "none: 0; top one: 1; all: 2")
    FLAGS = tf.flags.FLAGS

    parameters = {
        'extreme': FLAGS.extreme,
        'dir_logs': FLAGS.dir_logs,
        'num_topics': FLAGS.num_topics,
        'layer_sizes': FLAGS.layer_sizes,
        'embedding_sizes': FLAGS.embedding_sizes,
        'batch_size': FLAGS.batch_size,
        'year': FLAGS.year,
        'full': FLAGS.full,
        'num_epochs': FLAGS.num_epochs,
        'use_kl': FLAGS.use_kl,
    }
    dirname = os.path.join(file_base, FLAGS.dir_logs)
    print(dirname)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)
    save_configuration(parameters)

    # python main_gaus_softmax.py --dir_logs ./out/20news_gaus_softmax_kl0/

    num_topics = [int(e) for e in FLAGS.num_topics.strip().split()]
    layer_sizes = [int(e) for e in FLAGS.layer_sizes.strip().split()]
    embedding_sizes = [int(e) for e in FLAGS.embedding_sizes.strip().split()]

    FULL = FLAGS.full
    YEAR = FLAGS.year
    EXTREME = FLAGS.extreme
    N_EPOCHS = FLAGS.num_epochs
    BATCH_SIZE = FLAGS.batch_size
    FILE_OF_CKPT = os.path.join(dirname, "model.ckpt")
    print(FILE_OF_CKPT)

    # learning rate decay
    STARTER_LEARNING_RATE = FLAGS.learning_rate
    DECAY_AFTER = 2
    DECAY_INTERVAL = 5
    DECAY_FACTOR = 0.97

    # warming-up coefficient for KL-divergence term
    Nt = 50  # warmig-up during the first 2Nt epochs
    # _lambda_z_wu = np.concatenate((np.zeros(Nt), np.linspace(0, 1, Nt)))
    _lambda_z_wu = np.linspace(0, 1, Nt)

    d = Dataset(vocab_size=EXTREME, year=YEAR, for_training=True, batch_size=BATCH_SIZE, full=FULL)

    with tf.Graph().as_default() as g:

        ###########################################
        """        Build Model Graphs           """
        ###########################################
        with tf.variable_scope("topicmodel") as scope:
            m = TopicModel(d, FLAGS.use_kl, all_likelihood=True,
                           latent_sizes=num_topics, layer_sizes=layer_sizes, embedding_sizes=embedding_sizes)
            # latent_sizes = [64], layer_sizes = [512], embedding_sizes = [100])
            print('built the graph for training.')
            scope.reuse_variables()

        ###########################################
        """              Init                   """
        ###########################################
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        saver = tf.train.Saver()
        _lr, ratio = STARTER_LEARNING_RATE, 1.0
        Perp_all = []
        Perp_perdoc = []

        if FLAGS.restore:
            print("... restore from the last check point.")
            saver.restore(sess, FILE_OF_CKPT)
        ###########################################
        """         Training Loop               """
        ###########################################
        print('... start training')
        tf.train.start_queue_runners(sess=sess)
        best_perp = math.inf
        for epoch in range(1, N_EPOCHS + 1):
            X_train = shuffle(d.train_x_bow)
            X_test = shuffle(d.test_x_bow)
            # set coefficient of warm-up
            idx = -1 if Nt <= epoch else epoch
            time_start = time.time()
            perp_all = [0.0] * m.L
            perp_perdoc = [0.0] * m.L
            for i in range(d.num_train_batch):

                feed_dict = {m.x: X_train[i * d.batch_size: (i + 1) * d.batch_size],
                             m.lr: _lr,
                             m.lambda_z_wu: _lambda_z_wu[idx],
                             m.is_train: True}

                """ do update """
                r, op, current_lr = sess.run([m.out, m.op, m.lr], feed_dict=feed_dict)
                for l in range(m.L):
                    perp_all[l] += r['perp_all'][l]
                    perp_perdoc[l] += r['perp_perdoc'][l]
            for l in range(m.L):
                perp_all[l] /= d.num_train_batch
                perp_perdoc[l] /= d.num_train_batch
            elapsed_time = time.time() - time_start
            print(" epoch:%2d, train loss: %s, likelihood: %s, KL: %s, perp_all: %s, perp_perdoc: %s, time:%.3f" %
                  (epoch, r['loss'], r['Lr'], r['kl'], perp_all, perp_perdoc, elapsed_time))

            """ test """
            time_start = time.time()
            perp_all = [0.0] * m.L
            perp_perdoc = [0.0] * m.L
            for i in range(d.num_test_batch):
                feed_dict = {m.x: X_test[i * d.batch_size: (i + 1) * d.batch_size],
                             m.lambda_z_wu: _lambda_z_wu[idx],
                             m.is_train: False}
                r = sess.run([m.out], feed_dict=feed_dict)[0]
                for l in range(m.L):
                    perp_all[l] += r['perp_all'][l]
                    perp_perdoc[l] += r['perp_perdoc'][l]
            for l in range(m.L):
                perp_all[l] /= d.num_test_batch
                perp_perdoc[l] /= d.num_test_batch
            Perp_all.append(perp_all[0])
            Perp_perdoc.append(perp_perdoc[0])

            elapsed_time = time.time() - time_start
            print(" epoch:%2d, test loss: %s, likelihood: %s, KL: %s, perp_all: %s, perp_perdoc: %s, time:%.3f" %
                  (epoch, r['loss'], r['Lr'], r['kl'], perp_all, perp_perdoc, elapsed_time))

            """ save """
            if epoch % 10 == 0:
                print("Model saved in file: %s" % saver.save(sess, FILE_OF_CKPT))
            if perp_all[0] < best_perp:
                best_perp = perp_all[0]
                print("Model saved in file: %s" % saver.save(sess, os.path.join(dirname, "best-model.ckpt")))

            print("best test perp: ", best_perp)

            """ learning rate decay"""
            if (epoch % DECAY_INTERVAL == 0) and (epoch > DECAY_AFTER):
                ratio *= DECAY_FACTOR
                _lr = STARTER_LEARNING_RATE * ratio
                print('lr decaying is scheduled. epoch:%d, lr:%f <= %f' % (epoch, _lr, current_lr))

            np.save(os.path.join(dirname, "perp_all.npy"), Perp_all)
            np.save(os.path.join(dirname, "perp_perdoc.npy"), Perp_perdoc)

        sess.close()


main()
