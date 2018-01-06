import tensorflow as tf
import numpy as np
import pickle
import datetime as dt
import os
import math
import argparse
from six.moves import xrange
from densenet import densenet_model


def load_cifar10(cifar10_directory):
    if not hasattr(load_cifar10, 'cifar10_train_data') or not hasattr(load_cifar10, 'cifar10_train_labels'):
        with open(os.path.join(cifar10_directory, "data_batch_1"), "rb") as f:
            db1 = pickle.load(f, encoding='bytes')
        with open(os.path.join(cifar10_directory, "data_batch_2"), "rb") as f:
            db2 = pickle.load(f, encoding='bytes')
        with open(os.path.join(cifar10_directory, "data_batch_3"), "rb") as f:
            db3 = pickle.load(f, encoding='bytes')
        with open(os.path.join(cifar10_directory, "data_batch_4"), "rb") as f:
            db4 = pickle.load(f, encoding='bytes')
        with open(os.path.join(cifar10_directory, "data_batch_5"), "rb") as f:
            db5 = pickle.load(f, encoding='bytes')
        
        load_cifar10.cifar10_train_data = np.concatenate([db1[b'data'], db2[b'data'], db3[b'data'], db4[b'data'], db5[b'data']], axis=0)
        load_cifar10.cifar10_train_labels = np.array(db1[b'labels'] + db2[b'labels'] + db3[b'labels'] + db4[b'labels'] + db5[b'labels'])

    if not hasattr(load_cifar10, 'cifar10_test_data') or not hasattr(load_cifar10, 'cifar10_test_labels'):
        with open(os.path.join(cifar10_directory, "test_batch"), "rb") as f:
            db1 = pickle.load(f, encoding='bytes')
        load_cifar10.cifar10_test_data = np.array(db1[b'data'])
        load_cifar10.cifar10_test_labels = np.array(db1[b'labels'])

    return load_cifar10.cifar10_train_data, load_cifar10.cifar10_train_labels, load_cifar10.cifar10_test_data, load_cifar10.cifar10_test_labels
    

def generate_train_batch(cifar10_directory, batch_size):
    cifar10_train_data, cifar10_train_labels, _, _ = load_cifar10(cifar10_directory)

    indices = np.arange(len(cifar10_train_labels))
    np.random.shuffle(indices)
    data = cifar10_train_data[indices]
    labels = cifar10_train_labels[indices]

    for batch_index in xrange(0, len(cifar10_train_labels), batch_size):
        yield data[batch_index : batch_index + batch_size], labels[batch_index : batch_index + batch_size]


def generate_test_batch(cifar10_directory, batch_size):
    _, _, cifar10_test_data, cifar10_test_labels = load_cifar10(cifar10_directory)

    indices = np.arange(len(cifar10_test_labels))
    np.random.shuffle(indices)
    data = cifar10_test_data[indices]
    labels = cifar10_test_labels[indices]

    for batch_index in xrange(0, len(cifar10_test_labels), batch_size):
        yield data[batch_index : batch_index + batch_size], labels[batch_index : batch_index + batch_size]


def summarize_scalar(writer, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


def train(sess, global_step, epoch, args, model, train_writer):
    batch_size = args.batch_size
    keep_prob = args.keep_prob
    lr = args.lr
    reg = args.reg
    cifar10_dir = args.cifar10_dir

    step = 0
    print_every_n_steps = 50
    total_train_accuracy = 0.
    total_train_loss = 0.

    for batch_images, batch_labels in generate_train_batch(cifar10_dir, batch_size):
        epoch_step = (step+1) * batch_size / 50000.

        feed_dict = {
            model['X']: batch_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
            model['y']: batch_labels,
            model['keep_prob']: keep_prob,
            model['learning_rate']: lr,
            model['regularization']: reg,
        }

        if step % print_every_n_steps == 0:
            _, batch_acc, batch_loss, train_summary = sess.run([model['train_step'], model['accuracy'], model['loss'], model['summaries']], feed_dict)

            train_writer.add_summary(train_summary, global_step + step)
            train_writer.flush()

            total_train_accuracy += batch_acc
            total_train_loss += batch_loss

            print("{} : step={} epoch_step={:.4f} batch_loss={:.4f} batch_acc={:.4f} train_loss={:.4f} train_acc={:.4f}".format(
                  dt.datetime.now(), global_step + step, epoch_step + epoch - 1, batch_loss, batch_acc, total_train_loss / (step+1), total_train_accuracy / (step+1)), flush=True)
        else:
            _, batch_acc, batch_loss = sess.run([model['train_step'], model['accuracy'], model['loss']], feed_dict)
            total_train_accuracy += batch_acc
            total_train_loss += batch_loss

        step += 1

    total_train_loss /= step
    total_train_accuracy /= step

    summarize_scalar(train_writer, "global_summary/loss", total_train_loss, epoch)
    summarize_scalar(train_writer, "global_summary/accuracy", total_train_accuracy, epoch)
    train_writer.flush()

    return total_train_loss, total_train_accuracy


def test(sess, global_step, epoch, args, model, test_writer):
    keep_prob = args.keep_prob
    lr = args.lr
    reg = args.reg
    cifar10_dir = args.cifar10_dir

    total_test_loss = 0.
    total_correct_test_preds = 0.
    batch_size = 2000

    for X_test, y_test in generate_test_batch(cifar10_dir, batch_size):
        feed_dict = {
            model['X']: X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
            model['y']: y_test,
            model['keep_prob']: keep_prob,
            model['learning_rate']: lr,
            model['regularization']: reg,
        }

        test_correct_preds, test_loss = sess.run([model['correct_predictions'], model['loss']], feed_dict)

        total_correct_test_preds += test_correct_preds
        total_test_loss += test_loss

    total_test_loss /= 5.
    summarize_scalar(test_writer, "global_summary/loss", total_test_loss, epoch)

    total_test_accuracy = total_correct_test_preds / 10000.
    summarize_scalar(test_writer, "global_summary/accuracy", total_test_accuracy, epoch)

    test_writer.flush()

    return total_test_loss, total_test_accuracy


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="mini batch size")
    parser.add_argument('--epochs', type=int, default=300, help="number of epochs to train")
    parser.add_argument('--print_every', type=int, default=50, help="print every n training steps")
    parser.add_argument('--keep_prob', type=float, default=0.8, help="keep probability used during dropout")
    parser.add_argument('--growth_rate', type=int, default=12, help="growth rate")
    parser.add_argument('--reg', type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument('--log_dir', type=str, help="Directory to store tensorboard summary and models.  Note that a timestamp named subdirectory inside --log_dir will be created.")
    parser.add_argument('--cifar10_dir', type=str, help="cifar10 dataset directory.  The five training files and one test file are expected to be named and be identical to how" \
                                                        " they were released, python pickle files named data_batch_n where n=[1,2,3,4,5] and test_batch.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    lr = args.lr 
    batch_size = args.batch_size
    epochs = args.epochs
    print_every_n_steps = args.print_every
    keep_prob = args.keep_prob
    growth_rate = args.growth_rate
    reg = args.reg
    logdir = args.log_dir
    cifar10_dir = args.cifar10_dir

    if logdir == None:
        raise ValueError("Please specify the logdir to specify where to save tensorboard summary and models using --log_dir.")

    if cifar10_dir == None:
        raise ValueError("Please specify the cifar10 dataset directory using --cifar10_dir.")

    model = densenet_model(growth_rate)
    start = dt.datetime.now()
    train_logdir = os.path.join(logdir, "dn-train-" + start.strftime("%Y%m%d-%H%M%S"))
    train_writer = tf.summary.FileWriter(train_logdir)
    test_logdir = os.path.join(logdir, "dn-test-" + start.strftime("%Y%m%d-%H%M%S"))
    test_writer = tf.summary.FileWriter(test_logdir)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        global_step = 0
        num_batches_per_epoch = math.ceil(50000 / batch_size)

        for epoch in xrange(1, epochs + 1):
            if epoch == 150 or epoch == 225:
                lr /= 10.

            total_train_loss, total_train_accuracy = train(sess, global_step, epoch, args, model, train_writer)
            total_test_loss, total_test_accuracy = test(sess, global_step, epoch, args, model, test_writer)
            global_step += num_batches_per_epoch

            print("{} : step={} epoch_step={:.4f} train_loss={:.4f} train_acc={:.4f} test_loss={:.4f} test_acc={:.4f} EPOCH FINISHED={}".format(
                  dt.datetime.now(), global_step, epoch, total_train_loss, total_train_accuracy, total_test_loss, total_test_accuracy, epoch), flush=True)

            saver.save(sess, os.path.join(train_logdir, 'densenet-cifar10'), global_step=epoch)
                
        train_writer.close()
        test_writer.close()

if __name__ == '__main__':
    main()

