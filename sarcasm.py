import tensorflow as tf
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import gensim
import pandas as pd
import gzip
from tqdm import tqdm
# import ipdb
import os
import logging
import random
import shutil
from sklearn.utils import shuffle

from gensim.models.wrappers import FastText
max_word_count = 20
data_split = [0.8, 0.2, 0]
class Params(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if not hasattr(self, key):
                # exec(f"self.{key} = {val}")
                setattr(self, key, val)
training = False
test = True
params = Params(
    batch_size=64,
    dropout_keep_prob=1,
    learning_rate=0.0001,
    report_step=10,
    save_step=100,
    num_epochs=50,
    lstmUnits=[256, 128, 64],
    fc_layer_units=[128, 64, 32],
    output_classes=1,
)
class DataClass(object):
    def __init__(self):
        # add data imports
        self.data_df = pd.read_csv("sarcasm/train-balanced-sarcasm.csv")
        self.data_split = data_split
        # self.stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
        #                    'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or',  'who', 'as', 'from', 'him', 'each', 'the', 'themselves',
        #                    'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their',
        #                    'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then',
        #                    'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just',
        #                    'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
        #                    'further', 'was', 'here', ]
        self.stop_words = []
        print("Loading Vectors")
        self.vec_model = FastText.load_fasttext_format(
            '../../Science_Fair_Project/vectors/cc.en.300.bin/cc.en.300.bin')
        # self.vec_model = {}
        print("Completed Loading Vectors")
        # ipdb.set_trace()
        self.data_df = self.data_df[["comment", "parent_comment", "label"]]
        self.data_df = shuffle(self.data_df)
        self.data_df["label"] = self.data_df["label"].astype(int)
        self.data_df["comment"] = self.data_df["comment"].astype(str)
        self.data_df["comment"] = self.data_df["comment"].str.lower()
        self.data_df["comment"] = self.data_df["comment"].str.strip(
            to_strip=".!?,")
        self.data_df["comment"] = self.data_df["comment"].str.split()
        self.data_df["parent_comment"] = self.data_df["parent_comment"].astype(str)
        self.data_df["parent_comment"] = self.data_df["parent_comment"].str.lower()
        self.data_df["parent_comment"] = self.data_df["parent_comment"].str.strip(
            to_strip=".!?,")
        self.data_df["parent_comment"] = self.data_df["parent_comment"].str.split()

        # self.data_df["text"] = self.data_df["text"].apply(
        #     lambda x: [w for w in x if not w in self.stop_words])

    def replace_by_word_embeddings(self, row):
        # ipdb.set_trace()
        new_arr = np.zeros((max_word_count, 300))
        for w, word in enumerate(row):
            try:
                new_arr[w] = self.vec_model.wv[word]
            except Exception as e:
                new_arr[w] = np.zeros((300))
        return new_arr.astype(np.float32)

    def get_data_as_df(self, data_split):

        def get_generator(data_split):
            i = 0
            counter = 0
            data_split = list(map(lambda x: int(round(x*len(self.data_df))), data_split))
            # df = {}
            # for d in tqdm(parse(path)):
            #     if counter > start_examples:
            #         df[i] = d
            #         i += 1
            #     if counter > stop_examples:
            #         break
            #     else:
            #         counter += 1
            def padding_function(row):
                # ipdb.set_trace()
                if len(row) < max_word_count:
                    row += ["" for i in range(max_word_count - len(row))]
                else:
                    row = row[:max_word_count]
                return row

            def train_data():
                for i in range(data_split[0]):
                    ret_df = self.data_df.iloc[i].copy()
                    ret_df["comment"] = padding_function(ret_df["comment"])
                    ret_df["parent_comment"] = padding_function(ret_df["parent_comment"])

                    # ipdb.set_trace()

                    ret_df["comment"] = self.replace_by_word_embeddings(
                        ret_df["comment"])
                    ret_df["parent_comment"] = self.replace_by_word_embeddings(
                        ret_df["parent_comment"])

                    yield (ret_df["comment"], ret_df["parent_comment"], [ret_df["label"]])
            def val_data():
                for i in range(data_split[0], data_split[1] + data_split[0]):
                    ret_df = self.data_df.iloc[i].copy()
                    ret_df["comment"] = padding_function(ret_df["comment"])
                    ret_df["parent_comment"] = padding_function(ret_df["parent_comment"])

                    # ipdb.set_trace()

                    ret_df["comment"] = self.replace_by_word_embeddings(
                        ret_df["comment"])
                    ret_df["parent_comment"] = self.replace_by_word_embeddings(
                        ret_df["parent_comment"])

                    yield (ret_df["comment"], ret_df["parent_comment"], [ret_df["label"]])
            # for i in range(max_word_count):
            #     # ipdb.set_trace()
            #     temp_dict = {}
            #     temp_dict[str(i)] = self.data_df["text"][:, i]
            #     self.data_df = self.data_df.assign(**temp_dict)

            # self.data_df.drop(["text"])

            # data_df["text"] = pd.to_numeric(data_df["text"])
            # data_df["label"] = pd.to_numeric(data_df["label"])
            # ipdb.set_trace()
            return train_data, val_data
        return get_generator(data_split)
batch_size = params.batch_size
lstmUnits = params.lstmUnits

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def get_sent(vectors):
    word_list = []
    for i in vectors:
        if (i != np.zeros([300])).all():
            word_list.append(
                data_class.vec_model.similar_by_vector(i)[0][0])
    return " ".join(word_list)


def create_lstm_cell(units):
    # lstmCell = tf.contrib.rnn.LayerNormBasicLSTMCell(
    #     units, activation=tf.nn.sigmoid, dropout_keep_prob=params.dropout_keep_prob,)
    lstmCell = tf.nn.rnn_cell.LSTMCell(
        units, activation=tf.nn.tanh, initializer=tf.contrib.layers.xavier_initializer())
    lstmCell = tf.nn.rnn_cell.DropoutWrapper(
        lstmCell, params.dropout_keep_prob)
    # lstmCell = tf.contrib.rnn.AttentionCellWrapper(lstmCell, 5)
    return lstmCell
def build_rnn(comment, scope):
    data_x_tri = tf.layers.conv1d(comment, filters=150, kernel_size=[
        3], padding="same", activation=tf.nn.elu)
    data_x_bi = tf.layers.conv1d(comment, filters=150, kernel_size=[
        2], padding="same", activation=tf.nn.elu)
    data_x = tf.concat((data_x_tri, data_x_bi), 2)
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [create_lstm_cell(num_units) for num_units in lstmUnits])
    value, _ = tf.nn.dynamic_rnn(
        cell, data_x, dtype=tf.float32, sequence_length=length(data_x), scope=scope)
    dense = tf.layers.flatten(value)
    return dense
def build_model(comment, parent_comment):
    # comment = tf.layers.batch_normalization(comment, training=training)
    # parent_comment = tf.layers.batch_normalization(parent_comment, training=training)
    # data_x_tri = tf.layers.conv1d(comment, filters=150, kernel_size=[
    #     3], padding="same", activation=tf.nn.elu)
    # data_x_bi = tf.layers.conv1d(comment, filters=150, kernel_size=[
    #     2], padding="same", activation=tf.nn.elu)
    # data_x = tf.concat((data_x_tri, data_x_bi), 2)
    # data_x_tri_parent = tf.layers.conv1d(comment, filters=150, kernel_size=[
    #     3], padding="same", activation=tf.nn.elu)
    # data_x_bi_parent = tf.layers.conv1d(parent_comment, filters=150, kernel_size=[
    #     2], padding="same", activation=tf.nn.elu)
    # data_x_parent = tf.concat((data_x_tri_parent, data_x_bi_parent), 2)
    # single = tf.concat((data_x, data_x_parent), axis=1)
    rnn_comment = build_rnn(comment, "rnn_comment")
    rnn_comment = tf.layers.dense(rnn_comment, 2048, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    rnn_parent_comment = build_rnn(parent_comment, "parent_comment")
    rnn_parent_comment = tf.layers.dense(rnn_parent_comment, 2048, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())

    rnn_combined = tf.concat((rnn_comment, rnn_parent_comment), -1)
    # rnn_combined = build_rnn(single, "rnn")
    dense = tf.layers.dense(rnn_combined, 1024, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    dense = tf.layers.dropout(dense, rate=0.1)
    # dense = tf.layers.batch_normalization(dense, training=training)
    dense = tf.layers.dense(dense, 256, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    dense = tf.layers.dropout(dense, rate=0.15)
    # dense = tf.layers.batch_normalization(dense, training=training)
    dense = tf.layers.dense(dense, 1, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return dense, tf.round(dense)

if training:
    # ipdb.set_trace()
    data_class = DataClass()
    train, val = data_class.get_data_as_df(data_split)
    train_dataset = tf.data.Dataset.from_generator(
        train, (tf.float32, tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.from_generator(
        val, (tf.float32, tf.float32, tf.float32), (tf.TensorShape([max_word_count, 300]), tf.TensorShape([max_word_count, 300]), tf.TensorShape([1, ])))
    # val_dataset = val_dataset.shuffle(split[1] // 30)
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(batch_size)


    handle = tf.placeholder(tf.string, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    data_comment, data_parent_comment, data_y = iterator.get_next()

    data_one_hot = tf.reshape(data_y, (-1, 1))

    data_comment = tf.reshape(data_comment, (-1, max_word_count, 300))
    data_parent_comment = tf.reshape(data_parent_comment, (-1, max_word_count, 300))
    logits, predictions = build_model(data_comment, data_parent_comment)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=data_one_hot, logits=logits))
    tf.summary.scalar("loss", loss)
    _optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = _optimizer.minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        data_one_hot, predictions), tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # ipdb.set_trace()
        model_type = "RNN"
        train_writer = tf.summary.FileWriter(
            f"tensorboard_{model_type}", sess.graph)
        train_iterator = train_dataset.make_one_shot_iterator()
        val_iterator = val_dataset.make_one_shot_iterator()

        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())
        # ipdb.set_trace()
        tf.logging.set_verbosity(10)
        tf.logging.info("Starting Training")
        for epoch in range(params.num_epochs):
            # range(len(os.listdir("train_data/")) // params.batch_size + 1)):
            for iteration in range(100000 // (params.batch_size + 1)):
                try:
                    # ipdb.set_trace()
                    _accuracy, _loss, _ = sess.run(
                        [accuracy, loss, optimizer], feed_dict={handle: train_handle})
                    validation_acc, validation_loss = sess.run([accuracy, loss], feed_dict={
                        handle: val_handle})
                    if iteration % params.report_step == 0:
                        summary = sess.run(merged, feed_dict={
                            handle: train_handle})
                        train_writer.add_summary(
                            summary, iteration * epoch + 1)
                        # print_val_truth = get_sent(sess.run(data_x, feed_dict={
                        #     handle: test_handle})[0])
                        tf.logging.info(
                            f"Iteration: {iteration*(epoch+1)}, Loss: {_loss}, Accuracy: {_accuracy}, Validation Accuracy: {validation_acc}, Validation Loss: {validation_loss}")#, Test Ground Prediction: {print_val_pred}, {ground_truth}")
                    if (iteration) % params.save_step == 0 and iteration > 0:
                        save_path = saver.save(
                            sess, f"./tensorboard_{model_type}/model{iteration*(epoch+1)}.ckpt")
                        tf.logging.info(
                            f"Saved Checkpoint at iteration {iteration*(epoch+1)} and path {save_path}")
                except tf.errors.OutOfRangeError:
                    saver.save(
                        sess, f"./tensorboard_{model_type}/model{iteration*(epoch+1)}.ckpt")
                    sys.exit()
            # ipdb.set_trace()
            tf.logging.info(
                "################################################################################################")
            tf.logging.info(f"Finished Epoch {epoch}")
            tf.logging.info(
                f"Validation Accuracy: {validation_acc}, Validation Loss: {validation_loss}")
        # predict_during_train_iter(sess)
        save_path = saver.save(
            sess, f"./tensorboard_{model_type}/model{iteration*epoch}.ckpt")
        tf.logging.info(
            f"Saved Checkpoint at iteration {iteration*epoch} and path {save_path}")
elif test == True:
    def predict_text(text):
        def padding_function(row):
            # ipdb.set_trace()
            if len(row) < max_word_count:
                row += ["" for i in range(max_word_count - len(row))]
            else:
                row = row[:max_word_count]
            return row
        comment_list = text.lower().strip("!.?,").split()
        # comment_list = list(
        #     filter(lambda x: x not in data_class.stop_words, comment_list))
        comment_list = padding_function(comment_list)

        parent_comment_list = text.lower().strip("!.?,").split()
        # parent_comment_list = list(
        #     filter(lambda x: x not in data_class.stop_words, parent_comment_list))
        parent_comment_list = padding_function(parent_comment_list)
        return 0
else:
    with tf.Session() as sess:
        test_comment = tf.placeholder(
            tf.float32, shape=(None, max_word_count, 300))
        test_parent_comment = tf.placeholder(
            tf.float32, shape=(None, max_word_count, 300))
        output = build_model(test_comment, test_parent_comment)
        saver = tf.train.Saver()
        saver.restore(sess, "tensorboard_RNN/model1100.ckpt")
        data_class = DataClass()
        def predict_text(comment, parent_comment):
            ipdb.set_trace()
            def padding_function(row):
                # ipdb.set_trace()
                if len(row) < max_word_count:
                    row += ["" for i in range(max_word_count - len(row))]
                else:
                    row = row[:max_word_count]
                return row
            comment_list = comment.lower().strip("!.?,").split()
            # comment_list = list(
            #     filter(lambda x: x not in data_class.stop_words, comment_list))
            comment_list = padding_function(comment_list)
            comment_list = data_class.replace_by_word_embeddings(comment_list)

            parent_comment_list = parent_comment.lower().strip("!.?,").split()
            # parent_comment_list = list(
            #     filter(lambda x: x not in data_class.stop_words, parent_comment_list))
            parent_comment_list = padding_function(parent_comment_list)
            parent_comment_list = data_class.replace_by_word_embeddings(parent_comment_list)

            return sess.run(output, feed_dict={test_comment: [comment_list], test_parent_comment:[parent_comment_list]})
