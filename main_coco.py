from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
import numpy as np
import time
import coco_model
import itertools
from skimage import io, data
import argparse


try:
    import matplotlib.pyplot as plt
except:
    print('Cannot import matplotlib')


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--seed', dest='seed', type=int, default=123456,
                    help='random seed')


def KM_match(transformed, ground_truth):
    from KM import min_KM
    weight = np.zeros((transformed.shape[0], transformed.shape[0]))
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            weight[i, j] = np.mean(np.square(transformed[i, :] - ground_truth[j, :]))
    KMdist, cor = min_KM(weight)
    KMdist /= weight.shape[0]
    return KMdist, cor


args = parser.parse_args()
np.random.seed(args.seed)
nlabels = 5

# load dataset
dataset_path = 'data/'
docs = []
files = os.listdir(dataset_path)

corpus = []
corpus_size = 4000
#corpus_size = 3000

for i in range(corpus_size):
    corpus.append([])

joint_prob = np.zeros((nlabels, nlabels))
npairs = 0
nsel = 0
indexes = []

for fi in files:
    if fi[-3:] != 'png':
        continue
    fi_d = os.path.join(dataset_path, fi)
    img_id, cat_id, in_id = fi.split('.')[0].split('_')

    img_id = int(img_id[3:])
    cat_id = int(cat_id)
    in_id = int(in_id)

    if img_id >= corpus_size:
        continue

    image = io.imread(fi_d)
    corpus[img_id].append((image, cat_id))
    if len(corpus[img_id]) == 2:
        nsel += 1

for i in range(corpus_size):
    if len(corpus[i]) <= 1:
        pass
    else:
        cc = len(corpus[i]) * (len(corpus[i]) - 1)
        for c1 in range(len(corpus[i])):
            for c2 in range(c1):
                joint_prob[corpus[i][c1][1], corpus[i][c2][1]] += 1.0 / cc / nsel
                joint_prob[corpus[i][c2][1], corpus[i][c1][1]] += 1.0 / cc / nsel

print('Joint Prob Matrix')
print(joint_prob)
print(np.sum(joint_prob))


train_step = {
    'epochs': 21,
    'test_logging_step': 100,
    'logging_step': 100,
    'seq_len': 200,
    'markov': 2,
    'embed': 128
}


# pair-wise data
def sample_image_seq(seq_len=10):
    choice = [i for i in range(nlabels)]
    horizon = seq_len
    image = np.zeros((horizon, 128, 128, 3))
    label = np.zeros((horizon, nlabels))
    for i in range(seq_len // 2):
        img_id = np.random.randint(0, corpus_size)
        while (len(corpus[img_id]) <= 1):
            img_id = np.random.randint(0, corpus_size)

        l_id = np.random.randint(0, len(corpus[img_id]))
        r_id = np.random.randint(0, len(corpus[img_id]))
        while r_id == l_id:
            r_id = np.random.randint(0, len(corpus[img_id]))
        image[2 * i, :, :, :] = corpus[img_id][l_id][0]
        label[2 * i, corpus[img_id][l_id][1]] = 1.0
        image[2 * i + 1, :, :, :] = corpus[img_id][r_id][0]
        label[2 * i + 1, corpus[img_id][r_id][1]] = 1.0
    return image, label


def main():
    # GPU configure
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    path = 'coco/'
    if os.path.isdir(path):
        pass
    else:
        print('build new direction ...')
        os.mkdir(path)

    tf.set_random_seed(123456)
    with tf.Session(config=config) as s:
        # CycleGAN Model
        #tf.set_random_seed(123456)
        n_concat = 2

        model = \
            coco_model.UniMappingGAN(s,
                                      seq_len=train_step['seq_len'],
                                      width=128, height=128, nlabels=nlabels,
                                      markov=n_concat)

        # Initializing
        s.run(tf.global_variables_initializer())
        global_step = 0

        mses = []
        for my_iter in range(1000):
            _, pd, kk = s.run([model.e_op, model.loss_inner_product,
                              model.inner_product],
                              feed_dict={model.lr_decay: 1.0})
        for epoch in range(train_step['epochs']):
            # learning rate decay
            lr_decay = 1.
            if epoch >= 100 and epoch % 10 == 0:
                lr_decay = (train_step['epochs'] - epoch) / (
                    train_step['epochs'] / 2.)
            
            for i in range(1000):
                n_decay = 0.5 * max((20000 - global_step + 0.0) / 20000, 0) + 0.5

                for _ in range(model.n_train_critic):
                    a_img, a_label = sample_image_seq(train_step['seq_len'])
                    b_img, b_label = sample_image_seq(train_step['seq_len'])
                    _ = s.run(model.d_op,
                              feed_dict={model.image: a_img,
                                         model.label: b_label,
                                         model.is_training: True,
                                         model.lr_decay: lr_decay,
                                         model.n_decay: n_decay})

                a_img, a_label = sample_image_seq(train_step['seq_len'])
                b_img, b_label = sample_image_seq(train_step['seq_len'])

                w, gp, g_loss, acc, _, mse = \
                    s.run([model.w, model.gp, model.g_loss, model.accuracy,
                           model.g_op, model.g_super_loss],
                          feed_dict={model.image: a_img, model.label: b_label,
                                     model.olabel: a_label,
                                     model.is_training: True,
                                     model.lr_decay: lr_decay,
                                     model.n_decay: n_decay})
                mses.append(mse)

                if global_step % train_step['logging_step'] == 0:
                    # Print loss
                    tr, gt, pred, bd = s.run([model.a2b, model.b, model.pred, model.embeddings],
                                   feed_dict={model.image: a_img,
                                              model.label: a_label,
                                              model.is_training: False, model.n_decay: n_decay})
                    pred = np.argmax(pred, 1)
                    ans = np.argmax(a_label, 1)
                    #kmdist, _ = KM_match(tr, gt)
                    print("[+] Global Step %08d =>" % global_step,
                          " G loss     : {:.8f}".format(g_loss),
                          " w          : {:.8f}".format(w),
                          " gp         : {:.8f}".format(gp),
                          " Accuracy   : {:.8f}".format(acc),
                          " L2 norm    : {:.8f}".format(mse),
                          " KM dist    : {:.8f}".format(0.0))
                    mses = []
                    cm = np.zeros((nlabels, nlabels))
                    for u in range(train_step['seq_len']):
                        cm[ans[u]][pred[u]] += 1
                    print(cm)

                global_step += 1

            d_a, d_a2b = [], []

            #if args.model == 'nodop':
            #    continue
            
            print('[+] Supervised Shuffling, A->B ...')

            horizon = train_step['seq_len']
            myjoint = np.zeros((nlabels, nlabels))
            mycnt = 0.0

            cm = np.zeros((nlabels, nlabels))
            for u in range(5000):
                u_gta, u_ = sample_image_seq(horizon)
                u_gt = np.argmax(u_, 1)
                u_pd = s.run(model.pred, feed_dict={model.image: u_gta,
                                                    model.is_training: False})
                u_fs = np.argmax(u_pd, 1)
                for k in range(horizon // 2):
                    if (k + 1) % train_step['markov'] != 0:
                        myjoint[u_fs[k * 2], u_fs[k * 2 + 1]] += 1.0
                        myjoint[u_fs[k * 2 + 1], u_fs[k * 2]] += 1.0
                        mycnt += 2.0
                        cm[u_gt[k * 2], u_fs[k * 2]] += 1
                        cm[u_gt[k * 2 + 1], u_fs[k * 2 + 1]] += 1
            print(cm)
            myjoint /= mycnt
            print(np.sum(myjoint))

            myweight = np.zeros((nlabels, nlabels))
            for x in range(nlabels):
                for y in range(nlabels):
                    myweight[x, y] = joint_prob[y, x]

            best_se = 1e9
            reord_label = [0] * horizon
            solution = []
            for perm in itertools.permutations([k for k in range(nlabels)]):
                cur_se = 0.
                for x in range(nlabels):
                    for y in range(nlabels):
                        cur_se += np.abs(
                            myjoint[x, y] - myweight[perm[x], perm[y]])
                if cur_se < best_se:
                    best_se = cur_se
                    solution = [perm[k] for k in range(nlabels)]
            print('Discrete Optimization Solution')
            print(solution)

            for mm in range(10):
                d_a, d_a2b = [], []
                for miter in range(10):
                    gta, _ = sample_image_seq(horizon)
                    tb, pd, bd = s.run(
                        [model.a2b, model.pred, model.embeddings],
                        feed_dict={model.image: gta,
                                   model.is_training: False})
                
                    pred_idx = np.argmax(pd, 1)
                    gt_idx = np.argmax(_, 1)

                    for k in range(horizon):
                        reord_label[k] = solution[pred_idx[k]]

                    acc_o = sum([pred_idx[k] == gt_idx[k] for k in range(horizon)])
                    acc_r = sum([reord_label[k] == gt_idx[k]
                                 for k in range(horizon)])
                    if miter == 0:
                        print('Origin Acc {}, Reordered Acc {}, '.
                              format(acc_o, acc_r))

                    gtab = np.zeros((horizon, nlabels))
                    for k in range(horizon):
                        gtab[k, reord_label[k]] = 1.0

                    d_a.append(gta)
                    d_a2b.append(gtab)

                decay = 1.0
                for k in range(200):
                    idx = np.random.randint(10)
                    img, lb = sample_image_seq(train_step['seq_len'])
                    n_decay = max((20000 - global_step + 0.0) / 20000, 0) * 0.5 + 0.5
                    _, ls_ssp = \
                        s.run([model.gs_op, model.g_super_loss],
                              feed_dict={model.image: d_a[idx],
                                         model.label: lb,
                                         model.olabel: d_a2b[idx],
                                         model.lr_decay: decay,
                                         model.is_training: True,
                                         model.n_decay: n_decay})
                    if k % 100 == 0:
                        decay *= 0.8
                    if k % 100 == 0:
                        print('[+] Supervised Shuffling, MSE Loss = {}'.
                              format(ls_ssp))

    # Close tf.Session
    s.close()


if __name__ == '__main__':
    main()
