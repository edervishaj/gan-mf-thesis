import os
import json
import shutil
from warnings import simplefilter

from datasets.Movielens import Movielens
from recommenders.CFGAN import CFGAN
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Utils_ import plot_loss_acc, plot
from keras_utils import save_models, set_seed

# Supress Tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)

use_gpu = True
verbose = False
only_build = False
transposed = False

if not use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Seed for reproducibility of results
seed = 1337
set_seed(seed)

reader = Movielens(version='100K', use_local=True, implicit=True, verbose=False, seed=seed, split_ratio=[0.8, 0.2, 0.])
from Dummy import read_cfgan

URM_train = reader.get_URM_train(transposed=transposed)
URM_validation = reader.get_URM_validation(transposed=transposed)
URM_test = reader.get_URM_test(transposed=transposed)

# URM_train, URM_test, _, _ = read_cfgan('../CFGAN/datasets/ML100K/ML100K.train', '../CFGAN/datasets/ML100K/ML100K.test', transposed=False)
URM_train = URM_train.T.tocsr()

evaluator = EvaluatorHoldout(URM_test, [5, 20], exclude_seen=True)
# evaluatorValidation = EvaluatorHoldout(URM_test, [5], exclude_seen=True)
evaluatorValidation = EvaluatorHoldout(URM_validation, [5], exclude_seen=True)

gan = CFGAN(*URM_train.shape, verbose=verbose, seed=seed, d_nodes=[125], g_nodes=[400])

# Save this script inside the codesdir of the model
shutil.copy(os.path.abspath(__file__), os.path.join(gan.logsdir, 'code'))

# add dataset config to the folder where the model config is saved
with open(os.path.join(gan.logsdir, 'dataset_config.txt'), 'w') as f:
    json.dump(reader.config, f, indent=4)

gan.fit(URM_train=URM_train,
        epochs=25,
        sample_every=5,
        verbose=False,
        D_batch_size=64,
        G_batch_size=32,
        k_d=2,
        k_g=4,
        G_lr=1e-4,
        D_lr=1e-4,
        G_reg=1e-3,
        D_reg=0,
        neg_sample_ratio=0.7,
        zr_coefficient=0.03,
        allow_worse=10,
        validation_set=URM_test,
        validation_evaluator=evaluator,
        only_build=only_build,
        callback=plot_loss_acc)

# print(evaluate(gan, URM_train, URM_test, np.arange(URM_test.shape[0]), K=np.array([5, 20]), parallel=False))

if not only_build:
    save_models(gan)
    results_dic, results_run_string = evaluator.evaluateRecommender(gan)
    print(results_run_string)
    with open(os.path.join(gan.logsdir, 'results.txt'), 'w') as f:
        f.write(results_run_string)

    # Rename folder for easy access to best performing models
    map_folder = os.path.join('plots', gan.RECOMMENDER_NAME, 'MAP_' + str(results_dic[5]['MAP'])[:7])
    if os.path.exists(map_folder):
        shutil.rmtree(map_folder)
    # os.replace(src=gan.logsdir, dst=map_folder)
    shutil.move(src=gan.logsdir, dst=map_folder)

    # Run this on terminal
    # print('tensorboard --logdir=' + map_folder + '/tsb')

