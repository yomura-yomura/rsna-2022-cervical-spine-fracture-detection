import gc
import glob
import os
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.make_data import effnet_data
import utils
from src.model import custom_metric
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import warnings
import wandb
from pathlib import Path

warnings.simplefilter('ignore')
# SET CONFIG Effnet

cfg = utils.load_yaml(Path("../../config/config.yaml"))
#DATA PATH
RSNA_2022_PATH = cfg["data"]["RSNA_2022_PATH"]
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH = f'{RSNA_2022_PATH}/test_images'
EFFNET_CHECKPOINTS_PATH = cfg["data"]["EFFNET_CHECKPOINTS_PATH"]
METADATA_PATH = cfg["data"]["METADATA_PATH"]

#PARAMETER OF EFFNET
EFFNET_MAX_TRAIN_BATCHES = int(cfg["model"]["EFFNET_MAX_TRAIN_BATCHES"])
EFFNET_MAX_EVAL_BATCHES = int(cfg["model"]["EFFNET_MAX_EVAL_BATCHES"])
ONE_CYCLE_MAX_LR = float(cfg["model"]["ONE_CYCLE_MAX_LR"])
ONE_CYCLE_PCT_START = float(cfg["model"]["ONE_CYCLE_PCT_START"])
SAVE_CHECKPOINT_EVERY_STEP = int(cfg["model"]["SAVE_CHECKPOINT_EVERY_STEP"])
FRAC_LOSS_WEIGHT = float(cfg["model"]["FRAC_LOSS_WEIGHT"])
PREDICT_MAX_BATCHES = float(cfg["model"]["PREDICT_MAX_BATCHES"])
N_FOLDS = int(cfg["model"]["N_FOLDS"])
ONE_CYCLE_EPOCH = int(cfg["model"]["ONE_CYCLE_EPOCH"])
SEED = int(cfg["model"]["SEED"])
WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT

# Common
PROJECT_NAME = cfg["base"]["PROJECT_NAME"]
MODEL_NAME = cfg["base"]["MODEL_NAME"]
os.environ["WANDB_MODE"] = "online"
os.environ['WANDB_API_KEY'] = '2ba0031ab3db40ffc8d6c24ff43c9f3d51eabd04'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    BATCH_SIZE = cfg["model"]["BATCH_SIZE"]
else:
    BATCH_SIZE = 2

#Read csv data for slicing
df_train = pd.read_csv(f'{RSNA_2022_PATH}/train.csv')
df_train_slices = pd.read_csv(f'{METADATA_PATH}/train_segmented.csv')
df_test = pd.read_csv(f'{RSNA_2022_PATH}/test.csv')

#PreProcess and Effnetdata
df_train,df_train_slices,df_test,df_test_slices = effnet_data.preprocess(df_train = df_train,df_train_slices=df_train_slices,df_test=df_test,TEST_IMAGES_PATH=TEST_IMAGES_PATH,N_FOLDS=N_FOLDS)
ds_train = effnet_data.EffnetDataSet(df_train, TRAIN_IMAGES_PATH, WEIGHTS.transforms())
ds_test = effnet_data.EffnetDataSet(df_test, TEST_IMAGES_PATH, WEIGHTS.transforms())

class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s(weights=WEIGHTS)
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )

    def forward(self, x):
        # returns logits
        x = self.model(x)['flatten']
        return self.nn_fracture(x), self.nn_vertebrae(x)

    def predict(self, x):
        frac, vert = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)

def evaluate_effnet(model: EffnetModel, ds, max_batches=PREDICT_MAX_BATCHES, shuffle=False):
    torch.manual_seed(SEED)
    model = model.to(DEVICE)
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=os.cpu_count(),
                                          collate_fn=utils.filter_nones)
    pred_frac = []
    pred_vert = []
    with torch.no_grad():
        model.eval()
        frac_losses = []
        vert_losses = []
        with tqdm(dl_test, desc='Eval', miniters=1) as progress:
            for i, (X, y_frac, y_vert) in enumerate(progress):
                with autocast():
                    y_frac_pred, y_vert_pred = model.forward(X.to(DEVICE))
                    frac_loss = custom_metric.weighted_loss(y_frac_pred, y_frac.to(DEVICE),DEVICE=DEVICE).item()

                    #Classification of Bones
                    vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE)).item()
                    pred_frac.append(torch.sigmoid(y_frac_pred))
                    pred_vert.append(torch.sigmoid(y_vert_pred))
                    frac_losses.append(frac_loss)
                    vert_losses.append(vert_loss)
                if i >= max_batches:
                    break
        return np.mean(frac_losses), np.mean(vert_losses), torch.concat(pred_frac).cpu().numpy(), torch.concat(pred_vert).cpu().numpy()

def train_effnet(ds_train, ds_eval, logger, name):
    torch.manual_seed(SEED)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(),
                                           collate_fn=utils.filter_nones)
    model = EffnetModel().to(DEVICE)
    optim = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=ONE_CYCLE_MAX_LR, epochs=ONE_CYCLE_EPOCH,
                                                    steps_per_epoch=min(EFFNET_MAX_TRAIN_BATCHES, len(dl_train)),
                                                    pct_start=ONE_CYCLE_PCT_START)
    model.train()
    scaler = GradScaler()
    with tqdm(dl_train, desc='Train', miniters=1) as progress:
        for batch_idx, (X, y_frac, y_vert) in enumerate(progress):

            if ds_eval is not None and batch_idx % SAVE_CHECKPOINT_EVERY_STEP == 0 and EFFNET_MAX_EVAL_BATCHES > 0:
                frac_loss, vert_loss = evaluate_effnet(
                    model, ds_eval, max_batches=EFFNET_MAX_EVAL_BATCHES, shuffle=True)[:2]
                model.train()
                logger.log(
                    {'eval_frac_loss': frac_loss, 'eval_vert_loss': vert_loss, 'eval_loss': frac_loss + vert_loss})
                if batch_idx > 0:  # don't save untrained model
                    utils.save_model(name, model)

            if batch_idx >= EFFNET_MAX_TRAIN_BATCHES:
                break
            optim.zero_grad()
            # Using mixed precision training
            with autocast():
                y_frac_pred, y_vert_pred = model.forward(X.to(DEVICE))
                frac_loss = custom_metric.weighted_loss(y_frac_pred, y_frac.to(DEVICE),DEVICE=DEVICE)
                vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE))
                loss = FRAC_LOSS_WEIGHT * frac_loss + vert_loss

                if np.isinf(loss.item()) or np.isnan(loss.item()):
                    print(f'Bad loss, skipping the batch {batch_idx}')
                    del loss, frac_loss, vert_loss, y_frac_pred, y_vert_pred
                    utils.gc_collect()

            # scaler is needed to prevent "gradient underflow"
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            progress.set_description(f'Train loss: {loss.item() :.02f}')
            logger.log({'loss': (loss.item()), 'frac_loss': frac_loss.item(), 'vert_loss': vert_loss.item(),
                        'lr': scheduler.get_last_lr()[0]})
    utils.save_model(name, model)
    return model


def gen_effnet_predictions(effnet_models, df_train):
    if os.path.exists(os.path.join(EFFNET_CHECKPOINTS_PATH, 'train_{PROJECT_NAME}_{MODEL_NAME}_predictions.csv')):
        print('Found cached version of train_predictions.csv')
        df_train_predictions = pd.read_csv(os.path.join(EFFNET_CHECKPOINTS_PATH, 'train_{PROJECT_NAME}_{MODEL_NAME}_predictions.csv'))
    else:
        df_train_predictions = []
        df_eval_predictions = []
        with tqdm(enumerate(effnet_models), total=len(effnet_models), desc='Folds') as progress:
            for fold, effnet_model in progress:
                ds_eval = effnet_data.EffnetDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())
                ds_train = effnet_data.EffnetDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())

                #train_prediction
                train_frac_loss, train_vert_loss, train_effnet_pred_frac, train_effnet_pred_vert = evaluate_effnet(effnet_model, ds_train, PREDICT_MAX_BATCHES)
                progress.set_description(f'Fold score:{train_frac_loss:.02f}')
                df_train_effnet_pred = pd.DataFrame(data=np.concatenate([train_effnet_pred_frac, train_effnet_pred_vert], axis=1),
                                              columns=[f'C{i}_effnet_frac' for i in range(1, 8)] +
                                                      [f'C{i}_effnet_vert' for i in range(1, 8)])
                df_train = pd.concat(
                    [df_train.query('split != @fold').head(len(df_train_effnet_pred)).reset_index(drop=True), df_train_effnet_pred],
                    axis=1
                ).sort_values(['StudyInstanceUID', 'Slice'])
                df_train_predictions.append(df_train)

                #valid_prediction
                eval_frac_loss, eval_vert_loss, eval_effnet_pred_frac, eval_effnet_pred_vert = evaluate_effnet(effnet_model, ds_eval, PREDICT_MAX_BATCHES)
                progress.set_description(f'Fold score:{eval_frac_loss:.02f}')
                df_eval_effnet_pred = pd.DataFrame(data=np.concatenate([eval_effnet_pred_frac, eval_effnet_pred_vert], axis=1),
                                              columns=[f'C{i}_effnet_frac' for i in range(1, 8)] +
                                                      [f'C{i}_effnet_vert' for i in range(1, 8)])

                df_eval = pd.concat(
                    [df_train.query('split == @fold').head(len(df_eval_effnet_pred)).reset_index(drop=True), df_eval_effnet_pred],
                    axis=1
                ).sort_values(['StudyInstanceUID', 'Slice'])
                df_eval_predictions.append(df_eval)

        df_train_predictions = pd.concat(df_train_predictions)
        df_eval_predictions = pd.concat(df_eval_predictions)

        df_train_predictions.to_csv(f'{EFFNET_CHECKPOINTS_PATH}/{MODEL_NAME}_train_prediction.csv')
        df_eval_predictions.to_csv(f'{EFFNET_CHECKPOINTS_PATH}/{MODEL_NAME}_eval_prediction.csv')
        #
    return df_train_predictions,df_eval_predictions

def patient_prediction(df,frac_cols,vert_cols):
    c1c7 = np.average(df[frac_cols].values, axis=0, weights=df[vert_cols].values)
    pred_patient_overall = 1 - np.prod(1 - c1c7)
    return np.concatenate([[pred_patient_overall], c1c7])

def evaluate(effnet_models,df_train):
    
    df_train_pred,df_eval_pred = gen_effnet_predictions(effnet_models=effnet_models,df_train=df_train)
    target_cols = ['patient_overall'] + [f'C{i}_fracture' for i in range(1, 8)]
    frac_cols = [f'C{i}_effnet_frac' for i in range(1, 8)]
    vert_cols = [f'C{i}_effnet_vert' for i in range(1, 8)]

    df_patient_train_pred = df_train_pred.groupby('StudyInstanceUID').apply(lambda df: patient_prediction(df,vert_cols=vert_cols)).to_frame('pred').join(df_train_pred.groupby('StudyInstanceUID')[target_cols].mean())
    df_patient_eval_pred = df_eval_pred.groupby('StudyInstanceUID').apply(lambda df: patient_prediction(df,frac_cols=frac_cols,vert_cols=vert_cols)).to_frame('pred').join(df_eval_pred.groupby('StudyInstanceUID')[target_cols].mean())

    train_targets = df_patient_train_pred[target_cols].values
    train_predictions = np.stack(df_patient_train_pred.pred.values.tolist())

    eval_targets = df_patient_eval_pred[target_cols].values
    eval_predictions = np.stack(df_patient_eval_pred.pred.values.tolist())

    print('Train_CV score:', custom_metric.weighted_loss(torch.logit(torch.as_tensor(train_predictions)).to(DEVICE), torch.as_tensor(train_targets).to(DEVICE)))
    print('Valid_CV score:', custom_metric.weighted_loss(torch.logit(torch.as_tensor(eval_predictions)).to(DEVICE), torch.as_tensor(eval_targets).to(DEVICE)))

# N-fold models. Can be used to estimate accurate CV score and in ensembled submissions.
def main():
    effnet_models = []
    for fold in range(N_FOLDS):
        if os.path.exists(os.path.join(EFFNET_CHECKPOINTS_PATH, f'{MODEL_NAME}-f{fold}.tph')):
            print(f'Found cached version of effnetv2-f{fold}')
            effnet_models.append(utils.load_model(EffnetModel(), f'{MODEL_NAME}-f{fold}', EFFNET_CHECKPOINTS_PATH))
        else:
            with wandb.init(project=PROJECT_NAME, name=f'EffNet-v2-fold{fold}') as run:
                utils.gc_collect()
                ds_train = effnet_data.EffnetDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())
                ds_eval = effnet_data.EffnetDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())
                effnet_models.append(train_effnet(ds_train, ds_eval, run, f'{EFFNET_CHECKPOINTS_PATH}/{MODEL_NAME}-f{fold}'))

    # "Main" model that uses all folds data. Can be used in single-model submissions.
    if os.path.exists(os.path.join(EFFNET_CHECKPOINTS_PATH, f'{MODEL_NAME}.tph')):
        print(f'Found cached version of effnetv2')
        effnet_models.append(utils.load_model(EffnetModel(), f'{MODEL_NAME}', EFFNET_CHECKPOINTS_PATH))
    else:
        with wandb.init(project=PROJECT_NAME, name=f'EffNet-v2') as run:
            utils.gc_collect()
            ds_train = effnet_data.EffnetDataSet(df_train, TRAIN_IMAGES_PATH, WEIGHTS.transforms())

            train_effnet(ds_train, None, run, f'{EFFNET_CHECKPOINTS_PATH}/{MODEL_NAME}')
    evaluate(effnet_models=effnet_models[:-1],df_train=df_train)
if __name__ == "__main__":
    main()