import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision as tv
import utils
from src.make_data import effnet_binary_data as effnet_data
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import AUROC, Accuracy
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

import wandb

warnings.simplefilter('ignore')
# SET CONFIG Effnet

cfg = utils.load_yaml(Path("../../config/config_temp.yaml"))

#cfg = utils.load_yaml(Path("/home/jumpei.uchida/develop/kaggle_1080ti_1_2/rsna-2022-cervical-spine-fracture-detection/effnet/config/config_temp.yaml"))
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
df_train_box = pd.read_csv(f"{RSNA_2022_PATH}/cropped_2d_labels.csv")

#PreProcess and Effnetdata
df_train, df_train_slices, df_test, df_test_slices = effnet_data.preprocess( df_train=df_train,
    df_train_slices=df_train_slices,
    df_train_box=df_train_box,
    df_test=df_test,
    TEST_IMAGES_PATH=TEST_IMAGES_PATH,
    N_FOLDS=N_FOLDS,
)
ds_train = effnet_data.EffnetDataSet(df_train, TRAIN_IMAGES_PATH, WEIGHTS.transforms())
ds_test = effnet_data.EffnetDataSet(df_test, TEST_IMAGES_PATH, WEIGHTS.transforms())

class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s(weights=WEIGHTS)
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1280, 1),
        )

    def forward(self, x):
        # returns logits
        x = self.model(x)['flatten']
        return self.nn_fracture(x)

    def predict(self, x):
        frac = self.forward(x)
        return torch.sigmoid(frac)

def evaluate_effnet(model: EffnetModel, ds, max_batches=PREDICT_MAX_BATCHES, shuffle=False):
    torch.manual_seed(SEED)
    model = model.to(DEVICE)
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=os.cpu_count(),
                                          collate_fn=utils.filter_nones)

    pred_frac = []
    valid_list = []
    auroc = AUROC(pos_label = 1)
    with torch.no_grad():
        model.eval()
        frac_losses = []
        with tqdm(dl_test, desc='Eval', miniters=1) as progress:
            for i, (X, y_frac) in enumerate(progress):
                with autocast():
                    y_frac_pred= model.forward(X.to(DEVICE))
                    #Binary Cross Entoropy
                    frac_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_frac_pred.to(DEVICE),y_frac.to(DEVICE),reduction='none')
                    valid_score = auroc(torch.sigmoid(y_frac_pred).to(DEVICE),y_frac.to(DEVICE).to(torch.int64))

                    valid_list.append(valid_score.cpu())
                    pred_frac.append(torch.sigmoid(y_frac_pred))
                    frac_losses.append(torch.mean(frac_loss).cpu())
                if i >= max_batches:
                    break
        return np.mean(frac_losses), torch.concat(pred_frac).cpu().numpy(),np.mean(valid_list)

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
        for batch_idx, (X, y_frac) in enumerate(progress):

            if ds_eval is not None and batch_idx % SAVE_CHECKPOINT_EVERY_STEP == 0 and EFFNET_MAX_EVAL_BATCHES > 0:
                frac_loss,_,valid_score= evaluate_effnet(
                    model, ds_eval, max_batches=EFFNET_MAX_EVAL_BATCHES, shuffle=True)
                model.train()
                logger.log(
                    {'eval_loss': frac_loss,'eval_AUC_ROC_SCORE':valid_score})
                if batch_idx > 0:  # don't save untrained model
                    utils.save_model(name, model)

            if batch_idx >= EFFNET_MAX_TRAIN_BATCHES:
                break
            optim.zero_grad()
            # Using mixed precision training
            with autocast():
                y_frac_pred  = model.forward(X.to(DEVICE))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(y_frac_pred.to(DEVICE),y_frac.to(DEVICE),reduction='none')
                #loss = torch.mean(loss).cpu()

                if np.isinf(np.sum(loss.cpu().detach().numpy()).all()) or np.isnan(loss.cpu().detach().numpy().all()):
                    print(f'Bad loss, skipping the batch {batch_idx}')
                    del loss,  y_frac_pred 
                    utils.gc_collect()

            # scaler is needed to prevent "gradient underflow"
            scaler.scale(torch.mean(loss)).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            progress.set_description(f'Train loss: {torch.mean(loss) :.02f}')
            logger.log({'loss': torch.mean(loss),
                        'lr': scheduler.get_last_lr()[0]})
    utils.save_model(name, model)
    return model


def gen_effnet_predictions(effnet_models, df_train):
    if os.path.exists(os.path.join(EFFNET_CHECKPOINTS_PATH, 'train_{PROJECT_NAME}_{MODEL_NAME}_predictions.csv')):
        print('Found cached version of train_predictions.csv')
        df_eval_effnet_pred = pd.read_csv(os.path.join(EFFNET_CHECKPOINTS_PATH, 'eval_{PROJECT_NAME}_{MODEL_NAME}_predictions.csv'))
    else:
        df_eval_predictions = []
        with tqdm(enumerate(effnet_models), total=len(effnet_models), desc='Folds') as progress:
            for fold, effnet_model in progress:
                ds_eval = effnet_data.EffnetDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())

                #valid_prediction
                eval_frac_loss, eval_effnet_pred_frac,eval_valid_score = evaluate_effnet(effnet_model, ds_eval, PREDICT_MAX_BATCHES)
                progress.set_description(f'Fold loss:{eval_frac_loss:.02f}, Fold score:{eval_valid_score:.02f}')
                df_eval_effnet_pred = pd.DataFrame(data=eval_effnet_pred_frac,
                                              columns=["pred"])

                df_eval = pd.concat(
                    [df_train.query('split == @fold').head(len(df_eval_effnet_pred)).reset_index(drop=True), df_eval_effnet_pred],
                    axis=1
                ).sort_values(['StudyInstanceUID', 'Slice'])
                df_eval_predictions.append(df_eval)

        df_eval_predictions = pd.concat(df_eval_predictions)

        df_eval_predictions.to_csv(f'{EFFNET_CHECKPOINTS_PATH}/{MODEL_NAME}_eval_prediction.csv')
        #df_train_predictions,
    return df_eval_predictions

def patient_prediction(df,frac_cols,vert_cols):
    c1c7 = np.average(df[frac_cols].values, axis=0, weights=df[vert_cols].values)
    pred_patient_overall = 1 - np.prod(1 - c1c7)
    return np.concatenate([[pred_patient_overall], c1c7])

def evaluate(effnet_models,df_train):
    df_eval_pred = gen_effnet_predictions(effnet_models=effnet_models,df_train=df_train)

    df_patient_eval_pred = df_eval_pred.groupby('StudyInstanceUID')[["pred"]].mean()
    fold_data = df_train.groupby(["StudyInstanceUID","split"])[["patient_overall"]].sum().reset_index()
    fold_data["patient_overall"] = np.where(fold_data["patient_overall"].values > 0,1,0)
    df_patient_eval_pred = df_patient_eval_pred.merge(fold_data,on  = "StudyInstanceUID",how = "left")
    valid_list = []
    for fold in range(N_FOLDS):
        df_temp  = df_patient_eval_pred.query("split == @fold")
        eval_targets = df_temp['patient_overall'].values
        eval_predictions = np.stack(df_temp.pred.values.tolist())
        auroc = AUROC(pos_label = 1)
        valid_score = auroc(torch.as_tensor(eval_predictions),torch.as_tensor(eval_targets))
        valid_list.append(valid_score.cpu())
        print(f'Valid_CV score Fold_{fold}:', valid_score)
    
    
    print(f'Valid_CV score :',np.mean(np.array(valid_list)))


def main():
    effnet_models = []
    for fold in range(N_FOLDS):
        if os.path.exists(os.path.join(EFFNET_CHECKPOINTS_PATH, f'{MODEL_NAME}-f{fold}.tph')):
            print(f'Found cached version of effnetv2-f{fold}')
            effnet_models.append(utils.load_model(EffnetModel(), f'{MODEL_NAME}-f{fold}', EFFNET_CHECKPOINTS_PATH))
        else:
            with wandb.init(project=PROJECT_NAME, name=f'{MODEL_NAME}-fold{fold}') as run:
                utils.gc_collect()
                ds_train = effnet_data.EffnetDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())
                ds_eval = effnet_data.EffnetDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())
                effnet_models.append(train_effnet(ds_train, ds_eval, run, f'{EFFNET_CHECKPOINTS_PATH}/{MODEL_NAME}-f{fold}'))

    evaluate(effnet_models=effnet_models,df_train=df_train)
if __name__ == "__main__":
    main()