import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import warnings
from pathlib import Path
import joblib
import effnetv2_1 as binary_effnetv2
import effnetv2 as effnetv2
from CSFD.data import three_dimensions as th_dim
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
import utils
from tqdm import tqdm
from src.make_data import effnet_data

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
data_path = Path("/home/jumpei.uchida/develop/kaggle_1080ti_1_2/rsna-2022-cervical-spine-fracture-detection/fold0")


def transforms(temp,custom = True):
    assert temp.ndim == 4
    c_list = []
    for c in range(7):
        temp_list = []
        for c1 in temp[c]:
            temp_list.append(np.sum(c1).astype("float64"))
        #temp_list /= max(temp_list)
        c_list.append(temp_list)
    c_array = np.array(c_list)
    if custom:
        for num_i,sum_num in enumerate(np.sum(c_array,axis = 0)):
            if sum_num > 1.0:
                c_array[:,num_i] /= sum_num
    
    return c_array

def resize_origin(uid,i,flag_df):
    vert_cols = [f'C{i}_effnet_vert' for i in range(1, 8)]
    slice = uid_to_slice_map[uid]
    temp = np.load(data_path / f"{uid}.npz",allow_pickle=True)["arr_0"]
    temp = th_dim.resize_depth(temp,depth = slice,depth_range = None,enable_depth_resized_with_cv2=True)
    temp = transforms(temp)
    temp = np.nan_to_num(temp)
    temp = pd.DataFrame(temp.T,columns = vert_cols)
    temp["StudyInstanceUID"] = uid
    if flag_df.query("StudyInstanceUID == @uid")["Flag"].values[0] == 0:
        temp["Slice"] = [i for i in range(1,slice+1)]
    else:
        temp["Slice"] = list(reversed([i for i in range(1,slice+1)]))
    return temp,i

def get_dicom_paths(dicom_dir_path: Path):
    dicom_paths = sorted(
        dicom_dir_path.glob("*"),
        key=lambda p: int(p.name.split(".")[0])
    )
    if (
        dicom.dcmread(dicom_paths[0]).get("ImagePositionPatient")[2]
        >
        dicom.dcmread(dicom_paths[-1]).get("ImagePositionPatient")[2]
    ):
        return dicom_paths[::-1]
    return dicom_paths

def make_path_list(path,i):
    flag = 1
    temp = get_dicom_paths(path)
    if temp[0].parts[-1] == "1.dcm":
        flag =0
    uid = temp[0].parts[-2]
    return [uid,flag],i

df_eval_pred = pd.read_csv("/home/jumpei.uchida/develop/kaggle_1080ti_1_2/rsna-2022-cervical-spine-fracture-detection/effnet/src/saved_model/effnet/temp_eval_prediction.csv")
uid_to_slice_map = df_eval_pred.groupby("StudyInstanceUID")["Slice"].max().to_dict()
vert_cols = [f'C{i}_effnet_vert' for i in range(1, 8)]
paths = Path("/home/jumpei.uchida/develop/data/rsna/train_images")
path_list = joblib.Parallel(n_jobs=-1)([
    joblib.delayed(make_path_list)(path,i)
    for i,path in tqdm(enumerate(list(paths.iterdir())))])
path_list.sort(key=lambda x: x[1])
path_list = [t[0] for t in path_list]
flag_df = pd.DataFrame(path_list,columns = ["StudyInstanceUID","is_reversed"])
images = joblib.Parallel(n_jobs=-1)([
    joblib.delayed(resize_origin)(uid,i,flag_df)
    for i,uid in tqdm(enumerate(list(uid_to_slice_map.keys())))])



images.sort(key=lambda x: x[1])
images = [t[0] for t in images]
df_pred = pd.concat(images)
df_pred[vert_cols] += 0.000000000000000001
df_eval_pred = df_eval_pred.drop(vert_cols,axis = 1)
df_eval_pred = df_eval_pred.merge(df_pred,on = ["StudyInstanceUID","Slice"],how = "left")
if __name__ == "__main__":
    print("hello")