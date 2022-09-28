import numpy as np
import yaml
import cv2
import pydicom as dicom
import torch
import os
import gc


def _get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images. 
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data *= float(dicom.get("RescaleSlope"))
    data += float(dicom.get("RescaleIntercept"))

    center = _get_first_of_dicom_field_as_int(dicom.get("WindowCenter"))
    width = _get_first_of_dicom_field_as_int(dicom.get("WindowWidth"))
    data[:] = np.clip(data, center - width / 2, center + width / 2)
    
    
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img

def load_yaml(path_data):
    with open(path_data) as file:
        obj = yaml.safe_load(file)
    return obj
    
def filter_nones(b):
    return torch.utils.data.default_collate([v for v in b if v is not None])

def save_model(name, model):
    torch.save(model.state_dict(), f'{name}.tph')
    
def load_model(model, name, path='.',DEVICE ='cuda'):
    data = torch.load(os.path.join(path, f'{name}.tph'), map_location=DEVICE)
    model.load_state_dict(data)
    return model

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()