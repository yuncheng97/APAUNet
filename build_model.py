from model.APAUNet import APAUNet
from model.CoTUNet import CoTUNet
from model.UNet3Plus import UNet3Plus
from model.HFA import HFA
from model.UNETR import UNETR
from model.UNet import UNet
from model.UTNet import UTNet
from model.VTUNet import VTUNet
from model.CoTr.CoTr import CoTr
from model.TransBTS.TransBTS import TransBTS

def build_model(model_name, in_ch, out_ch):
    model_dict = {
        'APAUNet': APAUNet,
        'UTNet': UTNet,
        'CoTUNet': CoTUNet,
        'VTUNet': VTUNet,
        'HFA': HFA,
        'UNETR': UNETR,
        'UNet': UNet,
        'UNet3Plus': UNet3Plus,
        'CoTr': CoTr,
        'TransBTS': TransBTS
    }

    if model_name in model_dict:
        print(f'Loading model {model_name}!')
        if model_name == 'TransBTS':
            _, model = model_dict[model_name](dataset='liver', _conv_repr=True, _pe_type="learned")
            return model
        elif model_name == 'UNETR':
            return model_dict[model_name](in_ch, out_ch, (96, 96, 96))
        elif model_name == 'VTUNet' or model_name == 'HFA' or model_name == 'CoTr':
            return model_dict[model_name](out_ch)
        else:
            return model_dict[model_name](in_ch, out_ch)
    else:
        raise RuntimeError('Given model name not implemented!')

