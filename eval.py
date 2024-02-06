import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import math
import torchvision.transforms as T
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,auc
import matplotlib.pyplot as plt

data_type = "bottle"
#data_type = "carpet"

if data_type == "bottle":
    eval_dir = "./eval_bottle/"

    #学習用正常データの読み出し
    good_list = glob.glob(os.path.join("./mvtec_anomaly_detection/bottle/train/good/" , '*'))

    #評価用正常データの読み出し
    good_test_list = glob.glob(os.path.join("./mvtec_anomaly_detection/bottle/test/good/" , '*'))

    #評価用異常データの読み出し
    bad_test_list = glob.glob(os.path.join("./mvtec_anomaly_detection/bottle/test/broken_large" , '*'))
    bad_test_list = bad_test_list + glob.glob(os.path.join("./mvtec_anomaly_detection./bottle/test/broken_small" , '*'))
    bad_test_list = bad_test_list + glob.glob(os.path.join("./mvtec_anomaly_detection/bottle/test/contamination" , '*'))

else:
    eval_dir = "./eval_carpet/"

    #学習用正常データの読み出し
    good_list = glob.glob(os.path.join("./mvtec_anomaly_detection/carpet/train/good/" , '*'))

    #評価用正常データの読み出し
    good_test_list = glob.glob(os.path.join("./mvtec_anomaly_detection/carpet/test/good/" , '*'))

    #評価用異常データの読み出し
    bad_test_list = glob.glob(os.path.join("./mvtec_anomaly_detection/carpet/test/color" , '*'))
    bad_test_list = bad_test_list + glob.glob(os.path.join("./mvtec_anomaly_detection./carpet/test/cut" , '*'))
    bad_test_list = bad_test_list + glob.glob(os.path.join("./mvtec_anomaly_detection/carpet/test/hole" , '*'))
    bad_test_list = bad_test_list + glob.glob(os.path.join("./mvtec_anomaly_detection/carpet/test/metal_contamination" , '*'))
    bad_test_list = bad_test_list + glob.glob(os.path.join("./mvtec_anomaly_detection/carpet/test/thread" , '*'))




#正常・異常データの数を確認
print(f"good {len(good_list)} good_test {len(good_test_list)} bad {len(bad_test_list)}")

#出力
#good 209 good_test 20 bad 63


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel,self).__init__() 
        # Encoderの構築。
        # nn.Sequential内にはEncoder内で行う一連の処理を記載する。
        # create_convblockは複数回行う畳み込み処理をまとめた関数。
        # 畳み込み→畳み込み→プーリング→畳み込み・・・・のような動作
        self.Encoder = nn.Sequential(self.create_convblock(3,16),     #256
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(16,32),    #128
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(32,64),    #64
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(64,128),   #32
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(128,256),  #16
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(256,512),  #8
                                    )
        # Decoderの構築。
        # nn.Sequential内にはDecoder内で行う一連の処理を記載する。
        # create_convblockは複数回行う畳み込み処理をまとめた関数。
        # deconvblockは逆畳み込みの一連の処理をまとめた関数
        # 逆畳み込み→畳み込み→畳み込み→逆畳み込み→畳み込み・・・・のような動作
        self.Decoder = nn.Sequential(self.create_deconvblock(512,256), #16
                                     self.create_convblock(256,256),
                                     self.create_deconvblock(256,128), #32
                                     self.create_convblock(128,128),
                                     self.create_deconvblock(128,64),  #64
                                     self.create_convblock(64,64),
                                     self.create_deconvblock(64,32),   #128
                                     self.create_convblock(32,32),
                                     self.create_deconvblock(32,16),   #256
                                     self.create_convblock(16,16),
                                    )
        # 出力前の調整用
        self.last_layer = nn.Conv2d(16,3,1,1)
                                        
    # 畳み込みブロックの中身                            
    def create_convblock(self,i_fn,o_fn):
        conv_block = nn.Sequential(nn.Conv2d(i_fn,o_fn,3,1,1),
                                   nn.BatchNorm2d(o_fn),
                                   nn.ReLU(),
                                   nn.Conv2d(o_fn,o_fn,3,1,1),
                                   nn.BatchNorm2d(o_fn),
                                   nn.ReLU()
                                  )
        return conv_block
    # 逆畳み込みブロックの中身
    def create_deconvblock(self,i_fn , o_fn):
        deconv_block = nn.Sequential(nn.ConvTranspose2d(i_fn, o_fn, kernel_size=2, stride=2),
                                      nn.BatchNorm2d(o_fn),
                                      nn.ReLU(),
                                     )
        return deconv_block

    # データの流れを定義     
    def forward(self,x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = self.last_layer(x)           
        return x


###################


device="cpu"
model = CustomModel().to(device)

if data_type == "bottle":
    model_path = 'model_bottle_200.pth'
else:
    model_path = 'model_carpet_200.pth'


model.load_state_dict(torch.load(model_path))

margin_w = 10
prepocess = T.Compose([T.Resize((128,128)),
                                T.ToTensor(),
                                ])
model.eval()
loss_list = []
labels = [0]*len(good_test_list) + [1]*len(bad_test_list)
for idx , path in enumerate(tqdm(good_test_list + bad_test_list)):

    img = Image.open(path)
#    img = prepocess(img).unsqueeze(0).cuda()
    img = prepocess(img).unsqueeze(0).cpu()
    with torch.no_grad():
        output = model(img)[0]
    output = output.cpu().numpy().transpose(1,2,0)
    output = np.uint8(np.maximum(np.minimum(output*255 ,255),0))
    origin = np.uint8(img[0].cpu().numpy().transpose(1,2,0)*255)
    
    
    diff = np.uint8(np.abs(output.astype(np.float32) - origin.astype(np.float32)))
    loss_list.append(np.sum(diff))
    heatmap = cv2.applyColorMap(diff , cv2.COLORMAP_JET)
    margin = np.ones((diff.shape[0],margin_w,3))*255
    
    result = np.concatenate([origin[:,:,::-1],margin,output[:,:,::-1],margin,heatmap],axis = 1)
    label = 'good' if idx < len(good_test_list) else 'bad'
#    cv2.imwrite(f"/{idx}_{label}.jpg",result)
    cv2.imwrite(eval_dir + f"/{idx}_{label}.jpg",result)


fpr, tpr, thresholds = roc_curve(labels,loss_list)

plt.plot(fpr, tpr, marker='o')

plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.savefig('./sklearn_roc_curve.png')
print(roc_auc_score(labels,loss_list))
