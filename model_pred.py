import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet
import torch.nn.functional as F
import cv2

from torchvision import transforms as T
import torchvision
from PIL import Image
def get_device():
    return "cpu" if torch.cuda.is_available() else "cpu"
class ImgToTensor(object):
    def __call__(self, img):
        tf = T.Compose([T.ToTensor(),                                                                                    
])
        return tf(img)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).long()
class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x
class LinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=1):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()

        base = resnet.resnet18( weights=None)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        # self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
#         print("e3.shape: ",e3.shape)
        
        d4 = e3 + self.decoder4(e4)
#         print("d4.shape",d4.shape)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        # y = self.lsm(y)

        return y
def get_img_patches(img):

    img_height, img_width, _ = img.shape

    input_height = input_width = 256

    stride_ratio = 0.5
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []

    if img_height < img_width:
        assert img_height < 2*input_height
        y_corner = [0, img_height-input_height, int(0.5*(img_height-input_height))]
        for y in y_corner:
            for x in range(0, img_width - input_width + 1, stride):
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
            if x != img_width - input_width:
                x = img_width - input_width
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
    else:
        assert img_width < 2*input_width
        x_corner = [0, img_width-input_width, int(0.5*(img_width-input_width))]
        for x in x_corner:
            for y in range(0, img_height - input_height + 1, stride):
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
            if y != img_height - input_height:
                y = img_height - input_height
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
    
    assert np.all(normalization_map >= 1)

    patches.append(cv2.resize(img, (input_height, input_width), interpolation=cv2.INTER_CUBIC))

    patches = np.array(patches)
    
    return patches, patch_locs

def merge_pred_patches(img, preds, patch_locs):

    img_height, img_width, _ = img.shape

    input_height = input_width = 256

    probability_map = np.zeros((img_height, img_width), dtype=float)
    num1 = np.zeros((img_height, img_width), dtype=np.int16)
    
    for i, response in enumerate(preds):
        if i < len(preds)-1:
            coords = patch_locs[i]
            probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response
            num1[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += 1
        else:
            mskp = cv2.resize(response, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    assert np.all(num1 != 0)
    probability_map = probability_map / num1

    msk_pred = 0.5*probability_map + 0.5 * mskp

    return msk_pred
def patch_pred_total(model,device,img1):   
        patches,patch_locs=get_img_patches(img1)
        patch_totensor=ImgToTensor()
        preds=[]
        for i in patches:
            i=patch_totensor(Image.fromarray(i))
            i=(i.unsqueeze(0)).to(device, dtype=torch.float32)
        #     msk_pred=model(i)
            msk_pred = torch.sigmoid(model(i)) 
            msk_pred=(msk_pred>0.6).float()# torch.Size([1, 1, 256, 256])
            mask = msk_pred.cpu().detach().numpy()[0, 0] 
        #     plt.imshow(mask,cmap='gray')
        #     plt.show()
            preds.append(mask)
        
        mskp= merge_pred_patches(img1, preds, patch_locs)
        kernel = np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ], dtype=np.uint8)
        # plt.imshow(mskp,cmap='gray')
        # plt.show()
        mskp = cv2.morphologyEx(mskp, cv2.MORPH_CLOSE, kernel,iterations=1).astype(float)
        return mskp
def crack_pred(path):
    img1=cv2.imread(path)
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    #model_path='model_scripted.pt'
    model_path='best_model.pth'
    device=get_device()
    #print(device)
    model = LinkNet()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    #model=torch.jit.load(model_path,map_location=device)
    model.eval()
    out=patch_pred_total(model,device,img1)
    img_copy=img1.copy()
    img_copy[out>0] = [0,0,255]
    cv2.imwrite("output.jpg",img_copy)
    return 1
if __name__ == "__main__":
    start=time.time()
    image_path='CFD_084.jpg'
    crack_pred(image_path)
    end_time=time.time()
    elapsed_time=end_time -start
    #print("Elapsed Time for the model Running is : ",elapsed_time)
    #print(os.getcwd())
    
