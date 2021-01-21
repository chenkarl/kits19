import torch.utils.data as data
import numpy as np
import torch
from starter_code.utils import load_case,load_volume

DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]

#  原图归一化
def normalize(volume,t_max,t_min):
    if t_max is not None or t_min is not None:
        t_volume = np.clip(volume, t_min, t_max)
    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval) / max(mxval - mnval, 1e-3)
    # enlarge the image ratio
    # im_volume = 255 * im_volume
    return im_volume

#  取出一个肾脏分割图，二值化的分割图
def get_kid_img(segmentation):
    shp = segmentation.shape
    kid_img = np.zeros((shp[0], shp[1], shp[2]), dtype=np.float32)
    kid_img[np.equal(segmentation, 1)] = 1
    return kid_img

# data.Dataset:
# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)


class KidDataset(data.Dataset):
    # 创建KidDataset类的实例时，就是在调用init初始化
    def __init__(self, case_num, transform=None, target_transform=None):  # root表示图片路径
        volume=[]
        segmentation=[]
        if case_num == -1:  # 如果等于-1，将所有数据集都读入
            volume =[]
            segmentation=[]
            for i in range (210):
                tmp_volume, tmp_segmentation = load_case(i)
                np.vstack((volume,tmp_volume))
                np.vstack((segmentation,tmp_segmentation))
        elif case_num < 210:
            volume, segmentation = load_case(case_num)
            kid_seg_ims = get_kid_img(segmentation.get_data())  # 取出肾脏分割图
        else :
            volume = load_volume(case_num)

        vol_ims = normalize(volume.get_data(), DEFAULT_HU_MAX, DEFAULT_HU_MIN)  # 将原图归一化转为灰度图
        imgs = []
        if case_num == -1:
            for num in range(vol_ims.shape[0]):
                imgs.append([vol_ims[num], kid_seg_ims[num]])
        elif case_num < 210:
            for num in range(vol_ims.shape[0]):
                if not np.all(kid_seg_ims[num]==0):
                    imgs.append([vol_ims[num], kid_seg_ims[num]])
        else:
            for num in range(vol_ims.shape[0]):
                imgs.append([vol_ims[num],None])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # x_path, y_path = self.imgs[index]
        # img_x = Image.open(x_path)
        # img_y = Image.open(y_path)
        x,y = self.imgs[index]
        from imageio import imwrite
        from skimage import img_as_ubyte
        if self.transform is not None:
            # 将原图画出来
            # opath_seg = ("./orign/{:05d}.png".format(index))
            # imwrite(str(opath_seg), x)
            img_x = self.transform(x)
            img_x = img_x.type(torch.FloatTensor)
        if self.target_transform is not None:
            # 将标签图画出来
            # lpath_seg = ("./label/{:05d}.png".format(index))
            # imwrite(str(lpath_seg), img_as_ubyte(y))
            img_y = self.target_transform(y)
        else:
            img_y = []
        return img_x, img_y  # 返回的是图片

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]
