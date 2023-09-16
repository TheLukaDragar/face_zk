import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight,map_location=torch.device('mps')))
    net.eval()
    feat = net(img).numpy()
    print(feat.shape)
    return feat.flatten()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    f1 = inference(args.weight, args.network, "./in.png")
    f2 = inference(args.weight, args.network, "./i11.png")
    f3 = inference(args.weight, args.network, "./i112.png")

    #compoute cosine similarity
    from scipy.spatial.distance import cosine
    print(cosine(f1,f2))
    print(cosine(f1,f3))
    print(cosine(f2,f3))

    #q is smaller better? 
    #answ: yes because cosine similarity is 1-cosine distance and cosine distance is the angle between two vectors so the smaller the angle the better the similarity



