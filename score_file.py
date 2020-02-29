import torch
import torchvision.transforms as transforms
import sys
import os
import glob
from PIL import Image

torch.set_printoptions(linewidth=120)
class deepLearning():

    def __init__(self):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.val_transform = transforms.Compose([
            transforms.TenCrop((224,224)),
            transforms.Lambda(self.make_stack),
            transforms.Lambda(self.normalize)
        ])

        self.val_data_dir = os.path.join(os.getcwd(), './test_files/*.jpg')

    def make_stack(self,x):
        return torch.stack([transforms.ToTensor()(crop) for crop in x])

    def normalize(self,x):
        return torch.stack([transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])(crop) for crop in x])

    def score_model(self,model=None):

        if model is None:
          network = torch.load("final_model.pth",map_location=torch.device(self.device))
        else:
            network = model
        network.eval()
        classes = ["Shaver","smart-baby-bottle","toothbrush","wake-up-light"]
        for image_name in sorted(glob.glob(self.val_data_dir)):
            im = Image.open(image_name)
            im = self.val_transform(im)
            im = im.to(self.device)
            ncrops, c, h, w = im.size()
            preds = network(im.view(-1, c, h, w))
            preds = preds.view(1, ncrops, -1).mean(1)
            # image_names.append(image_name.split("/")[-1])
            # pred_classes.append(classes[preds.argmax(dim=1).item()])
            print(image_name.split("/")[-1], classes[preds.argmax(dim=1).item()])

if __name__ == '__main__':
    dModel = deepLearning()
    dModel.score_model()