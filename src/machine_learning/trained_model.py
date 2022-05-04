import torch
import glob
from PIL import Image
import torchvision.transforms as transforms
from DigitRecognitionTraining import Network


def classify(weights_path, img):
    loaded_model = Network()
    weights = torch.load(weights_path)
    loaded_model.load_state_dict(weights)
    image_trans = transforms.Compose([
                    transforms.Resize((32, 32), Image.ANTIALIAS),
                    transforms.ToTensor()
                 ])
    loaded_model.eval()

    image = image_trans(img).float()
    ready_img = image.unsqueeze(0)
    output = loaded_model(ready_img)
    _, predicted = torch.max(output, 1)
    
    return predicted


def predict_num(img, image_path=None):
    if image_path is not None:
        img = Image.open(image_path)
    path = glob.glob("*.pth")
    return classify(str(path[0]), img)
