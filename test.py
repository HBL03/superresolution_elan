import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import utils
import argparse
import yaml

# Argument parser
parser = argparse.ArgumentParser(description='ELAN')
parser.add_argument('--config', type=str, default=None, help='pre-config file for testing')
parser.add_argument('--model_path', type=str, required=True, help='path to the saved model')
parser.add_argument('--input_image_path', type=str, required=True, help='path to the input image')
parser.add_argument('--output_image_path', type=str, default='output_image.png', help='path to save the output image')
args = parser.parse_args()

# Load configuration
if args.config:
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in config.items():
        setattr(args, key, value)

# Load model
model = utils.import_module('models.{}_network'.format(args.model)).create_model(args)
model = nn.DataParallel(model).to('cuda')
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess input image
input_image = Image.open(args.input_image_path).convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor()
])
input_tensor = transform(input_image).unsqueeze(0).to('cuda')

# Perform inference
with torch.no_grad():
    output_tensor = model(input_tensor)

# Post-process and save the output image
output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to numpy array
output_image = (output_image * 255).clip(0, 255).astype('uint8')  # Scale to [0, 255]

# Save the output image
output_image_pil = Image.fromarray(output_image)
output_image_pil.save(args.output_image_path)

