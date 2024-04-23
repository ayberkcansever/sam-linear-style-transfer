import os
import sys

import torch
import utils

from libs.Matrix import MulLayer
from libs.models import encoder4, decoder4

#content_fn = "data/content/chicago.png"
content_fn = sys.argv[1]

content_im = utils.loadImage(content_fn)

style_fn = sys.argv[2]
style_im = utils.loadImage(style_fn)

#save_fn = "output.png"
save_fn = sys.argv[3]

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc_ref = encoder4()
dec_ref = decoder4()
matrix_ref = MulLayer('r41')

dir_path = os.path.dirname(os.path.realpath(__file__))
enc_ref.load_state_dict(torch.load(dir_path + '/models/vgg_r41.pth'))
dec_ref.load_state_dict(torch.load(dir_path + '/models/dec_r41.pth'))
matrix_ref.load_state_dict(torch.load(dir_path + '/models/r41.pth',map_location=torch.device('cpu')))


#content_im = content_im.to(device)
#style_im = style_im.to(device)
#enc_ref.to(device)
#dec_ref.to(device)
#matrix_ref.to(device)


with torch.no_grad():
    # Reference comparison
    cF_ref = enc_ref(content_im)
    sF_ref = enc_ref(style_im)
    feature_ref,transmatrix_ref = matrix_ref(cF_ref['r41'],sF_ref['r41'])
    result = dec_ref(feature_ref)
    

utils.saveImage(save_fn,result)

