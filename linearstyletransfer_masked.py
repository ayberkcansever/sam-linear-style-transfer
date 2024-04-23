
import torch
import utils

from libs.Matrix_masked import MulLayer
from libs.models_masked import encoder4, decoder4

content_fn = "data/content/FlyingBird.jpg"
content_im = utils.loadImage(content_fn)

masked_fn = "data/content/FlyingBird_mask.jpg"
masked_im = utils.loadMask(masked_fn,dtype=torch.float32)
#masked_im = None

style_fn = "data/style/antimonocromatismo.jpg"
style_im = utils.loadImage(style_fn)

save_fn = "output.png"

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc_ref = encoder4()
dec_ref = decoder4()
matrix_ref = MulLayer('r41')

enc_ref.load_state_dict(torch.load('models/vgg_r41.pth'))
dec_ref.load_state_dict(torch.load('models/dec_r41.pth'))
matrix_ref.load_state_dict(torch.load('models/r41.pth',map_location=torch.device('cpu')))


#content_im = content_im.to(device)
#style_im = style_im.to(device)
#enc_ref.to(device)
#dec_ref.to(device)
#matrix_ref.to(device)


with torch.no_grad():
    # Reference comparison
    cF_ref,small_mask = enc_ref(content_im,masked_im)
    sF_ref,_ = enc_ref(style_im)
    feature_ref,transmatrix_ref = matrix_ref(cF_ref['r41'],sF_ref['r41'],small_mask)
    result = dec_ref(feature_ref,masked_im)

#import matplotlib.pyplot as plt
#plt.imshow(result.detach().numpy()[0,:3].transpose(1,2,0));plt.show()

utils.saveImage(save_fn,result)
