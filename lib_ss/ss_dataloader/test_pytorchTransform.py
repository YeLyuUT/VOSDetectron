import PIL
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from ss_Transformer import Transformer as TF
import torchvision.transforms
if __name__=='__main__':
  img_path = '/home/yelyu/Work/MyDLSolutions/flownet2/scripts/25.bmp'
  img = Image.open(img_path)

  tf = TF(expected_blob_size=(400,400))
  crops = tf(img,batch_size = 12)
  columns=4
  rows=3
  fig=plt.figure(figsize=(columns*3, rows*3))
  for i in range(1, columns*rows +1):
    crop = crops[i-1]
    crop = np.array(crop)
    fig.add_subplot(rows, columns, i)
    plt.imshow(crop)
  plt.show()
  