import cv2
import numpy as np
import zero.image.normalizer

src_input = 'F:/33333/001.png'
dst_input = 'F:/33333/194.png'

src = cv2.imread(src_input)
dst = cv2.imread(dst_input)
normalizer = zero.image.normalizer.VahadaneNormalBGR()
real = cv2.imread('F:/33333/original-vahadane-001-194.png')

src_stain, src_concentration = normalizer.split(src)
dst_stain, dst_concentration = normalizer.split(dst)
merge = normalizer.merge(dst_stain, src_concentration)
print(np.abs(merge.astype('float32') - real).mean())

