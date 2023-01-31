import cv2
import zero.image.normalizer

src_input = 'F:/33333/001.png'
dst_input = 'F:/33333/194.png'

filepath = 'F:/33333/zero-macenko-%s-001-194.png'

src_bgr = cv2.imread(src_input)
dst_bgr = cv2.imread(dst_input)
src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
dst_rgb = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)
bgr_norm = zero.image.normalizer.MacenkoNormalBGR(dst_bgr)
rgb_norm = zero.image.normalizer.MacenkoNormalRGB(dst_rgb)

cv2.imwrite(filepath % "bgr", bgr_norm(src_bgr).round().clip(0, 255).astype('uint8'))
cv2.imwrite(filepath % "rgb", rgb_norm(src_rgb).round().clip(0, 255).astype('uint8')[..., [2, 1, 0]])
