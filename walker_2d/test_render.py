from walker import Walker
from PIL import Image

env = Walker()

env.reset()


pil_image=Image.fromarray(env.render(), mode="RGB")
flip_img = pil_image.transpose(Image.TRANSPOSE)
flip_img.show()