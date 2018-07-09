

from deepdreamer import model, load_image, recursive_optimize
import numpy as np
import PIL.Image

layer_tensor = model.layer_tensors[6]
file_name = "DD.jpg"
img_result = load_image(filename='{}'.format(file_name))

img_result = recursive_optimize(layer_tensor=layer_tensor, image=img_result,
                 # how clear is the dream vs original image        
                 num_iterations=20, step_size=1.0, rescale_factor=0.5,
                 # How many "passes" over the data. More passes, the more granular the gradients will be.
                 num_repeats=8, blend=0.2)

img_result = np.clip(img_result, 0.0, 255.0)
img_result = img_result.astype(np.uint8)
result = PIL.Image.fromarray(img_result, mode='RGB')
result.save('dream_image_out.jpg')
result.show()





