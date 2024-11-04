import requests
from PIL import Image, ImageDraw
from typing import List
import torch
import re
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.models.llava.configuration_llava import LlavaConfig
from transformers import SiglipVisionConfig, SiglipImageProcessor, GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


def extract_coords(s):
    pattern = r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
    match = re.search(pattern, s)
    if match:
        xmin, ymin, xmax, ymax = map(float, match.groups()) 
        return [xmin, ymin, xmax, ymax]
    else:
        return None

def draw_image_with_boxes(image, box):
    # 创建一个ImageDraw对象
    draw = ImageDraw.Draw(image)

    # 定义坐标和矩形边框颜色以及宽度
    xmin, ymin, xmax, ymax = box  # 示例坐标，假设这些是归一化坐标，需要转换为像素坐标
    border_color = (255, 0, 0)  # 红色
    border_width = 3  # 边框宽度

    # 注意：这里的坐标需要从归一化坐标转换为实际像素坐标，假设图片宽度为width，高度为height
    width, height = image.size
    xmin_pixel = int(xmin * width)
    ymin_pixel = int(ymin * height)
    xmax_pixel = int(xmax * width)
    ymax_pixel = int(ymax * height)

    # 在图片上画矩形框
    draw.rectangle([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], outline=border_color, width=border_width)

    # 保存带有标记的图片
    output_path = 'image_with_box.jpg'  # 输出图片的路径
    image.save(output_path)


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, token_id_list: List[int] = None):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # return np.argmax(scores[-1].detach().cpu().numpy()) in self.token_id_list
        # 储存scores会额外占用资源，所以直接用input_ids进行判断
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list

stopping_criteria = StoppingCriteriaList()
stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[128009]))


model_id = "/home/ubuntu/san/LYT/UniDetRet-exp/HumanLlama3/work_dirs/human_llama3_8b_instruct_siglip_so400m_large_p14_384_lora_e1_gpu8_finetune/iter_75000_hf"
prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\ngive the coords of the man<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")
image_file = "/home/ubuntu/san/LYT/UniDetRet-exp/HumanLlama3/1.jpg"

config = LlavaConfig.from_pretrained(model_id)
gen_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=False,
    eos_token_id=128001,
    pad_token_id=128001
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    config=config
).to(0)

processor = AutoProcessor.from_pretrained(model_id)
processor.image_processor = SiglipImageProcessor.from_pretrained('google/siglip-so400m-patch14-384')


raw_image = Image.open(image_file)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, generation_config=gen_config, bos_token_id=128000, stopping_criteria=stopping_criteria)
response = processor.decode(output[0][2:], skip_special_tokens=True)
coord = extract_coords(response)

# if coord is not None
if coord is not None:
    coord = [0.378, 0.489, 0.562, 0.997]
    draw_image_with_boxes(raw_image, box=coord)
print(response)