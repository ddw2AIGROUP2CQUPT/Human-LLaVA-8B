# HumanVLM
## DEMO



https://github.com/user-attachments/assets/9ce44481-3fa0-4ea7-8f60-1b639f8fa80a


### Introduction
![image/png](https://cdn-uploads.huggingface.co/production/uploads/64259db7d3e6fdf87e4792d0/ur3sls4faPNlOMZ6sA_qK.png)

Human-related vision and language tasks are widely applied across various social scenarios.  The latest studies demonstrate that the large vision-language model can enhance the performance of various downstream tasks in visual-language understanding.  Since, models in the general domain often not perform well in the specialized field.  In this study, we train a domain-specific Large Language-Vision model, Human-LLaVA, which aim to construct an unified multimodal Language-Vision Model for Human-related tasks.

Specifically, (1) we first construct **a large-scale and high-quality human-related image-text (caption) dataset** extracted from Internet for domain-specific alignment in the first stage (Coming soon);  (2) we also propose to construct **a multi-granularity caption for human-related images** (Coming soon), including human face, human body, and whole image, thereby fine-tuning a large language model.  Lastly, we evaluate our model on a series of downstream tasks, our **Human-LLaVA** achieved the best overall performance among multimodal models of similar scale.  In particular, it exhibits the best performance in a series of human-related tasks, significantly surpassing similar models and ChatGPT-4o.  We believe that the Huaman-LLaVA model and a series of datasets presented in this work can promote research in related fields.


## Result
human-llava has a good performance in both general and special fields

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64259db7d3e6fdf87e4792d0/X-712oVUBPXbfLcAz83fb.png)

## News and Update 🔥🔥🔥
* Oct.23, 2024.  **🤗[HumanCaption-HQ-311K](https://huggingface.co/datasets/OpenFace-CQUPT/HumanCaption-HQ-311K), is released!👏👏👏**
* Sep.12, 2024.  **🤗[HumanCaption-10M](https://huggingface.co/datasets/OpenFace-CQUPT/HumanCaption-10M), is released!👏👏👏**
* Sep.8, 2024.   **🤗[HumanLLaVA-llama-3-8B](https://huggingface.co/OpenFace-CQUPT/Human_LLaVA), is released!👏👏👏**


## 🤗 Transformers
To use Human-LLaVA for the inference, all you need to do is to input a few lines of codes as demonstrated below. However, please make sure that you are using latest code.
``` python
import requests
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForPreTraining


model_id = "OpenFace-CQUPT/Human_LLaVA"
cuda = 0
model = AutoModelForPreTraining.from_pretrained("OpenFace-CQUPT/Human_LLaVA", torch_dtype=torch.float16).to(cuda)

processor = AutoProcessor.from_pretrained(model_id,trust_remote_code=True)


text = "Please describe this picture"
prompt = "USER: <image>\n" + text + "\nASSISTANT:"
image_file = "./test1.jpg"
raw_image = Image.open(image_file)
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(cuda, torch.float16)

output = model.generate(**inputs, max_new_tokens=400, do_sample=False)
predict = processor.decode(output[0][:], skip_special_tokens=True)
print(predict)
```
## Get the Dataset
#### Dataset Example
![image/png](https://cdn-uploads.huggingface.co/production/uploads/64259db7d3e6fdf87e4792d0/vRojQxm8IMybBV0X5CKbf.png)
#### Domain Alignment Stage
[HumanCaption-10M](https://huggingface.co/datasets/OpenFace-CQUPT/HumanCaption-10M)(self construct): is released!

#### Instruction Tuning Stage
**All public data sets have been filtered, and we will consider publishing all processed text in the future**

[HumanCaption-HQ](https://huggingface.co/datasets/OpenFace-CQUPT/HumanCaption-HQ-311K)(self construct): is released!

[FaceCaptionA](https://huggingface.co/datasets/OpenFace-CQUPT/FaceCaption-15M)(self construct): is released!

CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

ShareGPT4V:https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md

LLaVA-Instruct_zh : https://huggingface.co/datasets/openbmb/llava_zh

verified_ref3rec: https://huggingface.co/datasets/lucasjin/refcoco/blob/main/ref3rec.json

verified_ref3reg: https://huggingface.co/datasets/lucasjin/refcoco/blob/main/ref3rec.json

verified_shikra: https://github.com/shikras/shikra




## Citation

```
Coming soon!!!
```

## contact

mailto: [S230201133@stu.cqupt.edu.cn](mailto:S230201133@stu.cqupt.edu.cn) or [dw_dai@163.com](mailto:dw_dai@163.com)
