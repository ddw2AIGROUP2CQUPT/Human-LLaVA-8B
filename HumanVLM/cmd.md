# pretrain human-llama3
## 1. modified config
xtuner copy-cfg llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain .
mv llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain_copy.py human_llama3_8b_instruct_siglip_so400m_large_p14_384_e1_gpu8_pretrain.py

## 2. change vision model
CLIPImageProcessor -> SiglipImageProcessor
CLIPVisionModel -> SiglipVisionModel
visual_encoder_name_or_path = 'google/siglip-so400m-patch14-384'


## 3. offline pt data
python ./xtuner/xtuner/tools/process_untokenized_llava_data.py ./HumanLlama3/human_llama3_8b_instruct_siglip_so400m_large_p14_384_e1_gpu8_pretrain.py --save-folder ./HumanCaption/pt_hfformat
参考：https://xtuner.readthedocs.io/zh-cn/latest/acceleration/train_large_scale_dataset.html#llava

# multi-node pretrain
HF_ENDPOINT=https://hf-mirror.com NPROC_PER_NODE=8 NNODES=2 PORT=11404 ADDR=192.168.24.4 NODE_RANK=0 xtuner train human_llama3_8b_instruct_siglip_so400m_large_p14_384_e1_gpu8_pretrain.py --deepspeed deepspeed_zero2 --seed 1024
HF_ENDPOINT=https://hf-mirror.com NPROC_PER_NODE=8 NNODES=2 PORT=11404 ADDR=192.168.24.4 NODE_RANK=1 xtuner train human_llama3_8b_instruct_siglip_so400m_large_p14_384_e1_gpu8_pretrain.py --deepspeed deepspeed_zero2 --seed 1024

deepspeed --hostfile hostfile --master_port=12345 ../xtuner/xtuner/tools/train.py HumanLlama3/human_llama3_8b_instruct_siglip_so400m_large_p14_384_e1_gpu8_pretrain.py --launcher pytorch  --deepspeed deepspeed_zero2 --seed 1024

## resume
set true to resume in  human_llama3_8b_instruct_siglip_so400m_large_p14_384_e1_gpu8_pretrain.py 
HF_ENDPOINT=https://hf-mirror.com NPROC_PER_NODE=8 NNODES=2 PORT=11404 ADDR=192.168.24.4 NODE_RANK=0 xtuner train human_llama3_8b_instruct_siglip_so400m_large_p14_384_e1_gpu8_pretrain.py --deepspeed deepspeed_zero2 --seed 1024 --resume ./work_dirs/human_llama3_8b_instruct_siglip_vit_large_p14_336_e1_gpu8_pretrain/iter_54000.pth

HF_ENDPOINT=https://hf-mirror.com NPROC_PER_NODE=8 NNODES=2 PORT=11404 ADDR=192.168.24.4 NODE_RANK=1 xtuner train human_llama3_8b_instruct_siglip_so400m_large_p14_384_e1_gpu8_pretrain.py --deepspeed deepspeed_zero2 --seed 1024 --resume ./work_dirs/human_llama3_8b_instruct_siglip_vit_large_p14_336_e1_gpu8_pretrain/iter_54000.pth

## merge pretrain model
HF_ENDPOINT=https://hf-mirror.com xtuner convert pth_to_hf ../../human_llama3_8b_instruct_siglip_so400m_large_p14_384_lora_e1_gpu8_finetune.py ./iter_45000.pth ./iter_45000_pretrain

## pretrain chat
python chat.py meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder google/siglip-so400m-patch14-384 --llava ./iter_54000_hf_merge --prompt-template llama3_chat --image test.jpg

# multi-node sft
两种方式都可以
deepspeed --hostfile hostfile --master_port=12345 ../xtuner/xtuner/tools/train.py human_llama3_8b_instruct_siglip_so400m_large_p14_384_lora_e1_gpu8_finetune.py --launcher pytorch  --deepspeed deepspeed_zero2 --seed 1024

HF_ENDPOINT=https://hf-mirror.com NPROC_PER_NODE=8 NNODES=2 PORT=12345 ADDR=192.168.24.5 NODE_RANK=0 xtuner train human_llama3_8b_instruct_siglip_so400m_large_p14_384_lora_e1_gpu8_finetune.py --deepspeed deepspeed_zero2 --seed 1024
HF_ENDPOINT=https://hf-mirror.com NPROC_PER_NODE=8 NNODES=2 PORT=12345 ADDR=192.168.24.5 NODE_RANK=1 xtuner train human_llama3_8b_instruct_siglip_so400m_large_p14_384_lora_e1_gpu8_finetune.py --deepspeed deepspeed_zero2 --seed 1024

## offline ft data
python /home/ubuntu/san/LYT/UniDetRet-exp/xtuner/xtuner/tools/process_untokenized_llava_data.py human_llama3_8b_instruct_siglip_so400m_large_p14_384_lora_e1_gpu8_finetune.py --save-folder /home/ubuntu/public-Datasets/HumanSFT/ft_hfformat
参考：https://xtuner.readthedocs.io/zh-cn/latest/acceleration/train_large_scale_dataset.html#llava

## convert merge ft model
1. 转换lora权重成hf格式
xtuner convert pth_to_hf ../../human_llama3_8b_instruct_siglip_so400m_large_p14_384_lora_e1_gpu8_finetune.py ./iter_45000.pth ./iter_45000_ft 

2. 合并llm lora
xtuner convert merge meta-llama/Meta-Llama-3-8B-Instruct ./iter_45000_ft/llm_adapter ./iter_45000_ft/llm_merge_lora

3. 合并vit lora
xtuner convert merge google/siglip-so400m-patch14-384 ./iter_45000_ft/visual_encoder_adapter ./iter_45000_ft/vit_merge_lora --is-siglip

4. convert to hf format
cp ../../convert_xtuner_weights_to_hf.py ./
python ./convert_xtuner_weights_to_hf.py --text_model_id ./iter_45000_ft/llm_merge_lora --vision_model_id ./iter_45000_ft/vit_merge_lora --projector_weight ./iter_45000_ft/projector/model.safetensors --save_path ./iter_45000_ft 


5. convert to llava format

cp ../../convert_xtuner_weights_to_llava.py ./
python ./convert_xtuner_weights_to_llava.py --text_model_id ./iter_45000_ft/llm_merge_lora --vision_model_id ./iter_45000_ft/vit_merge_lora --projector_weight ./iter_45000_ft/projector/model.safetensors --save_path ./iter_45000_llava


## eval
**MMB**
```
xtuner mmbench meta-llama/Meta-Llama-3-8B-Instruct \
--visual-encoder google/siglip-so400m-patch14-384 \
--llava ./iter_45000_ft \
--prompt-template llama3_chat \
--data-path ./eval/data/MMBench_TEST_EN.tsv \
--work-dir ./eval/logs/mmb_t_en \
--is-siglip
```
**POPE**

NPROC_PER_NODE=8 xtuner eval_pope meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder google/siglip-so400m-patch14-384 --llava /home/ubuntu/san/LYT/UniDetRet-exp/HumanLlama3/work_dirs/human_llama3_8b_siglip_base_attr_keypoint_new_0616/iter_45000_ft --prompt-template llama3_chat --data-path /home/ubuntu/san/LYT/UniDetRet-exp/Bunny/eval/pope/coco_pope_random.jsonl --work-dir /home/ubuntu/san/LYT/UniDetRet-exp/HumanLlama3/eval/logs/pope --launcher pytorch

### REC

**HumanVLM**
```
NPROC_PER_NODE=8 xtuner eval_refcoco meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder google/siglip-so400m-patch14-384 --llava ./work_dirs/human_llama3_8b_siglip_base_attr_keypoint_new_0616/iter_45000_ft  --prompt-template llama3_chat --data-path ../COCO/annotations/refcoco_testA.jsonl --work-dir ./eval/logs/refcoco/HumanVLM --launcher pytorch
```
***llava-1.5-7b***
```
NPROC_PER_NODE=8 xtuner llava_grounding  llava-hf/llava-1.5-7b-hf  --data-path ../COCO/annotations/refcoco_testA.jsonl  --work-dir ./eval/logs/refcoco/llava7b --launcher pytorch
```
***llava-1.5-13b***
```
NPROC_PER_NODE=8 xtuner llava_grounding llava-hf/llava-1.5-13b-hf --data-path ../COCO/annotations/refcoco_testA.jsonl  --work-dir ./eval/logs/refcoco/llava13b --launcher pytorch
```
***llava-llama-3-8b***
```
NPROC_PER_NODE=8 xtuner llava_llama_grounding  xtuner/llava-llama-3-8b-v1_1-transformers  --data-path ../COCO/annotations/refcoco_testA.jsonl  --work-dir ./eval/logs/refcoco/llava_llama3_8b --launcher pytorch
```
***Qwen/Qwen2-VL-7B-Instruct***
```
NPROC_PER_NODE=1 xtuner qwen2_eval  Qwen/Qwen2-VL-7B-Instruct  --data-path ../COCO/annotations/refcoco_testA.jsonl  --work-dir ./eval/logs/refcoco/qwen2 --launcher pytorch
```

### Attribute

**HumanLlama3**
```
NPROC_PER_NODE=8 xtuner attributes_eval meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder google/siglip-so400m-patch14-384 --llava ./work_dirs/human_llama3_8b_siglip_base_attr_keypoint_new_0616/iter_45000_ft --prompt-template llama3_chat --data-path ./humanhq_attribute_generate_val_en_5k.json --work-dir ./eval/logs/attributes --launcher pytorch

NPROC_PER_NODE=8 xtuner attributes_eval meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder google/siglip-so400m-patch14-384 --llava ./work_dirs/human_llama3_8b_siglip_base_attr_keypoint_new_0616/iter_45000_ft --prompt-template llama3_chat --data-path ./CelebA/celeba_attribute_generate_test_en.json --work-dir ./eval/logs/attributes --launcher pytorch
```

## chat
cp ../../chat.py ./
python chat.py meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder google/siglip-so400m-patch14-384 --llava ./iter_45000_ft --prompt-template llama3_chat --image 1.jpg

python chat_humanllava.py meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder google/siglip-so400m-patch14-384 --llava work_dirs/human_llama3_pretrain/iter_95907_ft --prompt-template llama3_chat --image 1.jpg --anyres-image
## convert to gguf

### 1. 配置llama.cpp环境
```shell
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
```

### 2. 转换成gguf
```shell
cd ./work_dirs/human_llama3_8b_instruct_siglip_so400m_large_p14_384_lora_e1_gpu8_finetune
1. Use llava-surgery-v2.py to split the LLaVA model to LLaMA and multimodel projector constituents:
python ./llama.cpp/examples/llava/llava-surgery-v2.py -C -m ./iter_45000_llava

4. Then convert the llm model to gguf format:
python ./llama.cpp/convert-hf-to-gguf.py ./iter_45000_ft/llm_merge_lora --outfile ./vit/human-llama3.gguf
or
python ./llama.cpp/convert.py ./iter_75000_llava --vocab-type bpe --skip-unknown --outfile ./vit/human-llama3.gguf

3.Copy the llava.clip file into a subdirectory (like vit), rename it to pytorch_model.bin and add a fitting vit configuration to the directory:
cp ./iter_45000_llava/llava.clip vit/pytorch_model.bin
cp ./iter_45000_llava/llava.projector vit/

nano vit/config.json
`
{
  "architectures": [
    "SiglipModel"
  ],
  "initializer_factor": 1.0,
  "model_type": "siglip",
  "text_config": {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "model_type": "siglip_text_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27
  },
  "projection_dim": 768,
  "layer_norm_eps": 1e-05,
  "torch_dtype": "float32",
  "transformers_version": "4.37.0.dev0",
  "vision_config": {
    "hidden_act": "gelu_pytorch_tanh",
    "projection_dim": 768,
    "layer_norm_eps": 1e-06,
    "hidden_size": 1152,
    "image_size": 384,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "patch_size": 14
  }
}
`

python ./llama.cpp/examples/llava/convert-image-encoder-to-gguf.py -m vit --llava-projector vit/llava.projector --output-dir vit --clip-model-is-openclip --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5 --use-f32


5. quantize llm model
/home/ubuntu/san/LYT/UniDetRet-exp/llama.cpp/quantize ./vit/human-llama3.gguf ./vit/human-llama3-q4_0.gguf q4_0

6. And finally we can run the llava-cli using the model version:
./llama.cpp/llava-cli -m ./vit/human-llama3.gguf --mmproj ./vit/mmproj-model-f16.gguf --image 1.jpg -c 4096 -e -p "<|start_header_id|>user<|end_header_id|>\n\n<image>\nDescribe this image<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

7. Ollama
ollama create human-llama3 -f Modelfile
ollama run human-llama3 "/home/ubuntu/san/LYT/UniDetRet-exp/HumanLlama3/1.jpg Describe this image"
```
### 3. Ollama
1. config file
```yaml
FROM ./human-llama3.gguf
FROM ./mmproj-model-f32.gguf
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER num_keep 4
PARAMETER num_ctx 4096
```

2. network error
```shell
sudo -s
mkdir -p /etc/systemd/system/ollama.service.d
echo '[Service]' > /etc/systemd/system/ollama.service.d/environment.conf
echo 'Environment="HTTPS_PROXY=http://10.16.56.42:7890"' >> /etc/systemd/system/ollama.service.d/environment.conf
echo 'Environment="HTTP_PROXY=http://10.16.56.42:7890"' >> /etc/systemd/system/ollama.service.d/environment.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
```


