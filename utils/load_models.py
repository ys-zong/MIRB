import transformers
import os
import torch
import sys

def load_i2t_model(engine, args=None):
    if engine == 'llava15-7b':
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='liuhaotian/llava-v1.5-7b', model_base=None, model_name='llava',
                                                                          device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        processor = image_processor
    elif engine == 'llava16-7b':
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='liuhaotian/llava-v1.6-vicuna-7b', model_base=None, model_name='llava',
                                                                          device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        processor = image_processor
    elif engine == 'llava16-13b':
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='liuhaotian/llava-v1.6-vicuna-13b', model_base=None, model_name='llava',
                                                                          device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        processor = image_processor
    elif engine == 'phi3-vision':
        model_id = "microsoft/Phi-3-vision-128k-instruct" 
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16)
        processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
        tokenizer = processor.tokenizer
        processor.image_processor.num_crops=1
    elif 'vila' in engine:
        import sys
        sys.path.append('/mnt/ceph_rbd/multimodal/long-vllm/VILA')
        from llava.model.builder import load_pretrained_model as load_llava_model
        model_path = 'Efficient-Large-Model/VILA-7B' if 'vila-7b' in engine else 'Efficient-Large-Model/VILA-2.7b'
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(model_path)
        
        tokenizer, model, image_processor, context_len = load_llava_model(
                model_path,
                model_base=None, model_name="VILA", 
                attn_implementation="flash_attention_2",
                device_map="auto", 
                config=config,
            )
        processor = image_processor
    elif engine == 'qwen-vl-chat':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", 
                                                                  trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        processor = None
    elif engine == 'qwen-vl':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        processor = None
    elif engine == 'internlm-x2':
        model = transformers.AutoModel.from_pretrained('internlm/internlm-xcomposer2-7b', trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
        tokenizer = transformers.AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-7b', trust_remote_code=True)
        model.tokenizer = tokenizer
        processor = None
    elif engine == 'internlm-x2-hd':
        model = transformers.AutoModel.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
        tokenizer = transformers.AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', trust_remote_code=True)
        model.tokenizer = tokenizer
        processor = None
    
    elif engine == 'emu2-chat':
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
        tokenizer = transformers.AutoTokenizer.from_pretrained("BAAI/Emu2-Chat")
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_pretrained(
                "BAAI/Emu2-Chat",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).eval()
        # adjust according to your device
        device_map = infer_auto_device_map(model, max_memory={0:'38GiB',1:'38GiB',2:'38GiB',3:'38GiB'}, no_split_module_classes=['Block','LlamaDecoderLayer'])
        device_map["model.decoder.lm.lm_head"] = 0

        model = load_checkpoint_and_dispatch(
            model, 
            'your/path/to/snapshot',
            device_map=device_map).eval()
        processor = None
    elif engine == 'idefics1-9b-instruct':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "HuggingFaceM4/idefics-9b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics1-9b':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "HuggingFaceM4/idefics-9b"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics1-80b-instruct':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
        checkpoint = "HuggingFaceM4/idefics-80b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics2-8b':
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
        model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b", torch_dtype=torch.bfloat16, 
                                                                 device_map="auto", low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer = processor.tokenizer
    elif 'gpt4v' in engine or 'gemini' in engine:
        model, tokenizer, processor = None, None, None
    else:
        raise NotImplementedError
    return model, tokenizer, processor