import torch
import torch.nn.functional as F
import os
import time
from PIL import Image
from .utils import load_image, encode_image, get_task_instruction, format_answer

def ICL_I2T_inference(args, engine, dataset, model, tokenizer, query, 
                    data_path, processor, max_new_tokens):
    if args.CoT:
        task_instruction = ""
    else:
        task_instruction = get_task_instruction(args, dataset)
    img_ids = query['images']
    query_images, query_image_paths = load_image(img_ids, data_path)
    query_text = query['questions']
    if 'qwen-vl' in engine:
        inputs = [{'text': f'You are a helpful assistant. {task_instruction}'}]
        query_text = query_text.replace('<img>', '< img >') # this causes error in some subsets as <img> is a special token for qwen-vl
        for query_image_path in query_image_paths:
            inputs.append({'image': query_image_path})
        inputs.append({'text': query_text})
        
        total_inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(total_inputs, return_tensors='pt')
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(pred[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'llava' in engine:
        from llava.conversation import conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token
        images = []
        input_text = f"{task_instruction}\n"
        
        for query_image in query_images:
            images.append(query_image)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += f"{query_text}\nAnswer:"
        if len(images) == 0:
            image_tensor = None
        else:
            image_tensor = torch.stack(
                    [
                        processor.preprocess(image_file, return_tensors="pt")["pixel_values"][0]
                        for image_file in images
                    ]
                )
            image_tensor = image_tensor.half().cuda()
        conv_mode = 'llava_v1'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                )
        predicted_answers = tokenizer.batch_decode(generated_ids[:, :], skip_special_tokens=True)[0]
    elif "phi3-vision" in engine:
        messages = [{"role": "user", "content": task_instruction}]
        image_count = 0
        
        input_text = ""
        for query_image in query_images:
            input_text += f"<|image_{image_count+1}|>\n"
            image_count += 1
        input_text += f"{query_text}"
        messages.append({"role": "user", "content": input_text})
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(query_images) == 0:
            query_images = None
        inputs = processor(prompt, query_images, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                eos_token_id=processor.tokenizer.eos_token_id
                )
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
    elif 'vila' in engine:
        import sys
        sys.path.append('/mnt/ceph_rbd/multimodal/long-vllm/VILA')
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
        )
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        images = []
        input_text = f"{task_instruction}\n"
        
        for query_image in query_images:
            images.append(query_image)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += query_text
        conv_mode = 'vicuna_v1'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if len(images) == 0:
            image_tensor = None
        else:
            images_tensor = process_images(
                images,
                processor,
                model.config
            ).to(model.device, dtype=torch.float16)
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                use_cache=False
                )
        predicted_answers = tokenizer.batch_decode(generated_ids[:, :], skip_special_tokens=True)[0]
    
    elif 'internlm-x2' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        
        for query_image in query_images:
            images.append(model.vis_processor(query_image))
            input_text += "<ImageHere>"
        input_text += f"{query_text}"
        if len(images) == 0:
            images = None
        else:
            images = torch.stack(images).to(torch.bfloat16).cuda()
        predicted_answers, history = model.chat(tokenizer, query=input_text, image=images, history=[], do_sample=False, max_new_tokens=max_new_tokens)
    elif 'emu2-chat' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for query_image in query_images:
            images.append(query_image)
            input_text += "[<IMG_PLH>]"
        input_text += f"[{query_text}"
        if len(images) == 0:
            images = None
        inputs = model.build_input_ids(
            text=[input_text],
            tokenizer=tokenizer,
            image=images
        )
        
        with torch.no_grad():
            predicted_answers = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16) if images is not None else None,
                max_new_tokens=max_new_tokens,)
        predicted_answers = tokenizer.decode(predicted_answers[:, :].cpu()[0], skip_special_tokens=True)
        
    elif 'idefics1' in engine:
        prompts = [f"You are a helpful assistant.\n{task_instruction}\n"]
        for query_image in query_images:
            prompts.append(query_image)
        prompts.append(f"\nUser: {query_text}")
        #prompts.append("<end_of_utterance>")
        prompts.append("\nAssistant:")
        inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = model.generate(**inputs, 
                                       eos_token_id=exit_condition, 
                                       bad_words_ids=bad_words_ids, 
                                       max_new_tokens=max_new_tokens,
                                       do_sample=False)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(generated_ids[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'idefics2' in engine:
        # https://huggingface.co/docs/transformers/en/model_doc/idefics2#usage-tips
        content = [{"type": "text", "text": task_instruction}]
        for query_image in query_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": query_text})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=query_images, text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = processor.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
    elif 'gemini' in engine:
        import litellm
        os.environ['GEMINI_API_KEY'] = "your_api_key_here"
        content = [{
                "type": "text",
                "text": f"{task_instruction}\nEnsure the generated answers only contain the answer to the question and no other information."
            }]
        for query_image_path in query_image_paths:
            base64_image, mime_type = encode_image(query_image_path)
            content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}",
                                  "detail": "low"},
                    
            })
        content.append({
                "type": "text",
                "text": query_text
        })
        messages = [{
            "role": "user",
            "content": content
        }]

        response = litellm.completion(
            model="gemini/gemini-pro-vision",
            messages=messages,
        )
        predicted_answers = response.get('choices', [{}])[0].get('message', {}).get('content')

    elif 'gpt4v' in engine:
        def _log_when_fail(retry_state):
            print(
                f"Request failed. Current retry attempts: {retry_state.attempt_number}. "
                f"Sleep for {retry_state.idle_for:.2f}. Exception: {repr(retry_state.outcome.exception())}"
            )
        from openai import AzureOpenAI
        from tenacity import retry, wait_random_exponential, stop_after_attempt
        # configure your openai key by `export OPENAI_API_KEY=""` in command line
        def ask_gpt4v_azure(messages, max_tokens):
            response = generate_with_retry(
                model="gpt-4-vision-preview-nofilter",
                messages=messages,
                max_tokens=max_tokens,
            )
            if "error" not in response:
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                print(f"Request cost is ${prompt_tokens / 1000 * 0.01 + completion_tokens / 1000 * 0.03:.3f}")
            return response
    
        # azure_endpoint = os.environ['AZURE_END_POINT']
        # api_key = os.environ['AZURE_KEY']
        api_version = "2023-05-15"
        client = AzureOpenAI(
            azure_endpoint="",
            api_key="azure_key",  # west us
            api_version=api_version,
        )
        
        generate_with_retry = retry(
            wait=wait_random_exponential(min=1, max=5),
            stop=stop_after_attempt(15),
            before_sleep=_log_when_fail
        )(client.chat.completions.create)
        
        content = [{
                "type": "text",
                "text": f"{task_instruction}\nEnsure the generated answers only contain the answer to the question and no other information."
            }]
        
        for query_image_path in query_image_paths:
            base64_image, mime_type = encode_image(query_image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": "low"},
            })
        content.append({
            "type": "text",
            "text": query_text
        })

        messages = [{
            "role": "user",
            "content": content
        }]
        response = ask_gpt4v_azure(messages, max_new_tokens)
        predicted_answers = response.choices[0].message.content

    return predicted_answers


def I2T_first_prob(args, engine, dataset, model, tokenizer, query, 
                    data_path, processor):
    task_instruction = get_task_instruction(args, dataset)
    img_ids = query['images']
    query_images, query_image_paths = load_image(img_ids, data_path)
    query_text = query['questions']

    if 'vila' in engine:
        # https://github.com/Efficient-Large-Model/VILA
        import sys
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
        )
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        images = []
        input_text = f"{task_instruction}\n"
        
        for query_image in query_images:
            images.append(query_image)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += query_text
        conv_mode = 'vicuna_v1'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if len(images) == 0:
            images_tensor = None
        else:
            images_tensor = process_images(
                    images,
                    processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            output = model(
                input_ids,
                images=images_tensor
                )
    elif "phi3-vision" in engine:
        messages = [{"role": "user", "content": task_instruction}]
        image_count = 0
        
        input_text = ""
        for query_image in query_images:
            input_text += f"<|image_{image_count+1}|>\n"
            image_count += 1
        input_text += f"{query_text}"
        messages.append({"role": "user", "content": input_text})
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(query_images) == 0:
            query_images = None
        inputs = processor(prompt, query_images, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model(
                **inputs,
                )
    elif 'qwen-vl' in engine:
        inputs = [{'text': f'You are a helpful assistant. {task_instruction}'}]
        
        for query_image_path in query_image_paths:
            inputs.append({'image': query_image_path})
        query_text = query_text.replace('<img>', '< img >') # this causes error in some subsets as <img> is a special token for qwen-vl
        inputs.append({'text': query_text})
        total_inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(total_inputs, return_tensors='pt')
        inputs = inputs.to(model.device)
        with torch.no_grad():
            output = model(**inputs)
    elif 'llava' in engine:
        from llava.conversation import conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token
        images = []
        input_text = f"{task_instruction}\n"
        
        for query_image in query_images:
            images.append(query_image)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += f"{query_text}\nAnswer:"
        if len(images) == 0:
            image_tensor = None
        else:
            image_tensor = torch.stack(
                    [
                        processor.preprocess(image_file, return_tensors="pt")["pixel_values"][0]
                        for image_file in images
                    ]
                )
            image_tensor = image_tensor.half().cuda()
        conv_mode = 'llava_v1'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(input_ids, images=image_tensor,)
    elif 'internlm-x2' in engine:
        images = []
        meta_instruction ='You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.',
        input_text = f"{task_instruction}\n"
        
        for query_image in query_images:
            images.append(model.vis_processor(query_image))
            input_text += "<ImageHere>"
        input_text += f"{query_text}"
        if len(images) == 0:
            images = None
            inputs = model.build_inputs(tokenizer, input_text, [], meta_instruction)
        else:
            images = torch.stack(images).to(torch.bfloat16).cuda()
            images = model.encode_img(images)
            inputs, _ = model.interleav_wrap_chat(tokenizer, input_text, images, history=[], meta_instruction=meta_instruction)
        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items() if torch.is_tensor(v)
        }
        with torch.no_grad():
            output = model(**inputs)
    elif 'idefics1' in engine:
        prompts = [f"You are a helpful assistant.\n{task_instruction}\n"]
        for query_image in query_images:
            prompts.append(query_image)
        prompts.append(f"\nUser: {query_text}")
        #prompts.append("<end_of_utterance>")
        prompts.append("\nAssistant:")
        inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(**inputs,)
    elif 'idefics2' in engine:
        # https://huggingface.co/docs/transformers/en/model_doc/idefics2#usage-tips
        content = [{"type": "text", "text": task_instruction}]
        for query_image in query_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": query_text})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        if len(query_images) == 0:
            query_images = None
        inputs = processor(images=query_images, text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model(**inputs,)
    logits = output.logits
    token_probs = F.softmax(logits[0, -1, :], dim=-1)

    return token_probs