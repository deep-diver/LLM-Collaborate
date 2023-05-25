import copy
import re
from typing import List
from pingpong.pingpong import PPManager

import gradio as gr
import random
import time

from pingpong.pingpong import PingPong
from pingpong.gradio import GradioAlpacaChatPPManager
from pingpong.gradio import GradioVicunaChatPPManager

import transformers
from transformers import GenerationConfig
from llmpool import LLModelPool
from llmpool import LocalLLModel, LocalLoRALLModel
from llmpool.model import LLModelMetadata

model_pool = LLModelPool()
model_pool.add_models([
  # alpaca-lora 13b
  LocalLoRALLModel(
    "alpaca-lora-13b",
    GenerationConfig(
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        num_beams=1,
        repetition_penalty=1.2,
        max_new_tokens=1024,
        do_sample=True,
    ),
    "elinas/llama-13b-hf-transformers-4.29",
    "LLMs/Vicuna-LoRA-EvolInstruct-13B",
    model_cls=transformers.AutoModelForCausalLM,
    metadata=LLModelMetadata(
        "https://i.ibb.co/PNDbvm7/alpaca-lora-2.png",
        "https://i.ibb.co/PNDbvm7/alpaca-lora-2.png",
    ),
    load_in_8bit=False
  ),

  LocalLoRALLModel(
    "gpt4-alpaca-lora-7b",
    GenerationConfig(
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        num_beams=1,
        repetition_penalty=1.2,
        max_new_tokens=1024,
        do_sample=True,
    ),
    "elinas/llama-7b-hf-transformers-4.29",
    "LLMs/AlpacaGPT4-LoRA-7B-elina",
    model_cls=transformers.AutoModelForCausalLM,
    metadata=LLModelMetadata(
        "https://i.ibb.co/920FmrD/gpt4-alpaca-lora.png",
        "https://i.ibb.co/920FmrD/gpt4-alpaca-lora.png",
    ),
    load_in_8bit=False      
  ),
    
  LocalLLModel(
    "t5-vicuna-3b",
    GenerationConfig(
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        num_beams=1,
        repetition_penalty=1.2,
        max_new_tokens=1024,
        do_sample=True,
    ),
    "lmsys/fastchat-t5-3b-v1.0",
    model_cls=transformers.AutoModelForSeq2SeqLM,
    tokenizer_cls=transformers.T5Tokenizer,
    metadata=LLModelMetadata(
        "https://i.ibb.co/cbjDCd6/t5-vicuna-logo.png",
        "https://i.ibb.co/cbjDCd6/t5-vicuna-logo.png",
    ),
    # load_in_8bit=False      
  )]
)

PARENT_BLOCK_CSS = """
#col-container {
    width: 95%; 
    margin-left: auto;
    margin-right: auto;
}

#chatbot {
    height: 800px; 
    overflow: auto;
}

#chatbot > .wrap {
    max-height: 800px;
}

.border {
    border-width: 1px;
    border-radius: 10px;
    padding: 7px;
    border-style: dashed;

    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 3; /* number of lines to show */
            line-clamp: 3;
    -webkit-box-orient: vertical;

    transition: all .2s ease-in-out;
}

.border:hover {
    height: 200px;
    background-color: #e89f69;
    z-index: 1000;
    white-space: normal;
    overflow: auto;
    text-overflow: unser;    
}
"""
####### 
def remove_markdown_images(text):
    pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    result = re.sub(pattern, '', text)
    
    result = result.replace(
        '<img src="https://i.ibb.co/920FmrD/gpt4-alpaca-lora.png" alt="">',
        ''
    ).replace(
        '<img src="https://i.ibb.co/cbjDCd6/t5-vicuna-logo.png" alt="">',
        ''
    ).replace(
        '<img src="https://i.ibb.co/PNDbvm7/alpaca-lora-2.png" alt="">',
        ''
    )

    pattern = re.compile(r'<p[^>]*>([^<]*)</p>')
    return re.sub(pattern, '\\1', result)
    
def select_first(btn, first_option, chat_state):
    ppm = chat_state["ppmanager"]
    ppm.replace_pong(first_option, at=0)
    chat_state["ppmanager"] = ppm
    return "Select", ppm.build_uis(), chat_state

def select_second(btn, second_option, chat_state):
    ppm = chat_state["ppmanager"]
    
    if btn == "GPT4Alpaca-LoRA(7B)":
        second_model = model_pool.models["gpt4-alpaca-lora-7b"]
        
        second_prompt = remove_markdown_images(ppm.ppmanagers[1].build_prompts())
        second_msg = f"![]({second_model.metadata.thumb_xs_path})"
        second_msg += second_model.batch_gen([second_prompt])[0].split("### Response:")[-1].strip()
        
        ppm.replace_pong(second_msg, at=0)
        chat_state["ppmanager"] = ppm
        return "Select", second_msg, ppm.build_uis(), chat_state
        
    elif btn == "Select":
        ppm.replace_pong(second_option)
        chat_state["ppmanager"] = ppm
        return btn, second_option, ppm.build_uis(), chat_state

def select_third(btn, third_option, chat_state):
    ppm = chat_state["ppmanager"]
    
    if btn == "VicunaT5(3B)":
        third_model = model_pool.models["t5-vicuna-3b"]
        
        third_prompt = remove_markdown_images(ppm.ppmanagers[2].build_prompts())
        third_msg = f"![]({third_model.metadata.thumb_xs_path})"
        third_msg += third_model.batch_gen([third_prompt])[0]
        
        ppm.replace_pong(third_msg, at=0)
        chat_state["ppmanager"] = ppm
        return "Select", third_msg, ppm.build_uis(), chat_state
    
    elif btn == "Select":
        ppm.replace_pong(third_option)
        chat_state["ppmanager"] = ppm
        return btn, third_option, ppm.build_uis(), chat_state

def respond(msg, chat_state):
    original_ppm = chat_state["ppmanager"]
    original_ppm.add_pingpong(
        PingPong(msg, "")
    )
    chat_state["ppmanager"] = original_ppm
    
    ppm = copy.deepcopy(original_ppm)

    model = model_pool.models["alpaca-lora-13b"]
    response = f"![]({model.metadata.thumb_xs_path})"
    prompt = remove_markdown_images(ppm.ppmanagers[0].build_prompts())
    print(prompt)
    _, streamer = model.stream_gen(prompt)
    
    for text in streamer:
        response += text + " "
        ppm.replace_pong(response)
        yield ppm.build_uis(), chat_state, response
  
    yield ppm.build_uis(), chat_state, response
  
def respond_alternative(chat_state):
    ppm = chat_state["ppmanager"]
    
    second_model = model_pool.models["gpt4-alpaca-lora-7b"]
    third_model = model_pool.models["t5-vicuna-3b"]
    
    second_prompt = remove_markdown_images(ppm.ppmanagers[1].build_prompts())
    second_msg = f"![]({second_model.metadata.thumb_xs_path})"
    second_msg += second_model.batch_gen([second_prompt])[0].split("### Response:")[-1].strip()
    
    third_prompt = remove_markdown_images(ppm.ppmanagers[2].build_prompts())
    third_msg = f"![]({third_model.metadata.thumb_xs_path})"
    third_msg += third_model.batch_gen([third_prompt])[0]

    return "", second_msg, third_msg

class MirrorGroupPPManager:
    def __init__(self, ppms: List[PPManager]):
        self.ppmanagers = ppms

    def append_pong(self, piece_pong, at=0):
        self.ppmanagers[at].append_pong(piece_pong)

    def add_pingpong(self, pingpong, at=None):
        if at == None:
            for ppm in self.ppmanagers:
                ppm.add_pingpong(copy.deepcopy(pingpong))
        else:      
            self.ppmanagers[at].add_pingpong(copy.deepcopy(pingpong))

    def replace_pong(self, pong, at=None):
        if at == None:
            for ppm in self.ppmanagers:
                ppm.pingpongs[-1].pong = pong
        else:
            self.ppmanagers[at].pingpongs[-1].pong = pong
            
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, truncate_size: int=None):
        results = []
        for ppm in self.ppmanagers:
            results.append(
                ppm.build_prompts(
                    from_idx=from_idx, 
                    to_idx=to_idx, 
                    truncate_size=truncate_size
                )
            )
        return results

    def build_uis(self, from_idx: int=0, to_idx: int=-1):
        return self.ppmanagers[0].build_uis(
            from_idx=from_idx, to_idx=to_idx
        )

def reset_btn_titles():
    return (
        "GPT4Alpaca-LoRA(7B)",
        "",
        "VicunaT5(3B)",
        ""
    )
    
with gr.Blocks(css=PARENT_BLOCK_CSS, theme='gradio/soft') as demo:
    choice = gr.State(0)
    chat_state = gr.State({
        "ppmanager": MirrorGroupPPManager(
            [
                GradioVicunaChatPPManager(),
                GradioAlpacaChatPPManager(),
                GradioVicunaChatPPManager()
            ]
        ),
        "choice": 0
    })

    chatbot = gr.Chatbot(elem_id="chatbot")

    msg = gr.Textbox()
    clear = gr.Button("Clear State")

    with gr.Accordion("Alternative Answers", open=False):
        with gr.Row():
            with gr.Column(min_width=100):
                first = gr.Button("Alpaca-LoRA(13B)")
                first_option = gr.Markdown("", elem_classes=["border"])

            with gr.Column(min_width=100):
                second = gr.Button("GPT4Alpaca-LoRA(7B)")
                second_option = gr.Markdown("", elem_classes=["border"])

            with gr.Column(min_width=100):
                third = gr.Button("VicunaT5(3B)")
                third_option = gr.Markdown("", elem_classes=["border"])

    msg.submit(
        respond,
        [msg, chat_state],
        [chatbot, chat_state, first_option],
    ).then(
        select_first,
        [first, first_option, chat_state],
        [first, chatbot, chat_state]
    ).then(
        reset_btn_titles,
        None,
        [second, second_option, third, third_option],
    )
    
    first.click(
        select_first,
        [first, first_option, chat_state], 
        [first, chatbot, chat_state]
    )

    second.click(
        select_second,
        [second, second_option, chat_state], 
        [second, second_option, chatbot, chat_state]
    )

    third.click(
        select_third,
        [third, third_option, chat_state], 
        [third, third_option, chatbot, chat_state]
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch(share=True, debug=True, server_name="0.0.0.0", server_port=6006)
