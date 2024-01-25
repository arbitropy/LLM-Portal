import os
import argparse
from typing import Iterator
import gradio as gr
from dotenv import load_dotenv
from distutils.util import strtobool
from inference_scripts import INFERENCE
import logging
import torchaudio
import langid
import numpy as np
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoProcessor, SeamlessM4Tv2Model
from seamless_communication.inference import Translator
import torch 


# loading facebookm4t model full, look into chunked model
AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

# using pipeline doesn't work with gradio, below is from facebooks demo
translator = Translator(
    model_name_or_card="seamlessM4T_v2_large",
    vocoder_name_or_card="vocoder_v2",
    device=device,
    dtype=dtype,
    apply_mintox=True,
)

def preprocess_audio(input_audio: str) -> None:
    if type(input_audio) is not str:
        return input_audio
    arr, org_sr = torchaudio.load(input_audio)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    torchaudio.save(input_audio, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
# translate 
def run_t2tt(input_text: str, source_language: str, target_language: str) -> str:
    out_texts, _ = translator.predict(
        input=input_text,
        task_str="T2TT",
        src_lang=source_language,
        tgt_lang=target_language,
    )
    return str(out_texts[0])

def translate(message):
    langid.set_languages(['ar', 'en'])
    lang, score = langid.classify(message)
    if (lang == 'en'):
        return run_t2tt(message, "eng", "arb")
    elif (lang == 'ar'):
        return run_t2tt(message, "arb", "eng")
    else:
        return message

def run_asr(input_audio: str) -> str:
    if input_audio is None:
        return ""
    preprocess_audio(input_audio)
    target_language_code = "eng"
    out_texts, _ = translator.predict(
        input=input_audio,
        task_str="ASR",
        src_lang=target_language_code,
        tgt_lang=target_language_code,
    )
    return str(out_texts[0])

# # whisper ASR setup
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# import whisper

# whisperModel = whisper.load_model("large-v3")
# def run_asr(input_audio: str) -> str:
#     if input_audio is None:
#         return ""
#     return whisperModel.transcribe(input_audio)["text"]

# # facebook mmt translation setup
# mmtModel = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# mmtTokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# def run_t2tt(input_text: str, source_language: str, target_language: str) -> str:
#     mmtTokenizer.src_lang = source_language
#     encoded_ar = mmtTokenizer(input_text, return_tensors="pt")
#     generated_tokens = mmtModel.generate(
#     **encoded_ar,
#     forced_bos_token_id=mmtTokenizer.lang_code_to_id[target_language]
#     )
#     return mmtTokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# def translate(message):
#     langid.set_languages(['ar', 'en'])
#     lang, score = langid.classify(message)
#     if (lang == 'en'):
#         return run_t2tt(message, "en_XX", "ar_AR")
#     elif (lang == 'ar'):
#         return run_t2tt(message, "ar_AR", "en_XX")
#     else:
#         return message

load_dotenv()


DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "")
MAX_MAX_NEW_TOKENS = int(os.getenv("MAX_MAX_NEW_TOKENS", 2048))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 1024))
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", 4000))

MODEL_PATH = os.getenv("MODEL_PATH")
assert MODEL_PATH is not None, f"MODEL_PATH is required, got: {MODEL_PATH}"
BACKEND_TYPE = "transformers"

LOAD_IN_8BIT = bool(strtobool(os.getenv("LOAD_IN_8BIT", "True")))

model_instance = INFERENCE(
    model_path=MODEL_PATH,
    backend_type=BACKEND_TYPE,
    max_tokens=MAX_INPUT_TOKEN_LENGTH,
    load_in_8bit=LOAD_IN_8BIT,
    # verbose=True,
)

DESCRIPTION = """
# **HajjLLM**
Disclaimer: This model is experimental and may produce wrong or hallucinated responses. Do not take them as religious rulings or authoritative guidance. This system is in a testing phase and is continually being improved.
"""

def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return "", message

def check_empty_input(message: str):
    if len(message) == 0:
        raise gr.Error(
            f"Please provide a valid non empty input."
        )

def check_empty_audio(audio):
    if audio is None:
        raise gr.Error(
            f"Please provide a non empty audio."
        )

def display_input(
    message: str, history: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    
    history.append((message, ""))
    return history

def delete_prev_fn(
    history: list[tuple[str, str]]
) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ""
    return history, message or ""


def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError
    try:
        langid.set_languages(['ar', 'en'])
        lang, score = langid.classify(message)
        english = (lang == 'en')
        print(lang)
        # print(history_with_input)
        history = history_with_input[:-1]
        # tranlates message if not in english
        if (not english):
            print(message)
            processedMessage = translate(message) # translate for generation if not english
            print(processedMessage)
        else:
            processedMessage = message
            
        generator = model_instance.run(
            processedMessage,
            history,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        )
        
        try:
            first_response = next(generator)
            yield history + [(message, first_response)]
        except StopIteration:
            yield history + [(message, "")]
        for response in generator:
            yield history + [(message, response)]
        # translate after the entire chat is generated
        # if(not english):
        #     print(response)
        #     response = translate(response) 
        #     print(response)
        #     yield history + [(message, response)]
        print("response: "+response)
    except Exception as e:
        logging.exception(e)

def check_input_token_length(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
) -> None:
    input_token_length = model_instance.get_input_token_length(
        message, chat_history, system_prompt
    )
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        raise gr.Error(
            f"The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again."
        )

with open("custom.css", "r", encoding="utf-8") as f:
    CSS = f.read()

with gr.Blocks(css=CSS, theme=gr.themes.Default(text_size = "sm")) as demo:
    state = gr.State(value="")
    with gr.Row():
        gr.Markdown(DESCRIPTION)
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            with gr.Group():
                chatbot = gr.Chatbot(label="Chatbot")
                with gr.Row():
                    textbox = gr.Textbox(
                        container=False,
                        show_label=False,
                        placeholder="Type a message...",
                        scale=20,
                    )
                    submit_button = gr.Button(
                        "Submit", variant="primary", scale=1, min_width=0
                    )
        with gr.Column():
            with gr.Row():
                with gr.Group():
                    with gr.Column():
                        audio2 = gr.Audio(sources = ["microphone"], type="filepath", interactive = True) 
                        audio_submit = gr.Button("Submit") 
            with gr.Row():
                upload_checkbox = gr.Checkbox(
                    label="Upload",
                    value=False,
                    container=False,
                    elem_classes="min_check",
                )
            with gr.Row(visible = False) as upload_row:
                audio = gr.Audio(sources = ["upload"], type="filepath")
                
            with gr.Row():
                retry_button = gr.Button("🔄  Retry", variant="secondary")
                undo_button = gr.Button("↩️ Undo", variant="secondary")
                clear_button = gr.Button("🗑️  Clear", variant="secondary")

            saved_input = gr.State()
            with gr.Row():
                advanced_checkbox = gr.Checkbox(
                    label="Advanced",
                    value=False,
                    container=False,
                    elem_classes="min_check",
                )
            with gr.Column(visible=False) as advanced_column:
                system_prompt = gr.Textbox(
                    label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=6
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    minimum=1,
                    maximum=MAX_MAX_NEW_TOKENS,
                    step=1,
                    value=DEFAULT_MAX_NEW_TOKENS,
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.01,
                    maximum=4.0,
                    step=0.1,
                    value=0.0,
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    minimum=0.05,
                    maximum=1.0,
                    step=0.05,
                    value=0.95,
                )
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=1,
                    maximum=1000,
                    step=1,
                    value=50,
                )
        advanced_checkbox.change(
            lambda x: gr.update(visible=x),
            advanced_checkbox,
            advanced_column,
            queue=False, 
        )
        upload_checkbox.change(
            lambda x: gr.update(visible=x),
            upload_checkbox,
            upload_row,
            queue=False,
        ) 

    textbox.submit(
        fn=check_empty_input,
        inputs=textbox,
        api_name=False,
        queue=False,
    ).success(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    audio_submit.click(
        fn=check_empty_audio,
        inputs=audio2,
        api_name=False,
        queue=False,
    ).success(
        fn=run_asr,
        inputs=audio2,
        outputs=textbox,
        api_name="asr",
    )
    audio.upload(
        fn=run_asr,
        inputs=audio,
        outputs=textbox,
        api_name="ast",
    )

    button_event_preprocess = (
        submit_button.click(
        fn=check_empty_input,
        inputs=textbox,
        api_name=False,
        queue=False,
        ).success(
            fn=clear_and_save_textbox,
            inputs=textbox,
            outputs=[textbox, saved_input],
            api_name=False,
            queue=False,
        )
        .then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        )
        .then(
            fn=check_input_token_length,
            inputs=[saved_input, chatbot, system_prompt],
            api_name=False,
            queue=False,
        )
        .success(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )

    clear_button.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch()
