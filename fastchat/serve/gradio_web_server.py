"""
The gradio demo server for chatting with a single model.
"""

import argparse
from collections import defaultdict
import datetime
import json
import os
import random
import time
import uuid

import gradio as gr
import requests

from fastchat.conversation import SeparatorStyle
from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INACTIVE_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.model.model_registry import get_model_info, model_info
from fastchat.serve.api_provider import (
    anthropic_api_stream_iter,
    openai_api_stream_iter,
    palm_api_stream_iter,
    init_palm_chat,
)
from fastchat.utils import (
    build_logger,
    violates_moderation,
    get_window_url_params_js,
    # get_window_url_params_js,
    parse_gradio_auth_creds,
)


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "FastChat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True, visible=True)
disable_btn = gr.Button.update(interactive=False)
invisible_btn = gr.Button.update(interactive=False, visible=False)

controller_url = None
enable_moderation = False

acknowledgment_md = """
### Acknowledgment
<div class="image-container">
    <p> We thank <a href="https://www.kaggle.com/" target="_blank">Kaggle</a>, <a href="https://mbzuai.ac.ae/" target="_blank">MBZUAI</a>, <a href="https://www.anyscale.com/" target="_blank">AnyScale</a>, and <a href="https://huggingface.co/" target="_blank">HuggingFace</a> for their <a href="https://lmsys.org/donations/" target="_blank">sponsorship</a>. </p>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Kaggle_logo.png/400px-Kaggle_logo.png" alt="Image 1">
    <img src="https://mma.prnewswire.com/media/1227419/MBZUAI_Logo.jpg?p=facebookg" alt="Image 2">
    <img src="https://docs.anyscale.com/site-assets/logo.png" alt="Image 3">
    <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" alt="Image 4">
</div>
"""

ip_expiration_dict = defaultdict(lambda: 0)

# Information about custom OpenAI compatible API models.
# JSON file format:
# {
#     "vicuna-7b": {
#         "model_name": "vicuna-7b-v1.5",
#         "api_base": "http://8.8.8.55:5555/v1",
#         "api_key": "password"
#     },
# }
openai_compatible_models_info = {}


class State:
    def __init__(self, model_name):
        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name

        if model_name == "palm-2":
            # According to release note, "chat-bison@001" is PaLM 2 for chat.
            # https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023
            self.palm_chat = init_palm_chat("chat-bison@001")

    def to_gradio_chatbot(self):
        return self.conv.to_gradio_chatbot()

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )
        return base


def set_global_vars(controller_url_, enable_moderation_):
    global controller_url, enable_moderation
    controller_url = controller_url_
    enable_moderation = enable_moderation_


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(
    controller_url, register_openai_compatible_models, add_chatgpt, add_claude, add_palm
):
    if controller_url:
        ret = requests.post(controller_url + "/refresh_all_workers")
        assert ret.status_code == 200
        ret = requests.post(controller_url + "/list_models")
        models = ret.json()["models"]
    else:
        models = []

    # Add API providers
    if register_openai_compatible_models:
        global openai_compatible_models_info
        openai_compatible_models_info = json.load(
            open(register_openai_compatible_models)
        )
        models += list(openai_compatible_models_info.keys())

    if add_chatgpt:
        models += ["gpt-3.5-turbo", "gpt-4"]
    if add_claude:
        models += ["claude-2", "claude-instant-1"]
    if add_palm:
        models += ["palm-2"]
    models = list(set(models))

    priority = {k: f"___{i:02d}" for i, k in enumerate(model_info)}
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


def load_demo_single(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown.update(
        choices=models, value=selected_model, visible=True
    )

    state = None
    return state, dropdown_update


def load_demo(url_params, request: gr.Request):
    global models

    ip = request.client.host
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    if args.model_list_mode == "reload":
        models = get_model_list(
            controller_url,
            args.register_openai_compatible_models,
            args.add_chatgpt,
            args.add_claude,
            args.add_palm,
        )

    return load_demo_single(models, url_params)


def take_user_feedback(state, user_answer, user_reason, user_name, user_PID, model_selector, request: gr.Request):
    logger.info(f"take_user_feedback. ip: {request.client.host}")
    
    if len(user_PID) <= 0:
        raise gr.Error("Please enter PID.")
    
    if len(user_name) <= 0:
        raise gr.Error("Please enter name.")
    
    if len(user_answer) <= 0 and len(user_reason) <= 0:
        raise gr.Error("Please enter answer or reason.")
    
    
    # save user feedback
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": "user_feedback",
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
            "user_answer": user_answer,
            "user_reason": user_reason,
            "user_name": user_name,
            "user_PID": user_PID,
        }
        fout.write(json.dumps(data) + "\n")
    gr.Info("Feedback submitted. Thank you!")

    return state, disable_btn

# def activate_user_submit_btn(state, user_answer, user_reason, user_name, user_PID, request: gr.Request):
#     logger.info(f"activate_user_submit_btn. ip: {request.client.host}")
#     if state is not None: 
#         # if len(user_answer) <= 0 or len(user_reason) <= 0 or len(user_name) <= 0 or len(user_PID) <= 0:  # nothing entered
#         #     return no_change_btn
#         if len(user_name) > 0 and len(user_PID) > 0 and len(user_answer) > 0:  # entered user_name, user_PID, user_answer
#             return enable_btn
#         elif len(user_name) > 0 and len(user_PID) > 0 and len(user_reason) > 0 : # entered user_name, user_PID, user_reason
#             return enable_btn
#         else:
#             return disable_btn
#     else:
#         return disable_btn

def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, user_answer, user_reason, user_name, user_PID, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    # user_submit_btn = activate_user_submit_btn(state, user_answer, user_reason, user_name, user_PID, request)
    
    return ("",) + (disable_btn,) * 3 + (enable_btn,)


def downvote_last_response(state, model_selector, user_answer, user_reason, user_name, user_PID, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    # user_submit_btn = activate_user_submit_btn(state, user_answer, user_reason, user_name, user_PID, request)
    
    return ("",) + (disable_btn,) * 3  + (enable_btn,) 


def flag_last_response(state, model_selector, user_answer, user_reason, user_name, user_PID, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    # user_submit_btn = activate_user_submit_btn(state, user_answer, user_reason, user_name, user_PID, request)

    return ("",) + (disable_btn,) * 3  + (enable_btn,)


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.conv.update_last_message(None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (state, [], "") + (disable_btn,) * 6


def add_text(state, model_selector, text, request: gr.Request):
    ip = request.client.host
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5

    if ip_expiration_dict[ip] < time.time():
        logger.info(f"inactive. ip: {request.client.host}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), INACTIVE_MSG) + (no_change_btn,) * 5

    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), MODERATION_MSG) + (
                no_change_btn,
            ) * 5

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {request.client.host}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (
            no_change_btn,
        ) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def model_worker_stream_iter(
    conv,
    model_name,
    worker_addr,
    prompt,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
):
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    logger.info(f"==== request ====\n{gen_params}")

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def bot_response(state, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"bot_response. ip: {request.client.host}")
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    conv, model_name = state.conv, state.model_name
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_name == "claude-2" or model_name == "claude-instant-1":
        prompt = conv.get_prompt()
        stream_iter = anthropic_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_name == "palm-2":
        stream_iter = palm_api_stream_iter(
            state.palm_chat, conv.messages[-2][1], temperature, top_p, max_new_tokens
        )
    elif model_name in openai_compatible_models_info:
        model_info = openai_compatible_models_info[model_name]
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_info["model_name"],
            prompt,
            temperature,
            top_p,
            max_new_tokens,
            api_base=model_info["api_base"],
            api_key=model_info["api_key"],
        )
    else:
        # Query worker address
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (
                state,
                state.to_gradio_chatbot(),
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
                # disable_btn,
            )
            return

        # Construct prompt.
        # We need to call it here, so it will not be affected by "▌".
        prompt = conv.get_prompt()

        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
        )

    conv.update_last_message("▌")
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        for i, data in enumerate(stream_iter):
            if data["error_code"] == 0:
                if i % 8 != 0:  # reduce gradio's overhead
                    continue
                output = data["text"].strip()
                conv.update_last_message(output + "▌")
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                    # disable_btn,
                )
                return
        output = data["text"].strip()
        if "vicuna" in model_name:
            output = post_process_code(output)
        conv.update_last_message(output)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
            # disable_btn,
        )
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
            # disable_btn,
        )
        return

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


block_css = """
#notice_markdown {
    font-size: 104%
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_markdown {
    font-size: 104%
}
#leaderboard_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_dataframe td {
    line-height: 0.1em;
}
#input_box textarea {
}
footer {
    display:none !important
}
.image-container {
    display: flex;
    align-items: center;
    padding: 1px;
}
.image-container img {
    margin: 0 30px;
    height: 20px;
    max-height: 100%;
    width: auto;
    max-width: 20%;
}
"""


def get_model_description_md(models):
    model_description_md = """
| | | |
| ---- | ---- | ---- |
"""
    ct = 0
    visited = set()
    for i, name in enumerate(models):
        minfo = get_model_info(name)
        if minfo.simple_name in visited:
            continue
        visited.add(minfo.simple_name)
        one_model_md = f"[{minfo.simple_name}]({minfo.link}): {minfo.description}"

        if ct % 3 == 0:
            model_description_md += "|"
        model_description_md += f" {one_model_md} |"
        if ct % 3 == 2:
            model_description_md += "\n"
        ct += 1
    return model_description_md


def build_single_model_ui(models, add_promotion_links=False):
    promotion = (
        """
- | [GitHub](https://github.com/lm-sys/FastChat) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |
- Introducing Llama 2: The Next Generation Open Source Large Language Model. [[Website]](https://ai.meta.com/llama/)
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)
"""
        if add_promotion_links
        else ""
    )

    notice_markdown = f"""
# 🤖 LMTutor for DSC250 Advanced Data Mining
Wecome to use LMTutor for answering your questions. You can ask it about the questions in the course material, logistics, etc. No need to wait for the TA's response! LMTutor answers your question within seconds!
"""
### How to use
# * It's easy. Just type your questions in the chatbox below and have a conversation with it just like you are talking to the TA.
# * Based on the answers provided by LMTutor, you can ask follow-up questions or start a new conversation.
# * Based on how satisfied you are with the answer, you can choose to either upvote, downvote or flag (for toxic answers) the answers generated by LMTutor. Make sure to do this step with honesty and integrity as your responses will help us improve LMTutor for you as well as your peers.

    contributing_rules = f"""
### Help the AI Tutor Improve: Your Contribution
* After having a conversation with LMTutor (that started with a question that you asked it), if you feel that the answer you received is incorrect or inaccurate, you can use your initial question as a submission.
* First make a vote to the question. Then try to find and compile an answer to your original question and submit it in the textbox under the chatbot along with your name, PID and a brief explanation for why your answer is better than the ones generated by LMTutor.
* You need to enter both your name and your PID in order to submit a feedback.
* Please DO NOT refresh the page or conversation with the chatbot before submitting your answer.
* That's it! LMTutor will record your conversation history, your details and the answer you submitted, after which someone will evaluate your submission.
"""

    contacts = f"""
### If you encounter any issue or bug, please contact us:
* Hao Zhang: haozhang at ucsd.edu 
* Pushkar Bhuse: pbhuse at ucsd.edu
* Yuheng Zha: yzha at ucsd.edu 
* Tiffany Yu: z5yu at ucsd.edu
* Anze Xie: a1xie at ucsd.edu 
* Licheng Hu: l2hu at ucsd.edu
"""

    state = gr.State()
    model_description_md = get_model_description_md(models)
    # model_description_md = "\n\nLMTutor: a tutor chatbot built based on vicuna-13b by LMTutor-org"
    gr.Markdown(notice_markdown + model_description_md, elem_id="notice_markdown")

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
            container=False,
        )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        label="Scroll down and start chatting",
        height=550,
        show_copy_button=True,
    )

    # with gr.Row():
    with gr.Column(scale=20):
        textbox = gr.TextArea(
            show_label=True,
            label="Enter your prompt here and press SHIFT + ENTER",
            placeholder="Enter your prompt here and press SHIFT + ENTER",
            container=True,
            elem_id="input_box",
            lines=3,
            max_lines=30,
            autofocus=True,
        )
    with gr.Column(scale=1, min_width=50):
        send_btn = gr.Button(value="Send", variant="primary", size="lg")
    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=(1024)*16,
            value=1024,
            step=1,
            interactive=True,
            label="Max output tokens",
        )

    with gr.Row():
        gr.Markdown(contributing_rules, elem_id="contributing_rules")    
    
    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Row():
        user_answer = gr.TextArea(
            show_label=True,
            placeholder="Enter your answer area",
            container=False,
            elem_id="input_box",
            lines=2,
            max_lines=30,
        )
    
    with gr.Row():
        user_reason = gr.TextArea(
            show_label=False,
            placeholder="Enter your expalanation",
            container=False,
            elem_id="input_box",
            lines=2,
            max_lines=30,
        )

    with gr.Row():
        with gr.Column(scale=15):
            user_name = gr.Textbox(
                show_label=False,
                placeholder="Enter your name",
                container=False,
                elem_id="input_box",
            )
        with gr.Column(scale=15):
            user_PID = gr.Textbox(
                show_label=False,
                placeholder="Enter your PID",
                container=False,
                elem_id="input_box",
            )
        with gr.Column(scale=5, min_width=50):
            user_submit_btn = gr.Button(value="Submit report", variant="primary", interactive=False)

    

    if add_promotion_links:
        gr.Markdown(acknowledgment_md)
    
    gr.Markdown(contacts, elem_id="contacts")

    # Register listeners
    # user_answer.change(
    #     activate_user_submit_btn,
    #     [state, user_answer, user_reason, user_name, user_PID],
    #     [user_submit_btn],
    # )
    # user_reason.change(
    #     activate_user_submit_btn,
    #     [state, user_answer, user_reason, user_name, user_PID],
    #     [user_submit_btn],
    # )
    # user_name.change(
    #     activate_user_submit_btn,
    #     [state, user_answer, user_reason, user_name, user_PID],
    #     [user_submit_btn],
    # )
    # user_PID.change(
    #     activate_user_submit_btn,
    #     [state, user_answer, user_reason, user_name, user_PID],
    #     [user_submit_btn],
    # )
    user_submit_btn.click(
        take_user_feedback,
        [state, user_answer, user_reason, user_name, user_PID, model_selector],
        [state, user_submit_btn],
    )

    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector, user_answer, user_reason, user_name, user_PID],
        [textbox, upvote_btn, downvote_btn, flag_btn, user_submit_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector, user_answer, user_reason, user_name, user_PID],
        [textbox, upvote_btn, downvote_btn, flag_btn, user_submit_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector, user_answer, user_reason, user_name, user_PID],
        [textbox, upvote_btn, downvote_btn, flag_btn, user_submit_btn],
    )
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list + [user_submit_btn])

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    textbox.submit(
        add_text, [state, model_selector, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text,
        [state, model_selector, textbox],
        [state, chatbot, textbox] + btn_list,
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    return [state, model_selector]


def build_demo(models):
    with gr.Blocks(
        title="LMTutor for DSC250 Advanced Data Mining",
        theme=gr.themes.Default(),
        css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        state, model_selector = build_single_model_ui(models)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        if args.show_terms_of_use:
            load_js = get_window_url_params_js
        else:
            load_js = get_window_url_params_js

        demo.load(
            load_demo,
            [url_params],
            [
                state,
                model_selector,
            ],
            _js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="Shows term of use before loading the demo",
    )
    parser.add_argument(
        "--add-chatgpt",
        action="store_true",
        help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        help="Add Anthropic's Claude models (claude-2, claude-instant-1)",
    )
    parser.add_argument(
        "--add-palm",
        action="store_true",
        help="Add Google's PaLM model (PaLM 2 for Chat: chat-bison@001)",
    )
    parser.add_argument(
        "--register-openai-compatible-models",
        type=str,
        help="Register custom OpenAI API compatible models by loading them from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate)
    models = get_model_list(
        args.controller_url,
        args.register_openai_compatible_models,
        args.add_chatgpt,
        args.add_claude,
        args.add_palm,
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
    )