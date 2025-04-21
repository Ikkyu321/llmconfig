import argparse
import json

import gradio as gr
import requests
import os

username_list_dict = {}


# def login(username, password):
#     if username_list_dict.get(username, None) is None:
#         username_list_dict[username] = password
#         return True
#     return False

def add_text(history, text):
    print(history)
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def get_template(template_name):
    try:
        response = requests.get("http://127.0.0.1:8001/prompt/templates?templateName=" + template_name)
        json_data = json.loads(response.text)
    except Exception as e:
        gr.Info(f"get template name error, info is {e}")
    return json_data["data"]


def get_template_multi(template_name):
    try:
        response = requests.get("http://127.0.0.1:8001/prompt/templates?templateName=" + template_name)
        json_data = json.loads(response.text)
    except Exception as e:
        gr.Info(f"get template name error, info is {e}")
    return json_data["data"], update_drop_down()


def get_template_names():
    try:
        response = requests.get("http://127.0.0.1:8001/prompt/templates/names")
        json_data = json.loads(response.text)
    except Exception as e:
        gr.Info(f"get template names error, info is {e}")
    return json_data["data"]


def get_first_name():
    try:
        response = requests.get("http://127.0.0.1:8001/prompt/templates/names")
        json_data = json.loads(response.text)
    except Exception as e:
        gr.Info(f"get template first name error, info is {e}")
    return json_data["data"][0]


def delete_template(template_name):
    if len(get_template_names()) == 1:
        gr.Info("can not delete this template, size is 1")
        return update_drop_down()
    try:
        response = requests.delete("http://127.0.0.1:8001/prompt/templates?templateName=" + template_name)
        json_data = json.loads(response.text)
    except Exception as e:
        gr.Info(f"delete template error, info is {e}")

    if json_data["code"] == "200":
        gr.Info("delete template success")
    else:
        gr.Error(f"delete failed, info is {json_data['detail']}")
    return update_drop_down()


def add_template(template_name, template_content):
    try:
        pload = {
            "data": json.loads(template_content)
        }
    except Exception as e:
        gr.Info(f"add failed, info is {e}")
        return update_drop_down()
    path_param = {"promptTemplateName": template_name}
    response = requests.post(f"http://127.0.0.1:8001/prompt/templates", json=pload, params=path_param)
    json_data = json.loads(response.text)
    if json_data["code"] == "200":
        gr.Info("add template success")
    else:
        gr.Info(f"add failed, info is {json_data}")
    return update_drop_down()


def update_drop_down():
    choices = get_template_names()
    fake_choices = []
    for choice in choices:
        fake_choices.append(choice)
    print(fake_choices)
    return gr.update(choices=fake_choices, value=fake_choices[0])


def get_drop_down():
    choices = get_template_names()
    fake_choices = []
    for choice in choices:
        fake_choices.append(choice)
    print(fake_choices)
    return fake_choices


def clear_bot(history):
    history = []
    return history


def bot(history, no_stream_chat, prompt_header="You are a helpful assistant.", prompt_template="",
        maxContentRound: int = 0, maxLength: int = 800, maxWindowSize: int = 800,
        temperature: float = 0.0, best_of: int = 0,
        presence_penalty: float = 0.0, frequency_penalty: float = 0.0, repetition_penalty = 0.0,
        top_p: float = 1.0, top_k: int = -1,
        length_penalty: float = 1.0
        ):
    print("\n\n")
    print(prompt_template)
    print(history)
    headers = {"User-Agent": "vLLM Client"}

    if best_of == 0:
        best_of = None

    real_history = []
    if maxContentRound > 0:
        if len(history) > 0:
            real_history = history[-(maxContentRound + 1):-1]

    template_name = prompt_template["templateName"]

    service_pload = {

        "system": prompt_header,
        "promptTemplateName": template_name,
        "maxWindowSize": maxWindowSize,
        "maxContentRound": maxContentRound,
        "maxOutputLength": maxLength,
        "draw": False,
        "stream": True,
        "useTool": False,
        "summarize": False,
        "generateStyle": "chat",        
    }

    model_pload = {

        "best_of": best_of,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "use_beam_search": False,
        "length_penalty": length_penalty
    }

    pload = {

        "input": history[-1][0],
        "history": real_history,
        "serviceParams": service_pload,
        "modelParams": model_pload
    }

    history[-1][1] = ""
    if no_stream_chat is True:
        try:
            pload["serviceParams"]["stream"] = False
            print(f'input data is {pload}')
            response = requests.post("http://127.0.0.1:8001/llm/generate",
                                     headers=headers,
                                     json=pload)
        except Exception as e:
            gr.Info(f"chat to server error, info is {e}")
            yield history

        if response.status_code == 200:
            data = json.loads(response.text)
            output = data["data"]["output"]
            history[-1][1] = output
            yield history
        else:
            if response.text:
                gr.Info((f'post error, error is {json.loads(response.text)}'))
            else:
                gr.Info((f'post error, error is {response}'))
            yield history

    else:
        try:
            response = requests.post("http://127.0.0.1:8001/llm/generate",
                                     headers=headers,
                                     json=pload,
                                     stream=True)
        except Exception as e:
            gr.Info(f"chat to server error, info is {e}")
            yield history

        if response.status_code == 200:
            for chunk in response.iter_lines(chunk_size=8192,
                                             decode_unicode=False,
                                             delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))
                    output = data["output"]
                    history[-1][1] = output
                    yield history
        else:
            if response.text:
                gr.Info(f'post error, error is {json.loads(response.text)}')
            else:
                gr.Info(f'post error, error is {response}')
            yield history


with gr.Blocks() as demo:
    gr.Markdown("# llmConfig vllm inference\n")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                prompt_header = gr.Textbox(placeholder="Enter the system prompt what's you needed",
                                           label="系统提示头(System Prompt)", scale=90)
                no_stream_chat = gr.Checkbox(label="非流式对话",
                                             elem_classes=["display:flex", "justify-content: center",
                                                           "align-items:center"], scale=10)
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                label="聊天机器人",
                bubble_full_width=False,
                avatar_images=(None, (os.path.join(os.path.abspath(''), "avatar.png"))),
                scale=20
            )

            with gr.Row():
                txt = gr.Textbox(
                    scale=10,
                    show_label=True,
                    placeholder="Enter text and press enter",
                    container=False,
                    label="输入你想问的"
                )

                clear = gr.Button(scale=1, size="sm", value="清除记忆")

            clear.click(clear_bot, inputs=chatbot, outputs=chatbot)

        with gr.Column():
            current_prompt_template = gr.JSON(label="提示模板(Prompt Template)")

            dropdown = gr.Dropdown(choices=get_drop_down(),
                                   value="qwen_default",
                                   label="模板名称(Prompt Name)")

            dropdown.select(get_template, inputs=dropdown, outputs=current_prompt_template)
            # dropdown.select(get_template_multi, inputs=dropdown, outputs=[current_prompt_template, dropdown])
            # dropdown.focus(get_template_multi, inputs=dropdown, outputs=[current_prompt_template, dropdown])
            # dropdown.focus(update_drop_down, outputs=dropdown)

            current_prompt_template.value = get_template(dropdown.value)

            with gr.Column():
                with gr.Row():
                    new_prompt_name = gr.Textbox(placeholder="Template name",
                                                 label="Template Name", type="text",
                                                 scale=20)
                    new_prompt_template = gr.Textbox(placeholder="Enter the prompt template body",
                                                     label="Prompt Template Body", type="text",
                                                     scale=80)

                with gr.Row():
                    btn_add = gr.Button(value="添加模板")
                    btn_delete = gr.Button(value="删除模板")
                    btn_refresh = gr.Button(value="Refresh")
                    btn_add.click(add_template, inputs=[new_prompt_name, new_prompt_template], outputs=dropdown)
                    btn_delete.click(delete_template, inputs=dropdown, outputs=dropdown)
                    btn_refresh.click(update_drop_down, outputs=dropdown)

            max_round = gr.Slider(minimum=0, step=1, maximum=10, value=0, label="maxContentRound")
            length = gr.Slider(minimum=0, maximum=4000, value=500, label="maxLength")
            window_size = gr.Slider(
                minimum=0, maximum=6144, step=1, value=2000, label="maxWindowSize"
            )

            with gr.Row():
                ## 只有这里设置和vllm稍微有点区别，这里是0.0
                temperature = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.0, label="temperature")
                best_of = gr.Slider(minimum=0, maximum=10, step=1, value=0, label="best_of")
            with gr.Row():
                presence_penalty = gr.Slider(minimum=-2.0, maximum=2.0, step=0.1, value=0.0, label="presence_penalty")
                frequency_penalty = gr.Slider(minimum=-2.0, maximum=2.0, step=0.1, value=0.0, label="frequency_penalty")
                repetition_penalty = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="repetition_penalty")
            with gr.Row():
                top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=1.0, label="top_p")
                top_k = gr.Slider(minimum=-1, maximum=100, step=1, value=-1, label="top_k")
                length_penalty = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="length_penalty")
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, inputs=[chatbot, no_stream_chat, prompt_header, current_prompt_template, max_round, length, window_size,
                     temperature, best_of,
                     presence_penalty, frequency_penalty, repetition_penalty,
                     top_p, top_k,
                     length_penalty],
        outputs=chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-url",
                        type=str,
                        default="http://127.0.0.1:8001/llm/generate")
    args = parser.parse_args()
    root_path = os.environ.get('DEMO_ROOT_PATH', '')
    demo.queue(concurrency_count=100).launch(server_name=args.host,
                                             server_port=args.port,
                                             share=True,
                                             root_path=root_path)
