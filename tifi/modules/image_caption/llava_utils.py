import os
import base64
from io import BytesIO

from PIL import Image

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llama3VisionAlpha


llm = None
chat_handler = None


def load_model(model_path="./model"):
    global llm, chat_handler
    chat_handler = Llama3VisionAlpha(
        clip_model_path=os.path.join(model_path, "mmproj-model-f16.gguf"),
        verbose=False,
    )
    llm = Llama(
        model_path=os.path.join(model_path, "llama3-llava-next-8b-Q6_K.gguf"),
        chat_handler=chat_handler,
        n_ctx=8192,
        n_gpu_layers=999,
        verbose=False,
    )


def image_to_base64_data_uri(pilimg):
    file = BytesIO()
    pilimg.save(file, format="PNG")
    file.seek(0)
    base64_data = base64.b64encode(file.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


def caption_llava_pil(pilimg: Image):
    result = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the content of this image briefly in 3 sentences.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_base64_data_uri(pilimg)},
                    },
                ],
            }
        ]
    )
    return result["choices"][0]["message"]["content"]


def caption_llava(img_file):
    file = os.path.abspath(img_file)
    file_path = file.replace("\\", "/")
    result = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the content of this image briefly in 3 sentences.",
                    },
                    {"type": "image_url", "image_url": {"url": f"file:///{file_path}"}},
                ],
            }
        ]
    )
    return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    load_model(input())
    while True:
        try:
            img_bytes = input()
        except EOFError:
            break
        result = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the content of this image briefly in 3 sentences.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": img_bytes},
                        },
                    ],
                }
            ]
        )
        with open("result.txt", "w") as f:
            f.write(result["choices"][0]["message"]["content"])
        print(result["choices"][0]["message"]["content"])
