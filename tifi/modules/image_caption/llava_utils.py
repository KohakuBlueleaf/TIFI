import os

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llama3VisionAlpha


llm = None
chat_handler = None

def load_model(model_path=r"G:\nn_app\llama.cpp\models\llama3-llava-next-8b-gguf"):
    global llm, chat_handler
    chat_handler = Llama3VisionAlpha(
        clip_model_path=os.path.join(model_path, "mmproj-model-f16.gguf"),
        verbose=False,
    )
    llm = Llama(
        model_path=os.path.join(model_path, "llama3-llava-next-8b-Q8_0.gguf"),
        chat_handler=chat_handler,
        n_ctx=8192,
        n_gpu_layers=999,
        verbose=False,
    )


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
