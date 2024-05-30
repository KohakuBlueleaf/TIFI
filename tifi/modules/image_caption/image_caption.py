import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union
from subprocess import Popen, PIPE

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import (
    Chat,
    CONV_VISION_LLama2,
    CONV_VISION_Vicuna0,
    Conversation,
    SeparatorStyle,
    StoppingCriteriaSub,
)

from . import llava_utils


@dataclass
class ConfigArguments:
    """The argument data class for the input of class "Config" of minigpt4"""

    cfg_path: str
    options: List[str]


class ImageCaption:
    """An abstract class for all image caption generator"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_caption(self, img: Union[Image.Image, torch.Tensor, str]) -> str:
        """This function takes an image as input,
        return a string that describe the given image.

        Note that if the given img is a string, it is the image path
        """


class BullshitImageCaption(ImageCaption):
    """An "mocked" image caption generator, which might be useful when testing or ablation study"""

    def generate_caption(self, img: Image.Image | torch.Tensor | str) -> str:
        return "This is an image"


class VLMImageCaption(ImageCaption):
    """An abstract class for VLM-based image caption generator.
    It is allowed to load your own model for the VLM
    """

    def __init__(self, model=None) -> None:
        super().__init__()

        # Leave the model blank (None if no model provided)
        self.model = model

    def set_model(self, model):
        if self.model != None:
            warnings.warn(
                "Current model is not None, new model will override the original one",
                UserWarning,
            )

        self.model = model


class LlavaImageCaption(ImageCaption):
    """
    Use another process to dynamicly offload the LLaVA model.
    Also avoid some potential error caused by different CUDA runtimes.
    """

    def __init__(self, model_dir: str) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.captioner_process = None

    def load(self):
        self.captioner_process = Popen(
            ["python", "-m", "tifi.modules.image_caption.llava_utils"],
            stdin=PIPE,
            stdout=PIPE,
        )
        self.captioner_process.stdin.write(self.model_dir.encode() + b"\n")

    def offload(self):
        self.captioner_process.stdin.close()
        self.captioner_process.stdout.close()
        self.captioner_process.kill()
        self.captioner_process.wait()
        self.captioner_process = None

    def generate_caption(self, img: Image.Image | torch.Tensor | str) -> str:
        if self.captioner_process == None:
            self.load()
        if isinstance(img, torch.Tensor):
            img = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        self.captioner_process.stdin.write(
            llava_utils.image_to_base64_data_uri(img).encode() + b"\n"
        )
        self.captioner_process.stdout.readline()
        result = self.captioner_process.stdout.readline()
        # print("LLaVA caption:", result)
        return result.decode().strip()


class MiniGPT4ImageCaption(VLMImageCaption):
    # Conversation dict
    conv_dict = {
        "pretrain_vicuna0": CONV_VISION_Vicuna0,
        "pretrain_llama2": CONV_VISION_LLama2,
    }

    def __init__(
        self,
        gpu_id: int,
        cfg_path: str,
        model_cfg_path: str = None,
        options: List[str] = None,
    ):
        super().__init__(model=None)

        # Create config by the file specified in cfg_path
        self.cfg = Config(ConfigArguments(cfg_path=cfg_path, options=options))

        # Override the model config by the file specified in model_cfg_path
        if model_cfg_path != None:
            self.cfg.config = OmegaConf.merge(
                self.cfg.config, OmegaConf.load(model_cfg_path)
            )

        # Initialize Chat later......
        self.chat = None

        # Get vision processor
        vis_processor_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)

        # Store important values
        self.gpu_id = gpu_id
        self.vis_processor = vis_processor

        # Initialize the llm automatically by the config "cfg"
        self.init_llm()

    def _get_new_chat_state(self) -> Conversation:
        """This method create and return a new chat state (Conversation)"""
        return self.conv_dict[self.cfg.model_cfg.model_type].copy()

    def init_llm(
        self,
    ):
        model_config = self.cfg.model_cfg
        model_config.device_8bit = self.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to("cuda:{}".format(self.gpu_id))

        self.set_model(model)

    def set_model(self, model):
        super().set_model(model)

        # Setting stopping criteria for the chat
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [
            torch.tensor(ids).to(device="cuda:{}".format(self.gpu_id))
            for ids in stop_words_ids
        ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

        # Use the class Chat directly from the minigpt4 repo
        self.chat = Chat(
            self.model,
            self.vis_processor,
            device="cuda:{}".format(self.gpu_id),
            stopping_criteria=stopping_criteria,
        )

    def generate_caption(
        self,
        img: Union[Image.Image, torch.Tensor, str],
        user_message: str = "Take a look at this image and describe what you notice.",
        num_beams=1,
        temperature=1,
        max_new_tokens=300,
        max_length: int = 2000,
    ) -> str:
        if self.chat == None:
            raise Exception("Please set the model first")

        # Upload the image first
        chat_state = self._get_new_chat_state()

        # A container that contains all image that have been uploaded and processed
        img_list = []
        llm_message = self.chat.upload_img(img, chat_state, img_list)
        self.chat.encode_img(img_list)

        # Ask the model to generate caption
        if len(user_message) == 0:
            raise Exception("Any non-empty prompt is required!")

        self.chat.ask(user_message, chat_state)

        llm_message = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )[0]
        return llm_message
