import numpy as np

import torch
from tqdm import tqdm

from PIL import Image
from diffusers import LMSDiscreteScheduler
from k_diffusion.external import DiscreteEpsDDPMDenoiser
from k_diffusion.sampling import sample_euler, get_sigmas_exponential

from library.train_util import replace_unet_modules
from library.sdxl_model_util import load_models_from_sdxl_checkpoint
from library.sdxl_train_util import get_size_embeddings

from lycoris import LycorisNetwork, create_lycoris_from_weights
from tifi.modules.animatediff.loader import load, inject
from tifi.utils.interpolation import blend_frame_optical_flow
from tifi.utils.color_fixing import match_color
from tifi.utils.jigsaw import jigsaw_schedule
from tifi.utils.diff_utils import load_tokenizers, encode_prompts_single
from tifi.logging import logger
from tifi.modules.image_caption.image_caption import MiniGPT4ImageCaption


class TemporalInpainting:
    FRAME_SIZE = 7
    MIN_FRAME_SIZE = 3

    def __init__(
        self,
        model_file: str = "./models/sdxl-1.0.safetensors",
        motion_module_file: str = "./models/mm_sdxl_hs.safetensors",
        lycoris_model_file: str = None,
    ):
        ## Load Model
        tokenizer, tokenizer_2 = load_tokenizers()
        text_model1, text_model2, vae, unet, _, _ = load_models_from_sdxl_checkpoint(
            "",
            model_file,
            "cpu",
            torch.float32,
        )
        unet.bfloat16()
        unet.requires_grad_(False)
        unet.cuda()
        # unet.enable_gradient_checkpointing()
        replace_unet_modules(unet, False, True, False)
        mm = load(motion_module_file)
        inject(unet, mm)

        if lycoris_model_file is not None:
            lycoris_network: LycorisNetwork = create_lycoris_from_weights(
                1.0, lycoris_model_file, unet
            )
            lycoris_network.cuda()
            lycoris_network.apply_to()
            lycoris_network.merge_to()
            lycoris_network.restore()
            lycoris_network.cpu()
            del lycoris_network
        torch.cuda.empty_cache()

        scheduler: LMSDiscreteScheduler = LMSDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
        )
        denoiser: DiscreteEpsDDPMDenoiser = DiscreteEpsDDPMDenoiser(
            unet, scheduler.alphas_cumprod, False
        ).cuda()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_model1 = text_model1
        self.text_model2 = text_model2
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.denoiser = denoiser
        
        ## MiniGPT4 Image captioner
        self.image_captioner = MiniGPT4ImageCaption(
            gpu_id=0,
            cfg_path='../../config/minigpt4_llama2_eval.yaml',
            model_cfg_path='../../config/minigpt4_llama2.yaml',
        )

    def sigmas(
        self,
        steps=16,
        sigma_schedule_function=get_sigmas_exponential,
        **kwargs,
    ):
        sigma_min = self.scheduler.sigmas[-2]
        sigma_max = self.scheduler.sigmas[0]
        sigmas = sigma_schedule_function(steps + 1, sigma_min, sigma_max, **kwargs)
        return torch.cat([sigmas[:-2], self.scheduler.sigmas.new_zeros([1])])

    def cfg_wrapper(
        self, prompts, pooleds, neg_prompt, neg_pooled, reference_frames, cfg=5.0
    ):
        jigsaw = None

        def denoise(x, t):
            nonlocal jigsaw, prompts, neg_prompt, pooleds, neg_pooled
            b, f, c, h, w = x.shape

            if jigsaw is None:
                jigsaw = jigsaw_schedule(f, self.FRAME_SIZE, self.MIN_FRAME_SIZE)

            size_emb = (
                get_size_embeddings(
                    torch.FloatTensor([[h * 8, w * 8]]),
                    torch.FloatTensor([[0, 0]]),
                    torch.FloatTensor([[h * 8, w * 8]]),
                    "cuda",
                )
                .float()
                .to(x.device)
            )

            all_pred = []
            for jigsaw_id, slices in jigsaw:
                for idx, part in tqdm(list(enumerate(slices)), desc="JigSaw Sampling", leave=False):
                    current_x = x[:, part]
                    current_t = t

                    b, f, c, h, w = current_x.shape
                    current_batch_size = b * f

                    prompt = prompts[jigsaw_id][idx]
                    pooled = pooleds[jigsaw_id][idx]
                    prompt = prompt.repeat(1, f, 1, 1).flatten(0, 1)
                    pooled = pooled.repeat(1, f, 1).flatten(0, 1)

                    s_emb = size_emb.repeat(current_batch_size, 1)
                    embed = torch.cat([pooled, s_emb], dim=-1)
                    neg_embed = torch.cat(
                        [neg_pooled.repeat(current_batch_size, 1), s_emb], dim=-1
                    )

                    cond = self.denoiser(
                        current_x,
                        current_t,
                        context=prompt,
                        y=embed,
                    )
                    uncond = self.denoiser(
                        current_x,
                        current_t,
                        context=neg_prompt.repeat(current_batch_size, 1, 1),
                        y=neg_embed,
                    )
                    all_pred.append(cfg * (cond - uncond) + uncond)
                break
            pred = torch.cat(all_pred, dim=1)
            for batch_idx in range(b):
                for frame_idx in range(f):
                    if reference_frames[batch_idx][frame_idx] is not None:
                        pred[batch_idx, frame_idx] = reference_frames[batch_idx][
                            frame_idx
                        ].to(pred)
            return pred

        return denoise

    def vae_encode(self, frame: torch.Tensor):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            frame = frame.unsqueeze(0).cuda() * 2 - 1
            frame_latent = self.vae.encode(frame).latent_dist.mode()
        return frame_latent[0] * self.vae.config.scaling_factor

    def vae_decode(self, frame: torch.Tensor):
        scale = 1 / self.vae.config.scaling_factor
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            frame = frame.unsqueeze(0).cuda() * scale
            frame_img = (self.vae.decode(frame).sample) / 2 + 0.5
        return frame_img[0].float().clamp(0, 1).cpu()

    def caption(self, image):
        # return "test"
        
        # MiniGPT4 image caption can take a torch.Tensor as input and (in [0, 1] and C, H, W)
        # See minigpt4.processors.Blip2ImageEvalProcessor for detail
        return self.image_captioner.generate_caption(image)

    def image_to_prompt_embeds(self, frames: list[torch.Tensor]):
        prompts = []
        embeds = []
        for frame in frames:
            caption = self.caption(frame)
            prompt, embed = encode_prompts_single(
                self.tokenizer,
                self.tokenizer_2,
                self.text_model1,
                self.text_model2,
                caption,
            )
            prompts.append(prompt)
            embeds.append(embed)
        return torch.stack(prompts), torch.stack(embeds)

    def __call__(self, videos, cfg=5.0):
        if not isinstance(videos[0], list):
            videos = [videos]

        logger.info("Starting Temporal Inpainting")

        logger.info("Preparing latents of interpolated videos...")
        video_latents = [[None for _ in video] for video in videos]
        reference_videos = []
        videos_fill1 = []
        videos_fill2 = []
        self.vae.cuda()
        for v_idx, video in enumerate(videos):
            assert video[0] is not None
            assert video[-1] is not None
            v1 = []
            v2 = []
            prev1_idx = 0
            prev1 = video[0]
            prev2_idx = len(video) - 1
            prev2 = video[-1]
            for idx, frame in enumerate(video):
                if frame is not None:
                    prev1 = frame
                    prev1_idx = idx
                v1.append((prev1, prev1_idx))
            for idx, frame in list(enumerate(video))[::-1]:
                if frame is not None:
                    prev2 = frame
                    prev2_idx = idx
                v2.append((prev2, prev2_idx))
            v2 = v2[::-1]
            videos_fill1.append(v1)
            videos_fill2.append(v2)

            ref_video = []
            for idx, ((ref1, ref1_idx), (ref2, ref2_idx)) in enumerate(zip(v1, v2)):
                if ref1_idx == ref2_idx or idx == ref1_idx or idx == ref2_idx:
                    latent = self.vae_encode(ref1)
                    ref_video.append(latent)
                    video_latents[v_idx][idx] = latent
                    continue
                else:
                    ref_video.append(None)
                    interpolated_frame = blend_frame_optical_flow(
                        ref1, ref2, ref2_idx - ref1_idx - 1
                    )[idx - ref1_idx - 1]
                    video[idx] = interpolated_frame
                    video_latents[v_idx][idx] = self.vae_encode(interpolated_frame)
            reference_videos.append(ref_video)
        video_latents = torch.stack([torch.stack(vid) for vid in video_latents])
        self.vae.cpu()
        logger.info("Video latents prepared")

        logger.info("Preparing prompts for each group")
        self.text_model1.cuda()
        self.text_model2.cuda()
        jigsaw = jigsaw_schedule(
            len(videos_fill1[0]), self.FRAME_SIZE, self.MIN_FRAME_SIZE
        )
        prompts_list1 = []
        pooled_list1 = []
        for sl in next(jigsaw)[1]:
            reference_frames = [p[sl][0][0] for p in videos_fill1]
            prompts, embeds = self.image_to_prompt_embeds(reference_frames)
            prompts_list1.append(prompts)
            pooled_list1.append(embeds)
        prompts_list2 = []
        pooled_list2 = []
        for sl in next(jigsaw)[1]:
            reference_frames = [p[sl][0][0] for p in videos_fill2]
            prompts, embeds = self.image_to_prompt_embeds(reference_frames)
            prompts_list2.append(prompts)
            pooled_list2.append(embeds)
        prompt = [prompts_list1, prompts_list2]
        pooled = [pooled_list1, pooled_list2]
        neg_prompt, neg_pooled = encode_prompts_single(
            self.tokenizer,
            self.tokenizer_2,
            self.text_model1,
            self.text_model2,
            "neg test",
        )
        self.text_model1.cpu()
        self.text_model2.cpu()
        logger.info("All prompts prepared")

        logger.info("Temporal Inpainting ...")
        denoise_func = self.cfg_wrapper(
            prompt, pooled, neg_prompt, neg_pooled, reference_videos, cfg
        )
        sigma_schedule = self.sigmas(16).cuda()
        sigma_schedule_inpaint = sigma_schedule[-9:]
        init_x = torch.randn_like(video_latents) * sigma_schedule_inpaint[0]
        init_x += video_latents

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            result = sample_euler(
                denoise_func,
                init_x,
                sigma_schedule_inpaint,
            )

        logger.info("Decode generated latents")
        self.vae.cuda()
        vids = []
        vids_tensor = []
        for video, inp_video in zip(result, videos):
            vid = []
            vid_tensor = []
            for frame, inp_frame in zip(video, inp_video):
                decoded_frame = self.vae_decode(frame)
                vid_tensor.append(match_color(decoded_frame, inp_frame))
                vid.append(
                    Image.fromarray(
                        (decoded_frame.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                    )
                )
            vids.append(vid)
            vids_tensor.append(vid_tensor)
        self.vae.cpu()
        logger.info("All done.")
        return vids
