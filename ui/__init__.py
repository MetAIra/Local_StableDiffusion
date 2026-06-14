"""UIモジュール"""
from .common import create_prompt_inputs
from .txt2img_tab import create_txt2img_tab
from .img2img_tab import create_img2img_tab
from .inpaint_tab import create_inpaint_tab
from .outpaint_tab import create_outpaint_tab
from .controlnet_tab import create_controlnet_tab
from .upscale_tab import create_upscale_tab
from .multiview_tab import create_multiview_tab

__all__ = [
    'create_prompt_inputs',
    'create_txt2img_tab',
    'create_img2img_tab',
    'create_inpaint_tab',
    'create_outpaint_tab',
    'create_controlnet_tab',
    'create_upscale_tab',
    'create_multiview_tab'
]
