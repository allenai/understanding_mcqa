from typing import Optional

from hf_olmo.configuration_olmo import OLMoConfig
from hf_olmo.modeling_olmo import OLMoForCausalLM, create_model_config_from_pretrained_config

from model.modified_olmo_model import ModifiedOLMo


class ModifiedOLMoForCausalLM(OLMoForCausalLM):
    """
    Huggingface wrapper which uses our modified model classes and functions.
    This code is otherwise the same as the implementation in hf_olmo v0.4.0 (https://github.com/allenai/OLMo/blob/v0.4.0/hf_olmo/modeling_olmo.py)
    """

    config_class = OLMoConfig
    base_model_prefix = "model"
    _no_split_modules = ["ModifiedOLMoBlock"]

    def __init__(
        self, config: OLMoConfig, model: Optional[ModifiedOLMo] = None, init_params: bool = False
    ):
        super().__init__(config, model, init_params)

        # replace model with one of type ModifiedOLMo
        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            model_config.init_device = "cpu"
            self.model = ModifiedOLMo(model_config, init_params=init_params)
        else:
            self.model = model
