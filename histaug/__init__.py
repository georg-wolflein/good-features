from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)  # ensure that eval is available in the config file
OmegaConf.register_new_resolver("model_name", lambda model: str(model).split(".")[-1] if "." in str(model) else model)
