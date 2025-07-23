import os
import torch
import copy
import numpy as np

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key in ['copy', 'from_dict', 'replace', 'to_dict', 'class_type', 'func']:
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        if element is not None or (not isinstance(element, dict)):
            if isinstance(element, np.ndarray):
                element = element.tolist()
            result[key] = element
    return result


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if 'wandb' in runs: runs.remove('wandb')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        if len(models) == 0:
            raise ValueError("No models in this directory: " + load_run)
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def export_policy_as_jit(policy_latent, action, path, normalizer=None, filename="policy.pt"):
    policy_exporter = TorchPolicyExporter(policy_latent, action, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(policy_latent, action, path, normalizer=None, filename="policy.onnx"):
    policy_exporter = OnnxPolicyExporter(policy_latent, action, normalizer)
    policy_exporter.export(path, filename)


class TorchPolicyExporter(torch.nn.Module):
    def __init__(self, policy_latent, action, normalizer=None):
        super().__init__()
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.policy_latent = copy.deepcopy(policy_latent)
        self.action = copy.deepcopy(action)

    def forward(self, x):
        return self.action(self.policy_latent(self.normalizer(x)))

    @torch.jit.export
    def reset(self):
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class OnnxPolicyExporter(torch.nn.Module):
    def __init__(self, policy_latent, action, normalizer=None):
        super().__init__()
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.policy_latent = copy.deepcopy(policy_latent)
        self.action = copy.deepcopy(action)

    def forward(self, x):
        return self.action(self.policy_latent(self.normalizer(x)))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.policy_latent[0].in_features)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
