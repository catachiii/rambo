import os
import uuid

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    def __init__(self, log_dir, cfg, flush_secs, project, group, offline_mode):
        super().__init__(log_dir, flush_secs)

        # self.log_path = os.path.dirname(os.path.abspath(log_dir))
        self.log_path = log_dir
        self.project = project
        self.offline_mode = offline_mode
        run_name = os.path.basename(os.path.abspath(log_dir))
        self.entity = os.environ["WANDB_USERNAME"]
        if self.offline_mode:
            os.environ["WANDB_MODE"] = "offline"
        convert_slices_to_serializable(cfg)
        self.id = run_name + str(uuid.uuid4())
        wandb.init(project=self.project,
                   entity=self.entity,
                   name=run_name,
                   id=self.id,
                   config=cfg,
                   dir=self.log_path)

        wandb.define_metric("video_step")
        wandb.define_metric("video", step_metric="video_step")
        self.log_dict = {}

    def add_scalar(self, tag, scalar_value, global_step=None):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
        )
        self.log_dict[tag] = scalar_value

    def flush_logger(self, global_step):
        self.log_dict["video_step"] = global_step
        wandb.log(self.log_dict, step=global_step)
        self.log_dict = {}

    def stop(self):
        # video folder
        video_folder = os.path.join(self.log_path, "videos")
        all_files = list_sorted_mp4_files(video_folder)
        for file in all_files:
            if file.endswith(".mp4"):
                step_id = int((file.split(".")[0]).split("-")[-1])
                video_path = os.path.join(video_folder, file)
                wandb.log({"video": wandb.Video(video_path, format="mp4"), "video_step": step_id})
                print(f"Video logged to wandb: {video_path} with step {step_id}")

        wandb.finish()
        if self.offline_mode:
            os.system(
                f"wandb sync {self.log_path}/wandb/latest-run --id {self.id} --project {self.project} --entity {self.entity}")


def convert_slices_to_serializable(obj, visited=None):
    """
    Recursively replaces `slice` objects with a JSON-serializable dictionary format.
    This works for nested dicts, lists, sets, tuples, and object attributes.

    Parameters:
    - obj: Any Python object.
    - visited: A set of visited object ids to avoid infinite recursion.

    Returns:
    - The modified object with `slice` objects replaced by serializable dicts.
    """
    if visited is None:
        visited = set()

    if id(obj) in visited:
        return obj
    visited.add(id(obj))

    if isinstance(obj, slice):
        return {
            "__slice__": True,
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step
        }
    elif isinstance(obj, dict):
        return {convert_slices_to_serializable(k, visited): convert_slices_to_serializable(v, visited) for k, v in
                obj.items()}
    elif isinstance(obj, list):
        return [convert_slices_to_serializable(item, visited) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_slices_to_serializable(item, visited) for item in obj)
    elif isinstance(obj, set):
        return {convert_slices_to_serializable(item, visited) for item in obj}
    elif hasattr(obj, '__dict__') or hasattr(obj, '__slots__'):
        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                setattr(obj, key, convert_slices_to_serializable(value, visited))
        if hasattr(obj, '__slots__'):
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    setattr(obj, slot, convert_slices_to_serializable(getattr(obj, slot), visited))
        return obj
    else:
        return obj


def list_sorted_mp4_files(folder_path):
    """
    List all .mp4 files in the folder ending with '-n.mp4',
    sorted by the numeric value n in ascending order.
    """
    mp4_files = []

    for file in os.listdir(folder_path):
        if file.endswith(".mp4") and '-' in file:
            base = file[:-4]  # remove ".mp4"
            parts = base.rsplit('-', 1)  # split on last '-'
            if len(parts) == 2 and parts[1].isdigit():
                num = int(parts[1])
                mp4_files.append((file, num))

    # Sort by the numeric part
    mp4_files.sort(key=lambda x: x[1])

    return [file[0] for file in mp4_files]
