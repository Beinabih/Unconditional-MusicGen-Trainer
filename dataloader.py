from audiocraft.data.audio_dataset import AudioDataset
from torch.utils.data import DataLoader
import typing as tp

def create_dataloaders(dataset_cfg:tp.Dict):
    """creates two dataloaders with the AudioCraft AudioDataset

    Args:
        dataset_cfg (dict): parameters

    Returns:
        tuple(Dataloader, Dataloader): 
    """
    dataset_train = AudioDataset.from_path(
        dataset_cfg["dataset_path_train"],
        minimal_meta=True,
        segment_duration=dataset_cfg["segment_duration"],
        num_samples=dataset_cfg["num_examples_train"],
        sample_rate=dataset_cfg["sample_rate"],
        channels=1,
        shuffle=dataset_cfg["shuffle"],
        return_info=dataset_cfg["return_info"],
    )
    
    dataset_eval = AudioDataset.from_path(
        dataset_cfg["dataset_path_eval"],
        minimal_meta=True,
        segment_duration=dataset_cfg["segment_duration"],
        num_samples=dataset_cfg["num_examples_eval"],
        sample_rate=dataset_cfg["sample_rate"],
        channels=1,
        shuffle=dataset_cfg["shuffle"],
        return_info=dataset_cfg["return_info"],
    )

    dataloader_train = DataLoader(
            dataset_train,
            batch_size=dataset_cfg["batch_size"],
            collate_fn=dataset_train.collater,
            num_workers=4
        )
    
    dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=dataset_cfg["batch_size"],
            collate_fn=dataset_eval.collater,
            num_workers=4
        )
    
    return dataloader_train, dataloader_eval