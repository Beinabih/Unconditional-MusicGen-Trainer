from audiocraft.models import MusicGen
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
import torch.nn.functional as F
import typing as tp
import random

from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout


import os
from tqdm.notebook import tqdm

from dataloader import create_dataloaders
from model_loader import load_model


def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans


def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)

    return result


def get_condition_tensor(model, attributes) -> torch.Tensor:
    null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(attributes)
    conditions = attributes + null_conditions
    tokenized = model.lm.condition_provider.tokenize(conditions)
    cfg_conditions = model.lm.condition_provider(tokenized)
    return cfg_conditions


def train(
    model,
    dataloader,
    optimizer,
    criterion,
    model_name,
    scaler,
    use_wandb,
    run,
    current_step,
    scheduler,
    config
):
    model.lm.train()
    for i, batch in enumerate(dataloader):
        

        audio = batch.cuda()
        with torch.no_grad():
            codes, _ = model.compression_model.encode(audio)

        if codes.shape[0] == 1:
            prompt = [model_name]
        else:
            prompt = [model_name] * (codes.shape[0] // 2)

        if config["cross_attention"]:
            attributes, _ = model._prepare_tokens_and_attributes(prompt, None)
            condition_tensors = get_condition_tensor(model, attributes)
        else:
            condition_tensors = None

        # for batchsize 1
        if codes.shape[0] == 1:
            codes = torch.cat([codes, codes], axis=0)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            lm_output = model.lm.compute_predictions(
                codes=codes, conditions=[], condition_tensors=condition_tensors
            )

            codes = codes
            logits = lm_output.logits
            mask = lm_output.mask
            codes = F.one_hot(codes, 2048).type(logits.dtype)
            codes = codes.cuda()
            logits = logits.cuda()
            mask = mask.cuda()
            mask = mask.reshape(-1)
            masked_logits = logits.reshape(-1, 2048)[mask]
            masked_codes = codes.reshape(-1, 2048)[mask]
            loss = criterion(masked_logits, masked_codes)

        assert count_nans(masked_logits) == 0
        scaler.scale(loss).backward()


        if ((i + 1) % config["accum_iter"] == 0) or (i + 1 == len(dataloader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)
            scaler.step(optimizer)
            if scheduler:
                scheduler.step(current_step / len(dataloader))
            scaler.update()
            optimizer.zero_grad()
            if use_wandb:
                run.log({"loss train": loss.item()}, step=current_step)
                
        current_step += 1

    return current_step


def evaluate(model, dataloader, optimizer, criterion, model_name, use_wandb, run, config):
    epoch_loss = 0
    model.lm.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            audio = batch.cuda()
            codes, _ = model.compression_model.encode(audio)

            if codes.shape[0] == 1:
                prompt = [model_name]
            else:
                prompt = [model_name] * (codes.shape[0] // 2)

            # attributes, _ = model._prepare_tokens_and_attributes(prompt, None)
            # condition_tensors = get_condition_tensor(model, attributes)

            if config["cross_attention"]:
                attributes, _ = model._prepare_tokens_and_attributes(prompt, None)
                condition_tensors = get_condition_tensor(model, attributes)
            else:
                condition_tensors = None

            # for batchsize 1
            if codes.shape[0] == 1:
                codes = torch.cat([codes, codes], axis=0)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output = model.lm.compute_predictions(
                    codes=codes, conditions=[], condition_tensors=condition_tensors
                )

                codes = codes
                logits = lm_output.logits
                mask = lm_output.mask

                
                codes = F.one_hot(codes, 2048).type(logits.dtype)
                codes = codes.cuda()
                logits = logits.cuda()
                mask = mask.cuda()
                mask = mask.reshape(-1)
                masked_logits = logits.reshape(-1, 2048)[mask]
                masked_codes = codes.reshape(-1, 2048)[mask]
                loss = criterion(masked_logits, masked_codes)

            epoch_loss += loss.item()
            assert count_nans(masked_logits) == 0

            # tqdm.write(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(dataset)}, Loss: {loss.item()}")

    return epoch_loss / len(dataloader)


def main(
    model_name: str,
    config,
    dataset_cfg: dict,
    project_name,
):
    torch.backends.cudnn.deterministic = True
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = None
    if config["use_wandb"]:
        import wandb

        run = wandb.init(
            project=project_name,
            config={
                "learning_rate": config["learning_rate"],
                "dataset": model_name,
                "epochs": config["epochs"],
                "seed": config["seed"],
            },
        )

    dataloader_train, dataloader_eval = create_dataloaders(dataset_cfg)
    model = load_model(config, model_name)
    model.lm = model.lm.to(torch.float32)  # important

    scaler = torch.cuda.amp.GradScaler()
    optimizer = AdamW(
        model.lm.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    if config['CosineAnnealing']:
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config["lr_steps"])
    else:
        scheduler = None
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
    #                                              verbose=True)
    criterion = nn.CrossEntropyLoss()

    save_path = "models/"
    os.makedirs(save_path, exist_ok=True)

    best_loss = 10
    current_step = 0
    for epoch in tqdm(range(config["epochs"])):
        current_step = train(
            model,
            dataloader_train,
            optimizer,
            criterion,
            model_name,
            scaler,
            config["use_wandb"],
            run,
            current_step, 
            scheduler, 
            config
        )
        valid_loss = evaluate(
            model, dataloader_eval, optimizer, criterion, model_name, config["use_wandb"], run, config
        )

        if config["use_wandb"]:
            run.log({"loss eval": valid_loss}, step=current_step)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.lm.state_dict(), f"{save_path}/lm_{model_name}_final.pt")

    torch.save(model.lm.state_dict(), f"{save_path}/lm_{model_name}_end.pt")
    if config["use_wandb"]:
        wandb.finish()
