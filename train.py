import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.nn import functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from mappers import MemoryBank, DualMemoryBank



def Trainer(model, train_loader, valid_loader, args, llm_emb_size):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if args.iters_to_accumulate > 1:
        accelerator = Accelerator(gradient_accumulation_steps=args.iters_to_accumulate, kwargs_handlers=[kwargs])
    else:
        accelerator = Accelerator(kwargs_handlers=[kwargs])

    device = accelerator.device
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_loader),)
    ## introduce all components to accelerate library
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    valid_loader = accelerator.prepare(valid_loader)

    best_valid_loss = float("inf")
    accelerator.wait_for_everyone()
    auto_loss_curve, contr_loss_curve = [], []
    auto_loss_curve_val, contr_loss_curve_val = [], []

    if args.mem_bank == 'yes':
        if args.seq_sim == 'query' or args.seq_sim == 'mean':
            memory_bank = MemoryBank(device, llm_emb_size, bank_size=args.len_train)
            memory_bank_val = MemoryBank(device, llm_emb_size, bank_size=args.len_val)
        elif args.seq_sim == 'dual':
            memory_bank = DualMemoryBank(args.num_open, args.num_close, llm_emb_size, args.class_open, args.class_close, args.global2local_open, args.global2local_close, device)
            memory_bank_val = DualMemoryBank(args.num_open_val, args.num_close_val, llm_emb_size, args.class_open_val, args.class_close_val, args.global2local_open_val, args.global2local_close_val, device)

    for epoch in range(args.epochs):
        with tqdm(total=args.batch_size * len(train_loader), disable=not accelerator.is_main_process) as epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch}")
            start_time = time.time()
            model.train()
            total_loss = 0.0
            Loss_auto = 0.0
            Loss_contr = 0.0

            for i, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        if args.abl == 'baseline':
                            outputs, sequence, ans_s, len_ans = model(batch, device)
                        elif args.abl == 'bs_mg':
                            if args.mem_bank == 'yes':
                                outputs, sequence, ans_s, len_ans, contrastive_loss = model.forward_mem(batch, device, memory_bank, split='train')
                            else:
                                outputs, sequence, ans_s, len_ans, contrastive_loss = model(batch, device)
                    logits = outputs.logits
                    auto_loss = 0.
                    for b in range(logits.size(0)):
                        condensed_tokens = sequence[b, ans_s[b]:ans_s[b]+len_ans[b]]
                        condensed_logits = logits[b, ans_s[b]-1: ans_s[b]+len_ans[b]-1]
                        cur_loss = F.cross_entropy(condensed_logits.reshape(-1, logits.shape[-1]), condensed_tokens.flatten())
                        auto_loss += cur_loss
                    auto_loss = auto_loss/logits.size(0)

                    if args.abl == 'baseline':
                        loss = auto_loss
                    elif args.abl == 'bs_mg':
                        loss = auto_loss + args.lamda * contrastive_loss

                    if args.iters_to_accumulate > 1:
                        loss = loss / accelerator.gradient_accumulation_steps
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                    else:
                        accelerator.backward(loss)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    Loss_auto += auto_loss.item()
                    if args.abl == 'bs_mg':
                        Loss_contr += contrastive_loss.item()

                    total_loss += loss.item() * accelerator.gradient_accumulation_steps
                    avg_loss = total_loss / (i+1)
                    desc = f"Epoch {epoch} - loss {avg_loss:.20f}"
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(batch['images'].size(0))

            auto_loss_curve.append(Loss_auto / (i + 1))
            if args.abl == 'bs_mg':
                contr_loss_curve.append(Loss_contr / (i + 1))

        model.eval()
        with tqdm(total=args.batch_size * len(valid_loader), disable=not accelerator.is_main_process) as epoch_pbar:
            epoch_pbar.set_description(f"VAL Epoch {epoch}")

            total_loss_v = 0.0
            Loss_auto_v = 0.0
            Loss_contr_v = 0.0
            for i, batch in enumerate(valid_loader):
                with torch.no_grad():
                    if args.abl == 'baseline':
                        outputs, sequence, ans_s, len_ans = model(batch, device)
                    elif args.abl == 'bs_mg':
                        if args.mem_bank == 'yes':
                            outputs, sequence, ans_s, len_ans, contrastive_loss = model.forward_mem(batch, device, memory_bank_val, split='val')
                        else:
                            outputs, sequence, ans_s, len_ans, contrastive_loss = model(batch, device)
                    logits = outputs.logits
                    auto_loss = 0.
                    for b in range(logits.size(0)):
                        condensed_tokens = sequence[b, ans_s[b]:ans_s[b]+len_ans[b]]
                        condensed_logits = logits[b, ans_s[b]-1: ans_s[b]+len_ans[b]-1]
                        auto_loss += F.cross_entropy(condensed_logits.reshape(-1, logits.shape[-1]), condensed_tokens.flatten())
                    auto_loss=auto_loss/logits.size(0)
                    if args.abl == 'baseline':
                        loss = auto_loss
                    elif args.abl == 'bs_mg':
                        loss = auto_loss + args.lamda * contrastive_loss

                    Loss_auto_v += auto_loss.item()
                    if args.abl == 'bs_mg':
                        Loss_contr_v += contrastive_loss.item()

                    total_loss_v += loss.item()
                avg_val_loss = total_loss_v / (i + 1)
                desc = f"VAL Epoch {epoch} - loss {avg_val_loss:.20f}"
                epoch_pbar.set_description(desc)
                epoch_pbar.update(batch['images'].size(0))

        auto_loss_curve_val.append(Loss_auto_v / (i + 1))
        if args.abl == 'bs_mg':
            contr_loss_curve_val.append(Loss_contr_v / (i + 1))

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            model = accelerator.unwrap_model(model)
            accelerator.save(model.state_dict(), os.path.join(args.out_dir, f"checkpoint.pt")) #f"epoch_{epoch}.pt"
        scheduler.step()
        elapsed_time = time.time() - start_time
        print("VAL epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s".format(epoch + 1, args.epochs, avg_loss, avg_val_loss, elapsed_time))

    if args.abl == 'bs_mg':
        # train
        np.savez(os.path.join(args.out_dir, "bs_mg_loss_train.npz"), auto=np.array(auto_loss_curve), contr=np.array(contr_loss_curve))
        plot_loss(args, auto_loss_curve, contr_loss_curve, 'train')
        # val
        np.savez(os.path.join(args.out_dir, "bs_mg_loss_val.npz"), auto=np.array(auto_loss_curve_val), contr=np.array(contr_loss_curve_val))
        plot_loss(args, auto_loss_curve_val, contr_loss_curve_val, 'val')

    return model


def plot_loss(args, auto_loss_curve, contr_loss_curve, split):
    fig, ax1 = plt.subplots()
    # auto loss
    ax1.plot(list(range(args.epochs)), auto_loss_curve, color='blue', label='Auto regressive loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Auto regressive loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    # contrastive loss
    if args.abl == 'bs_mg':
        ax2 = ax1.twinx()
        ax2.plot(list(range(args.epochs)), contr_loss_curve, color='red', label='Contrastive loss')
        ax2.set_ylabel('Contrastive loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    lines, labels = ax1.get_legend_handles_labels()
    if args.abl == 'bs_mg':
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc='upper right')
    plt.savefig(os.path.join(args.out_dir, split+'_bs_mg_loss.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
