import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
import logging
logging.getLogger().setLevel(logging.ERROR)
os.environ["DEEPSPEED_LOG_LEVEL"] = "warning"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from PIL import Image
from transformers import CLIPImageProcessor, LlamaTokenizer, GPT2Tokenizer, AutoTokenizer, CLIPTokenizer
from train import Trainer
from test import Tester
from test0 import Tester0
from models import VQAmodel
from data_loaders.dataloader import VQAdataset, VQACollator
from data_loaders.dataloader_test import VQAdataset_test, VQACollator_test
from utils import extract_feat, process_img, set_random_seeds



def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='SLAKE', choices=('Path_VQA', 'VQA_RAD', 'SLAKE'))
    parser.add_argument("--model_type", type=str, default="stablelm", choices=("gpt2-xl","llama_7b", "stablelm", "mistral_7b"))
    parser.add_argument("--llmlora", type=str, default="lora", choices=("all", "lora", "frozen"))
    parser.add_argument("--vislora", type=str, default="lora", choices=("all", "lora", "frozen"))
    parser.add_argument("--mapping_type", type=str, default="All", choices=("All", "Attention", "MLP"))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument("--phase", type=str, default="train", choices=('train', 'test'))
    parser.add_argument("--testsplit", type=str, default="test", choices=('test', 'train'))
    parser.add_argument("--img_tokens", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--iters_to_accumulate", type=int, default=1)
    parser.add_argument("--select_layer", type=int, default=-1)
    parser.add_argument("--vis_hidden", type=str, default='yes', choices=('yes', 'no'))
    parser.add_argument("--multi_gpu", type=str, default="yes", choices=('yes', 'no'))
    parser.add_argument("--train_all", type=str, default="yes", choices=('yes', 'no'))
    parser.add_argument("--mem_bank", type=str, default="no", choices=('yes', 'no'))
    parser.add_argument("--wrag", type=str, default="yes", choices=('yes', 'no'))
    parser.add_argument("--img_res", type=int, default=384)
    parser.add_argument("--num_beam", type=int, default=1)
    parser.add_argument("--abl", type=str, default='bs_mg', choices=('baseline', 'bs_mg'))
    parser.add_argument("--pad", type=str, default='right', choices=('right', 'left'))
    parser.add_argument("--seq_sim", type=str, default="mean", choices=('query', 'mean', 'dual'))
    parser.add_argument("--gen", type=str, default='bs', choices=('bs', 'sig'))
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--headratio", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--level", type=str, default="sample", choices=('proto', 'sample', 'sample_proto'))
    parser.add_argument("--retrive_fuse", type=str, default="concat", choices=('cross_attn', 'concat'))
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--out_dir", default="./dualcon/checkpoint")
    parser.add_argument("--dataset_path", type=str, default="./VQA_datasets")
    parser.add_argument("--pretrain_path", type=str, default="./checkpoints")
    parser.add_argument("--vis_path", type=str, default="clip-vit-base-patch32")
    parser.add_argument("--clip_text_path", type=str, default="clip-vit-base-patch32")
    args = parser.parse_args()
    set_random_seeds(args.seed)
    return args


if __name__ == "__main__":
    args = parse_argument()
    args.IGNORE_INDEX = -100
    args.IMAGE_TOKEN_INDEX = -200

    if args.train_all == 'yes':
        suffix = f"dataset_{'vqa3'}_llm_{args.model_type}_mapping_{args.mapping_type}_llmlora_{args.llmlora}_vislora_{args.vislora}_abl_{args.abl}"
    else:
        if args.wrag == 'yes':
            suffix = f"dataset_{args.dataset}_llm_{args.model_type}_mapping_{args.mapping_type}_llmlora_{args.llmlora}_vislora_{args.vislora}_abl_{args.abl}_mem_bank_{args.mem_bank}_seq_sim_{args.seq_sim}_wrag_{args.wrag}_topk_{args.top_k}"
        else:
            suffix = f"dataset_{args.dataset}_llm_{args.model_type}_mapping_{args.mapping_type}_llmlora_{args.llmlora}_vislora_{args.vislora}_abl_{args.abl}_mem_bank_{args.mem_bank}_seq_sim_{args.seq_sim}_lamda_{args.lamda}"

    args.out_dir = os.path.join(args.out_dir, suffix)
    args.text_path = os.path.join(args.pretrain_path, args.model_type)
    args.vis_path = os.path.join(args.pretrain_path, args.vis_path)
    args.clip_text_path = os.path.join(args.pretrain_path, args.clip_text_path)
    print(args)

    '''================================ tokenizer =================================='''
    args.clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_text_path)
    if 'gpt2' in args.model_type:
        args.tokenizer = GPT2Tokenizer.from_pretrained(args.text_path)
        args.tokenizer.pad_token = args.tokenizer.eos_token
    elif 'llama' in args.model_type:
        args.tokenizer = LlamaTokenizer.from_pretrained(args.text_path)
        args.tokenizer.bos_token = '<|endoftext|>'
        args.tokenizer.eos_token = '<|endoftext|>'
        args.tokenizer.pad_token = '<|endoftext|>'
    elif 'mistral' in args.model_type:
        args.tokenizer = AutoTokenizer.from_pretrained(args.text_path)
        args.tokenizer.bos_token = '<|endoftext|>'
        args.tokenizer.eos_token = '<|endoftext|>'
        args.tokenizer.pad_token = '<|endoftext|>'
    elif 'stablelm' in args.model_type:
        args.tokenizer = AutoTokenizer.from_pretrained(args.text_path, padding_side="right", use_fast=False)
        args.tokenizer.unk_token = '<|reg0|>'

    if 'clip' in args.vis_path:
        args.image_processor = CLIPImageProcessor.from_pretrained(args.vis_path)
    elif 'vision' in args.vis_path:
        train_transform, test_transform = process_img(image_res=args.img_res)
        args.image_processor_train = train_transform
        args.image_processor_test = test_transform

    '''================================ dataset =================================='''
    if args.train_all =='yes':
        train_datapath = os.path.join(args.dataset_path, 'vqa3')
    else:
        train_datapath = os.path.join(args.dataset_path, args.dataset)
    test_datapath = os.path.join(args.dataset_path, args.dataset)

    train_dataset = VQAdataset(args, train_datapath, split='train')
    if args.testsplit == 'test':
        test_dataset = VQAdataset_test(args, test_datapath, split='test')
    elif args.testsplit == 'train':
        test_dataset = VQAdataset_test(args, test_datapath, split='train')

    if args.train_all == 'no':
        if args.dataset == 'VQA_RAD':
            val_dataset = VQAdataset(args, train_datapath, split="train")
        else:
            val_dataset = VQAdataset(args, train_datapath, split="val")
    else:
        val_dataset = VQAdataset(args, train_datapath, split="val")

    data_collator = VQACollator(tokenizer=args.tokenizer, clip_tokenizer=args.clip_tokenizer)
    data_collator_test = VQACollator_test(tokenizer=args.tokenizer, clip_tokenizer=args.clip_tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  collate_fn=data_collator, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                collate_fn=data_collator, num_workers=8, pin_memory=True, prefetch_factor=2)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                 collate_fn=data_collator_test)

    args.len_train = len(train_dataset)
    args.len_val = len(val_dataset)
    print('len train_dataloader: ', len(train_dataloader))
    print('len val_dataloader: ', len(val_dataloader))
    print('len test_dataloader: ', len(test_dataloader))
    print('len_train: ', args.len_train)
    print('len_val: ', args.len_val)

    args.num_close = train_dataset.num_close
    args.num_open = train_dataset.num_open
    close_class_ids = train_dataset.close_class_ids
    open_class_ids = train_dataset.open_class_ids
    args.class_close = 2
    args.class_open = train_dataset.class_answers - args.class_close
    args.global2local_close = {global_id: local_id for local_id, global_id in enumerate(close_class_ids)}
    args.global2local_open = {global_id: local_id for local_id, global_id in enumerate(open_class_ids)}

    args.num_close_val = val_dataset.num_close
    args.num_open_val = val_dataset.num_open
    close_class_ids_val = val_dataset.close_class_ids
    open_class_ids_val = val_dataset.open_class_ids
    args.class_close_val = 2
    args.class_open_val = val_dataset.class_answers - args.class_close_val
    args.global2local_close_val = {global_id: local_id for local_id, global_id in enumerate(close_class_ids_val)}
    args.global2local_open_val = {global_id: local_id for local_id, global_id in enumerate(open_class_ids_val)}

    test_answer_list = test_dataset.answer_list
    test_answer_list_tokens = args.tokenizer(test_answer_list, padding='longest', return_tensors="pt")
    args.answer_list_tokens = test_answer_list_tokens
    args.answer_list = test_answer_list

    '''==================================== model ======================================='''
    model = VQAmodel(args=args)
    if args.phase == 'train':
        model = Trainer(model, train_dataloader, val_dataloader, args, model.llm_emb_size)
    elif args.phase == 'test':
        ckpt_files = glob.glob(os.path.join(args.out_dir, "*.pt"))
        if ckpt_files:
            checkpoint_path = ckpt_files[0]
            if args.gen == 'bs':
                model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(args.device)), strict=False)
            else:
                model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(args.device)), strict=False)

        if args.wrag == 'no':
            if args.gen == 'bs':
                Tester(model, test_dataloader, args)
            else:
                Tester0(model, test_dataset, args)
        elif args.wrag == 'yes':
            extract_feat(model.to(args.device), train_dataloader, args.device, 'train', args.out_dir)
            db_data = torch.load(os.path.join(args.out_dir, "retrieval_db_train.pkl"))
            db_qi_aggr = db_data['qi']
            db_a_hidden = db_data['a']
            if args.gen == 'bs':
                Tester(model, test_dataloader, args, db_qi_aggr, db_a_hidden)
            else:
                Tester0(model, test_dataset, args, db_qi_aggr, db_a_hidden)