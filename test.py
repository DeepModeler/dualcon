import os
import re
import json
from tqdm import tqdm
import random
import evaluate
from evaluate import load
import bert_score
from bert_score import score
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import numpy as np
from torch.cuda.amp import autocast
from utils import token_metric, clean_answer, unique_with_priority



def Tester(model, test_loader, args, db_qi_aggr=None, db_a_hidden=None):
    model.eval()
    model = model.to(args.device)
    result = [] 
    for batch in test_loader:
        with torch.no_grad():
            if args.wrag == 'yes':
                out_text_all = model.generate_wrag(batch, db_qi_aggr, db_a_hidden, args.device)
            elif args.wrag == 'no':
                out_text_all = model.generate_bs(batch, args.device)

        bsize = len(out_text_all)
        for i in range(bsize):
            out_text = out_text_all[i].lower()
            answers_raw = str(batch["answers"][i])
            result_dict = {
                "image_name": batch["image_names"][i],
                "answer_type": batch["answer_types"][i],
                "question": batch["questions"][i],
                "answer": answers_raw,
                "is_head": int(batch['is_heads'][i].item()),
                "predicted": out_text,
            }
            result.append(result_dict)
            print('ground truth: ', answers_raw, '    predicted: ', out_text)


    print('test samples : ', len(result))
    f = open(args.out_dir + '/' + args.testsplit + '_result.json', 'w')
    json.dump(result, f)
    f.close()

    comput_result(result, args.tokenizer)


def comput_result(result_dict, tokenizer):
    refs = [itm['answer'] for itm in result_dict]
    preds = [itm['predicted'] for itm in result_dict]
    is_head= [itm['is_head'] for itm in result_dict]
    is_head = np.array(is_head, dtype=bool)
    is_tail = ~is_head

    is_close = np.array([ans in ('yes', 'no') for ans in refs], dtype=bool)
    is_open = ~is_close
    # ---------- BERTScore ----------
    bert_P, bert_R, bert_F1 = bert_score.score(
        refs=refs,
        cands=preds,
        model_type='./checkpoints/bert-base-uncased',
        num_layers=12,
        lang='en',
        batch_size=32,
    )

    # ---------- BLEU, token-level, accuracy ----------
    bleu_1_gram = []
    token_f1_list, token_recall_list = [], []
    acc_list, acc_yn_list, acc_oe_list = [], [], []

    for ans, pred in zip(refs, preds):
        # BLEU
        # bleu_1 = sentence_bleu(ans, pred, weights=(1, 0, 0, 0))
        # bleu_1_gram.append(bleu_1)

        ref_tokens = word_tokenize(ans)
        pred_tokens = word_tokenize(pred)
        bleu_1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0))
        bleu_1_gram.append(bleu_1)

        # token metric
        gold_tok = tokenizer.encode(ans)
        pred_tok = tokenizer.encode(pred)
        f1_cur, r_cur = token_metric(gold_tok, pred_tok)
        token_f1_list.append(f1_cur)
        token_recall_list.append(r_cur)
        # accuracy
        acc_list.append(pred == ans)
        if ans in ['yes', 'no']:
            acc_yn_list.append(pred == ans)
        else:
            acc_oe_list.append(pred == ans)

    bleu_1_gram = np.array(bleu_1_gram)
    token_f1_list = np.array(token_f1_list)
    token_recall_list = np.array(token_recall_list)

    acc = np.mean(acc_list)
    all_bleu_1_gram = bleu_1_gram.mean()
    all_bert_f1 = bert_F1.mean()
    all_bert_recall = bert_R.mean()
    all_token_f1 = token_f1_list.mean()
    all_token_recall = token_recall_list.mean()

    acc_yn = np.mean(acc_yn_list)

    acc_oe = np.mean(acc_oe_list)
    oe_bleu_1_gram = bleu_1_gram[is_open].mean().item()
    oe_bert_f1 = bert_F1[is_open].mean().item()
    oe_bert_recall = bert_R[is_open].mean().item()
    oe_token_f1 = token_f1_list[is_open].mean().item()
    oe_token_recall = token_recall_list[is_open].mean().item()

    print(f"All accuracy={acc:.4f}")
    print(f"All BLEU 1-gram: {all_bleu_1_gram:.4f}")
    print(f"All BERTScore: f1={all_bert_f1:.4f} recall={all_bert_recall:.4f}")
    print(f"All Token-level F1={all_token_f1:.4f} recall={all_token_recall:.4f}")

    print(f"close accuracy={acc_yn:.4f}")
    print(f"open accuracy={acc_oe:.4f}")

    print(f"open BLEU 1-gram: {oe_bleu_1_gram:.4f}")
    print(f"open BERTScore: f1={oe_bert_f1:.4f} recall={oe_bert_recall:.4f}")
    print(f"open Token-level F1={oe_token_f1:.4f} recall={oe_token_recall:.4f}")
    # ==============
    hd_oe_bleu_1_gram = bleu_1_gram[is_open*is_head].mean().item()
    hd_oe_bert_f1 = bert_F1[is_open*is_head].mean().item()
    hd_oe_bert_recall = bert_R[is_open*is_head].mean().item()
    hd_oe_token_f1 = token_f1_list[is_open*is_head].mean().item()
    hd_oe_token_recall = token_recall_list[is_open*is_head].mean().item()
    print(f"head (open) BLEU 1-gram: {hd_oe_bleu_1_gram:.4f}")
    print(f"head (open) BERTScore: f1={hd_oe_bert_f1:.4f} recall={hd_oe_bert_recall:.4f}")
    print(f"head (open) Token-level F1={hd_oe_token_f1:.4f} recall={hd_oe_token_recall:.4f}")

    tl_oe_bleu_1_gram = bleu_1_gram[is_open*is_tail].mean().item()
    tl_oe_bert_f1 = bert_F1[is_open*is_tail].mean().item()
    tl_oe_bert_recall = bert_R[is_open*is_tail].mean().item()
    tl_oe_token_f1 = token_f1_list[is_open*is_tail].mean().item()
    tl_oe_token_recall = token_recall_list[is_open*is_tail].mean().item()
    print(f"tail (open) BLEU 1-gram: {tl_oe_bleu_1_gram:.4f}")
    print(f"tail (open) BERTScore: f1={tl_oe_bert_f1:.4f} recall={tl_oe_bert_recall:.4f}")
    print(f"tail (open) Token-level F1={tl_oe_token_f1:.4f} recall={tl_oe_token_recall:.4f}")