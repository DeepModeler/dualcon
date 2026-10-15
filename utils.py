import os
import re
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from randaug import RandomAugment
import collections



def token_metric(gold_toks, pred_toks):
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, recall


def clean_answer(ans: str) -> str:
    """ Remove newlines, tabs, multiple spaces, and strip leading/trailing whitespace. """
    ans = ans.replace('\n', ' ')
    ans = ans.replace('\t', ' ')
    ans = re.sub(r'\s+', ' ', ans)
    return ans.strip()


def unique_with_priority(lst, priority_item):
    unique_items = set(lst)
    unique_items.add(priority_item)
    unique_items.discard(priority_item)
    result = [priority_item] + list(unique_items)
    return result


def extract_feat(model, loader, device, split, out_dir):
    model.eval()
    all_qi = []
    all_a = []
    all_a_tokenemb = []
    ans_text = []
    with torch.no_grad():
        for batch in loader:
            ans_text = ans_text + batch['answers']
            qi_aggr_emb, a_aggr_emb, a_tokenemb_list = model.forward_extract(batch, device)
            all_qi.append(qi_aggr_emb)
            all_a.append(a_aggr_emb)
            all_a_tokenemb.extend(a_tokenemb_list)
    all_qi = torch.cat(all_qi, dim=0)

    torch.save({
        "qi": all_qi,
        "a": all_a_tokenemb,
        }, os.path.join(out_dir, "retrieval_db_" + split + ".pkl"))


def process_img(image_res):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_res, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)