import re
import os
import json
from collections import Counter
from dataclasses import dataclass
from typing import Sequence, Dict
from PIL import Image
import transformers
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100
class VQAdataset(Dataset):
    def __init__(self, args, path, split='train'):
        super(VQAdataset, self).__init__()
        self.IGNORE_INDEX = args.IGNORE_INDEX
        self.IMAGE_TOKEN_INDEX = args.IMAGE_TOKEN_INDEX
        self.split = split
        imgpath = os.path.join(path, 'images')
        jspath = os.path.join(path, 'json', split + '.json')
        self.jslists = json.load(open(jspath, 'r'))
        for item in self.jslists: item['image_name'] = os.path.join(imgpath, item['image_name'] + '.jpg')
        answer_list = [pre_answer(self.jslists[i]['answer']) for i in range(len(self.jslists))]

        self.answer2id = {ans: i for i, ans in enumerate(sorted(set(answer_list)))}
        self.answer_freq = Counter(answer_list)

        self.head_answers, head_perc, head_ans_classes, total = get_head_by_coverage(self.answer_freq, desired_coverage=args.headratio)
        head_sample_count = sum([self.answer_freq[ans] for ans in self.head_answers])

        all_open_classes = set([ans for ans in self.answer_freq.keys()]) - set(['yes', 'no'])
        tail_answers = all_open_classes - self.head_answers
        num_tail_classes = len(tail_answers)
        tail_sample_count = sum([self.answer_freq[ans] for ans in tail_answers])

        print('open head_perc: ', head_perc, ' open head_ans_classes: ', head_ans_classes, ' head_sample_count: ', head_sample_count,
              'open tail_ans_classes: ', num_tail_classes, ' tail_sample_count: ', tail_sample_count,
              ' open total: ', total, 'head+tail: ', head_sample_count + tail_sample_count)

        close_answers = {'yes', 'no'}
        open_answers = [ans for ans in self.answer2id.keys() if ans not in close_answers]
        self.close_class_ids = [self.answer2id[ans] for ans in close_answers]
        self.open_class_ids = [self.answer2id[ans] for ans in open_answers]
        self.num_close = sum(1 for ans in answer_list if ans in close_answers)
        self.num_open = len(answer_list) - self.num_close

        print('num_close: ', self.num_close, ' num_open: ', self.num_open)
        self.class_answers = len(self.answer_freq)

        self.answer_list = list(set(answer_list))
        self.args = args
        self.clip_tokenizer = args.clip_tokenizer
        self.tokenizer = args.tokenizer
        if 'vision' in args.vis_path:
            self.image_processor_train = args.image_processor_train
            self.image_processor_test = args.image_processor_test
        else:
            self.image_processor = args.image_processor

        self.prefix_q = self.tokenizer('question: ', return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_ctx = self.tokenizer(' context:', return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_a = self.tokenizer('answer: ', return_tensors="pt", add_special_tokens=False).input_ids[0]

    def __len__(self):
        return len(self.jslists)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        item = self.jslists[index]
        raw_answer = pre_answer(item['answer'])
        ans_type = item['answer_type']
        image_path = item['image_name']
        image_input = Image.open(image_path).convert('RGB')
        if 'vision' in self.args.vis_path:
            if self.split == 'train':
                image = self.image_processor_train(image_input)
            else:
                image = self.image_processor_test(image_input)
        else:
            image = self.image_processor(image_input, return_tensors="pt").pixel_values[0] # [c,w,h]

        question = pre_question(item['question'])
        answer = raw_answer + '<|endoftext|>'
        question_id = self.tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids[0]
        answer_id = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids[0]
        question_id_clip = self.clip_tokenizer(question, return_tensors="pt").input_ids[0]

        input_id = torch.cat([
            self.prefix_q,
            question_id,
            self.prefix_ctx,
            torch.tensor([self.IMAGE_TOKEN_INDEX], dtype=torch.long),
            self.prefix_a,
            answer_id], dim=0)

        question_start = len(self.prefix_q)
        question_len = len(question_id)
        answer_len = len(answer_id)
        prefix_a_len = len(self.prefix_a)

        answer_label = self.answer2id[raw_answer]
        is_close = int(raw_answer in ['yes', 'no'])
        is_head = int(raw_answer in self.head_answers or raw_answer in ['yes', 'no'])
        answersv = pre_answer(item['answer'])

        data_dict = dict(input_id=input_id, question_id_clip=question_id_clip, image=image, question_start=question_start, question_len=question_len,
                         prefix_a_len=prefix_a_len, answer_len=answer_len, question=question, answer=answersv,
                         image_name=image_path, answer_type=ans_type, is_close=is_close, is_head=is_head, answer_label=answer_label)

        return data_dict


@dataclass
class VQACollator(object):
    def __init__(self, tokenizer, clip_tokenizer):
        self.tokenizer = tokenizer
        self.clip_tokenizer = clip_tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_id"] for instance in instances]
        question_id_clips = [instance["question_id_clip"] for instance in instances]
        images = [instance["image"] for instance in instances]
        question_start = [instance["question_start"] for instance in instances]
        question_len = [instance["question_len"] for instance in instances]
        prefix_a_len = [instance["prefix_a_len"] for instance in instances]
        answer_len = [instance["answer_len"] for instance in instances]
        questions = [instance["question"] for instance in instances]
        answers = [instance["answer"] for instance in instances]
        image_names = [instance["image_name"] for instance in instances]
        answer_types = [instance["answer_type"] for instance in instances]
        is_closes = [instance["is_close"] for instance in instances]
        is_heads = [instance["is_head"] for instance in instances]
        answer_labels = [instance["answer_label"] for instance in instances]

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                        batch_first=True,
                                        padding_value=IGNORE_INDEX)
        padded_input_ids = padded_input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = padded_input_ids.ne(IGNORE_INDEX)

        padded_question_id_clips = torch.nn.utils.rnn.pad_sequence(question_id_clips,
                                        batch_first=True,
                                        padding_value=self.clip_tokenizer.pad_token_id)
        attention_mask_clip = padded_question_id_clips.ne(self.clip_tokenizer.pad_token_id)

        question_start = torch.tensor(question_start, dtype=torch.long)
        question_len = torch.tensor(question_len, dtype=torch.long)
        prefix_a_len = torch.tensor(prefix_a_len, dtype=torch.long)
        answer_len = torch.tensor(answer_len, dtype=torch.long)
        images = torch.stack(images, dim=0)  # [B, C, H, W]

        is_closes = torch.tensor(is_closes, dtype=torch.long)
        is_heads = torch.tensor(is_heads, dtype=torch.long)
        answer_labels = torch.tensor(answer_labels, dtype=torch.long)

        return {
            "input_ids": padded_input_ids,
            "question_id_clips": padded_question_id_clips,
            "attention_mask": attention_mask,
            "attention_mask_clip": attention_mask_clip,
            "images": images,
            "question_start": question_start,
            "question_len": question_len,
            "prefix_a_len": prefix_a_len,
            "answer_len": answer_len,
            "questions": questions,
            "answers": answers,
            "image_names": image_names,
            "answer_types": answer_types,
            "is_closes": is_closes,
            "is_heads": is_heads,
            "answer_labels": answer_labels
        }


def pre_question(question):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace(' \t', ' ').replace('is/are', 'is').replace('near/in', 'in')
    question = question.replace('>', 'more than ').replace('-yes/no', '')
    question = question.replace('x ray', 'xray').replace('x-ray', 'xray')
    question = question.rstrip(' ')
    return question


def pre_answer(answer):
    answer = str(answer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace(' \t', ' ')
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    answer = answer.replace(' - ', '-')
    answer = answer.replace('/', ' ')
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


def get_head_by_coverage(answer_freq, desired_coverage=0.6, exclude=set(['yes','no'])):
    freq_sorted = sorted([(ans, f) for ans, f in answer_freq.items() if ans not in exclude],
        key=lambda x: x[1], reverse=True
    )
    total = sum([f for _, f in freq_sorted])
    cum = 0
    head_answers = []
    for ans, f in freq_sorted:
        cum += f
        head_answers.append(ans)
        if cum / total >= desired_coverage:
            break
    head_answers_set = set(head_answers)
    head_perc = cum / total
    head_ans_classes = len(head_answers)
    return head_answers_set, head_perc, head_ans_classes, total