import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, LlamaForCausalLM, PhiForCausalLM, StableLmForCausalLM, GPT2LMHeadModel, CLIPVisionModel, CLIPTextModel, CLIPTokenizer, BeitModel, DetrModel, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from mappers import MLP, T2I_Attention, Linear_proj, QueryAggregator, ProjectionHead
from vit import VisionTransformer, interpolate_pos_embed
from nltk.translate.bleu_score import sentence_bleu
from utils import clean_answer



def load_checkpoint(model, pthpath):
    checkpoint = torch.load(pthpath, map_location='cpu')
    state_dict = checkpoint['model']
    state_dict['pos_embed'] = interpolate_pos_embed(state_dict['pos_embed'], model)
    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % pthpath)
    return model, msg


class VQAmodel(nn.Module):
    def __init__(self, args):
        super(VQAmodel, self).__init__()
        #------------------------------parameters---------------------------
        self.args = args
        self.model_type = args.model_type
        self.img_tokens = args.img_tokens
        self.mapping_type = args.mapping_type
        # ------------------------tokenizer and text encoder------------------
        if 'gpt2' in self.model_type:
            self.text_encoder = GPT2LMHeadModel.from_pretrained(args.text_path)
            self.text_encoder.config.pad_token_id = self.args.tokenizer.pad_token_id
            self.llm_emb_size = self.text_encoder.config.n_embd
        elif 'stablelm' in self.model_type:
            self.text_encoder = StableLmForCausalLM.from_pretrained(args.text_path)
            self.llm_emb_size = self.text_encoder.config.hidden_size
        elif 'llama' in self.model_type:
            self.text_encoder = LlamaForCausalLM.from_pretrained(args.text_path)
            self.llm_emb_size = self.text_encoder.config.hidden_size
        elif 'mistral' in args.model_type:
            self.text_encoder = AutoModelForCausalLM.from_pretrained(args.text_path)
            self.llm_emb_size = self.text_encoder.config.hidden_size
        # fine-tuning strategy
        if self.args.llmlora == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1, bias="none")
            self.text_encoder = get_peft_model(self.text_encoder, peft_config)
            self.text_encoder.print_trainable_parameters()
        elif self.args.llmlora == 'frozen':
            for param in self.text_encoder.transformer.parameters():
                param.requires_grad = False
        #-----------------------------visual encoder---------------------------
        if 'clip' in self.args.vis_path:
            self.image_encoder = CLIPVisionModel.from_pretrained(args.vis_path)
            target_modules = ["q_proj", "v_proj"]
            self.vis_emb_size = self.image_encoder.config.hidden_size
        elif 'vision' in self.args.vis_path:
            target_modules = ["qkv"]
            vitmodel = VisionTransformer(img_size=self.args.img_res)
            self.image_encoder, msg = load_checkpoint(vitmodel, self.args.vis_path)
            self.vis_emb_size = 768
        # fine-tuning strategy
        if self.args.vislora=='lora':
            loraconfig = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules = target_modules, bias="none",)
            self.image_encoder = get_peft_model(self.image_encoder, loraconfig)
            self.image_encoder.print_trainable_parameters()
        elif self.args.vislora=='frozen':
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        #---------------------- aggregation ----------------------------
        if self.args.seq_sim == 'query':
            self.query_aggregator = QueryAggregator(self.llm_emb_size)
        elif self.args.seq_sim == 'mean' or self.args.seq_sim == 'dual':
            self.mean_aggregator = ProjectionHead(self.llm_emb_size)
        #----------------------- projector ------------------------------
        if self.mapping_type == "MLP":
            self.img_project = MLP((self.vis_emb_size, (self.llm_emb_size * self.img_tokens) // 2,
                    self.llm_emb_size * self.img_tokens,
                    self.llm_emb_size * self.img_tokens))
        elif self.mapping_type == "Attention":
            self.img_project = T2I_Attention(self.vis_emb_size, self.llm_emb_size)
        elif self.mapping_type == "All":
            self.img_project = Linear_proj(self.vis_emb_size, self.llm_emb_size)
        else:
            raise ValueError("select valid mapping type")


    def forward(self, batch, device):
        if 'gpt2' in self.model_type:
            token_embder = self.text_encoder.transformer.wte
        elif 'stablelm' in self.model_type or 'llama' in self.model_type or 'mistral' in self.model_type:
            if self.args.llmlora == 'lora':
                token_embder = self.text_encoder.base_model.model.model.embed_tokens
            else:
                token_embder = self.text_encoder.model.embed_tokens

        image = batch['images']
        image = image.to(device=device, dtype=next(self.image_encoder.parameters()).dtype)
        if self.args.vis_hidden == 'yes' and 'vision' not in self.args.vis_path:
            img_feat = self.image_encoder(image, output_hidden_states=True)
        elif self.args.vis_hidden == 'no' or 'vision' in self.args.vis_path:
            img_feat = self.image_encoder(image)

        if self.mapping_type == "MLP":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat[:,0,:]
            else:
                imgemb = img_feat['pooler_output']
        elif self.mapping_type == "Attention" or self.mapping_type == "All":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat
            else:
                if self.args.vis_hidden == 'yes':
                    imgemb = img_feat.hidden_states[self.args.select_layer]
                else:
                    imgemb = img_feat.last_hidden_state

        embbedings, masks, sequences, ans_starts, len_ans, end_iq, end_ans, noans_emb_padded, noans_mask_padded, noans_seq_padded = self.prepare_multimodal_inputs_train(
            batch, imgemb, device, token_embder)
        if self.args.abl == 'baseline':
            outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks)
            return outputs, sequences, ans_starts, len_ans
        elif self.args.abl == 'bs_mg':
            outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, output_hidden_states=True)
            last_feat = outputs.hidden_states[-1]
            if self.args.seq_sim == 'query':
                loss = aggregate_emb_contrastive(last_feat, self.query_aggregator, end_iq, end_ans, temperature=self.args.temperature)
            elif self.args.seq_sim == 'mean':
                loss = mean_emb_contrastive(last_feat, self.mean_aggregator, end_iq, end_ans, temperature=self.args.temperature)
            return outputs, sequences, ans_starts, len_ans, loss


    def forward_extract(self, batch, device):
        if 'gpt2' in self.model_type:
            token_embder = self.text_encoder.transformer.wte
        elif 'stablelm' in self.model_type or 'llama' in self.model_type or 'mistral' in self.model_type:
            if self.args.llmlora == 'lora':
                token_embder = self.text_encoder.base_model.model.model.embed_tokens
            else:
                token_embder = self.text_encoder.model.embed_tokens

        image = batch['images']
        image = image.to(device=device, dtype=next(self.image_encoder.parameters()).dtype)
        if self.args.vis_hidden == 'yes' and 'vision' not in self.args.vis_path:
            img_feat = self.image_encoder(image, output_hidden_states=True)
        elif self.args.vis_hidden == 'no' or 'vision' in self.args.vis_path:
            img_feat = self.image_encoder(image)

        if self.mapping_type == "MLP":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat[:,0,:]
            else:
                imgemb = img_feat['pooler_output']
        elif self.mapping_type == "Attention" or self.mapping_type == "All":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat
            else:
                if self.args.vis_hidden == 'yes':
                    imgemb = img_feat.hidden_states[self.args.select_layer]
                else:
                    imgemb = img_feat.last_hidden_state

        embbedings, masks, sequences, ans_starts, len_ans, end_iq, end_ans, noans_emb_padded, noans_mask_padded, noans_seq_padded = self.prepare_multimodal_inputs_train_(batch, imgemb, device, token_embder)
        outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, output_hidden_states=True)
        last_feat = outputs.hidden_states[-1]
        bs = last_feat.size(0)
        aggr_IQ = []
        aggr_A = []
        emb_token_A = []
        for i in range(bs):
            if self.args.seq_sim == 'query':
                query_img_ques = self.query_aggregator(last_feat[i, :end_iq[i]].unsqueeze(0))
                query_ans = self.query_aggregator(last_feat[i, end_iq[i]: end_ans[i]].unsqueeze(0))
            elif self.args.seq_sim == 'mean' or self.args.seq_sim == 'dual':
                query_img_ques = self.mean_aggregator(last_feat[i, :end_iq[i]].mean(0, keepdim=True))
                query_ans = self.mean_aggregator(last_feat[i, end_iq[i]: end_ans[i]].mean(0, keepdim=True))
            aggr_IQ.append(query_img_ques)
            aggr_A.append(query_ans)
            emb_token_A.append(embbedings[i, end_iq[i]: end_ans[i]])
        aggr_IQ = torch.cat(aggr_IQ)
        aggr_A = torch.cat(aggr_A)
        aggr_IQ = F.normalize(aggr_IQ, dim=-1)
        aggr_A = F.normalize(aggr_A, dim=-1)

        return aggr_IQ, aggr_A, emb_token_A



    def forward_mem(self, batch, device, memory_bank, split):
        if 'gpt2' in self.model_type:
            token_embder = self.text_encoder.transformer.wte
        elif 'stablelm' in self.model_type or 'llama' in self.model_type or 'mistral' in self.model_type:
            if self.args.llmlora == 'lora':
                token_embder = self.text_encoder.base_model.model.model.embed_tokens
            else:
                token_embder = self.text_encoder.model.embed_tokens

        image = batch['images']
        image = image.to(device=device, dtype=next(self.image_encoder.parameters()).dtype)
        if self.args.vis_hidden == 'yes' and 'vision' not in self.args.vis_path:
            img_feat = self.image_encoder(image, output_hidden_states=True)
        elif self.args.vis_hidden == 'no' or 'vision' in self.args.vis_path:
            img_feat = self.image_encoder(image)

        if self.mapping_type == "MLP":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat[:,0,:]
            else:
                imgemb = img_feat['pooler_output']
        elif self.mapping_type == "Attention" or self.mapping_type == "All":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat
            else:
                if self.args.vis_hidden == 'yes':
                    imgemb = img_feat.hidden_states[self.args.select_layer]
                else:
                    imgemb = img_feat.last_hidden_state

        embbedings, masks, sequences, ans_starts, len_ans, end_iq, end_ans, noans_emb_padded, noans_mask_padded, noans_seq_padded = self.prepare_multimodal_inputs_train(
            batch, imgemb, device, token_embder)
        outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, output_hidden_states=True)
        last_feat = outputs.hidden_states[-1]
        if self.args.seq_sim == 'query':
            loss = aggregate_emb_contrastive_mem0(last_feat, self.query_aggregator, end_iq, end_ans, memory_bank, temperature=self.args.temperature)
        elif self.args.seq_sim == 'mean':
            loss = mean_emb_contrastive_mem0(last_feat, self.mean_aggregator, end_iq, end_ans, memory_bank, temperature=self.args.temperature)
        elif self.args.seq_sim == 'dual':
            label_ans = batch['answer_labels']
            label_oc = batch['is_closes']
            label_ht = batch['is_heads']
            if split == 'train':
                loss_close, loss_open, loss_open_head, loss_open_tail = dual_emb_contrastive_mem0(last_feat, self.mean_aggregator, end_iq, end_ans, memory_bank,
                                                          label_ans, label_oc, label_ht, self.args.global2local_close,
                                                          self.args.global2local_open, level=self.args.level,
                                                          temperature=self.args.temperature)
            elif split == 'val':
                loss_close, loss_open, loss_open_head, loss_open_tail = dual_emb_contrastive_mem0(last_feat, self.mean_aggregator, end_iq, end_ans, memory_bank,
                                                          label_ans, label_oc, label_ht, self.args.global2local_close_val,
                                                          self.args.global2local_open_val, level=self.args.level,
                                                          temperature=self.args.temperature)
            loss = loss_close + loss_open

        return outputs, sequences, ans_starts, len_ans, loss



    def generate(self, batch, device):
        if 'gpt2' in self.model_type:
            token_embder = self.text_encoder.transformer.wte
        elif 'stablelm' in self.model_type or 'llama' in self.model_type or 'mistral' in self.model_type:
            if self.args.llmlora == 'lora':
                token_embder = self.text_encoder.base_model.model.model.embed_tokens
            else:
                token_embder = self.text_encoder.model.embed_tokens

        image = batch['image']
        image = image.to(device=device, dtype = next(self.image_encoder.parameters()).dtype)
        if self.args.vis_hidden == 'yes' and 'vision' not in self.args.vis_path:
            img_feat = self.image_encoder(image.unsqueeze(0), output_hidden_states=True)
        elif self.args.vis_hidden == 'no' or 'vision' in self.args.vis_path:
            img_feat = self.image_encoder(image.unsqueeze(0))

        if self.mapping_type == "MLP":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat[:,0,:]
            else:
                imgemb = img_feat['pooler_output']
        elif self.mapping_type == "Attention" or self.mapping_type == "All":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat
            else:
                if self.args.vis_hidden == 'yes':
                    imgemb = img_feat.hidden_states[self.args.select_layer]
                else:
                    imgemb = img_feat.last_hidden_state
        # imgemb = imgemb.type(torch.float32)

        input_emb, masks, sequences, end_iq = self.prepare_multimodal_inputs_test(batch, imgemb, device, token_embder)

        if self.args.num_beam > 1:
            output_ids = self.text_encoder.generate(inputs_embeds=input_emb, attention_mask=masks,
                                                    max_new_tokens=20,
                                                    pad_token_id=self.args.tokenizer.pad_token_id,
                                                    eos_token_id=self.args.tokenizer.eos_token_id,
                                                    num_beams=self.args.num_beam,
                                                    num_return_sequences=self.args.num_beam)
            beam_pred_ans = self.args.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            beam_pred = [beam_pred_ans[k].strip() for k in range(self.args.num_beam)]
            if 'llama' in self.model_type or 'mistral' in self.model_type:
                beam_pred = [beam_predi.split('<')[0].replace("� ", "") for beam_predi in beam_pred]

            return beam_pred
        else:
            output_ids = self.text_encoder.generate(inputs_embeds=input_emb, attention_mask=masks,
                                                    max_new_tokens=20, pad_token_id=self.args.tokenizer.pad_token_id, eos_token_id=self.args.tokenizer.eos_token_id)
            predicted_ans = self.args.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if 'llama' in self.model_type or 'mistral' in self.model_type:
                predicted_ans = predicted_ans.split('<')[0].replace("� ", "")

            return predicted_ans


    def generate_bs(self, batch, device):
        if 'gpt2' in self.model_type:
            token_embder = self.text_encoder.transformer.wte
        elif 'phi' in self.model_type or 'stablelm' in self.model_type or 'llama' in self.model_type or 'mistral' in self.model_type:
            if self.args.llmlora == 'lora':
                token_embder = self.text_encoder.base_model.model.model.embed_tokens
            else:
                token_embder = self.text_encoder.model.embed_tokens

        image = batch['images']
        image = image.to(device=device, dtype=next(self.image_encoder.parameters()).dtype)
        if self.args.vis_hidden == 'yes' and 'vision' not in self.args.vis_path:
            img_feat = self.image_encoder(image, output_hidden_states=True)
        elif self.args.vis_hidden == 'no' or 'vision' in self.args.vis_path:
            img_feat = self.image_encoder(image)

        if self.mapping_type == "MLP":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat[:, 0, :]
            else:
                imgemb = img_feat['pooler_output']
        elif self.mapping_type == "Attention" or self.mapping_type == "All":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat
            else:
                if self.args.vis_hidden == 'yes':
                    imgemb = img_feat.hidden_states[self.args.select_layer]
                else:
                    imgemb = img_feat.last_hidden_state

        batch_pred = []
        for idd in range(len(imgemb)):
            input_emb, masks, sequences = self.prepare_multimodal_inputs_test_single(idd, batch, imgemb[idd].unsqueeze(0), device, token_embder)
            output_ids = self.text_encoder.generate(inputs_embeds=input_emb, attention_mask=masks,
                                                    max_new_tokens=20,
                                                    pad_token_id=self.args.tokenizer.pad_token_id)
            predicted_ans = self.args.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if 'llama' in self.model_type or 'mistral' in self.model_type:
                predicted_ans = clean_answer(predicted_ans).split('<')[0].replace("� ", "")
            batch_pred.append(predicted_ans)
        return batch_pred


    def generate_wrag(self, batch, db_qi_aggr, db_a_hidden, device):
        if 'gpt2' in self.model_type:
            token_embder = self.text_encoder.transformer.wte
        elif 'stablelm' in self.model_type or 'llama' in self.model_type or 'mistral' in self.model_type:
            if self.args.llmlora == 'lora':
                token_embder = self.text_encoder.base_model.model.model.embed_tokens
            else:
                token_embder = self.text_encoder.model.embed_tokens

        image = batch['image']
        image = image.to(device=device, dtype = next(self.image_encoder.parameters()).dtype)
        if self.args.vis_hidden == 'yes' and 'vision' not in self.args.vis_path:
            img_feat = self.image_encoder(image.unsqueeze(0), output_hidden_states=True)
        elif self.args.vis_hidden == 'no' or 'vision' in self.args.vis_path:
            img_feat = self.image_encoder(image.unsqueeze(0))

        if self.mapping_type == "MLP":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat[:,0,:]
            else:
                imgemb = img_feat['pooler_output']
        elif self.mapping_type == "Attention" or self.mapping_type == "All":
            if 'vision' in self.args.vis_path:
                imgemb = img_feat
            else:
                if self.args.vis_hidden == 'yes':
                    imgemb = img_feat.hidden_states[self.args.select_layer]
                else:
                    imgemb = img_feat.last_hidden_state
        # imgemb = imgemb.type(torch.float32)

        ans_pre_len = batch['ans_pre_len']
        input_emb, masks, sequences, end_iq, img_patches = self.prepare_multimodal_inputs_test_rag(batch, imgemb, device, token_embder)
        outputs = self.text_encoder(inputs_embeds=input_emb, attention_mask=masks, output_hidden_states=True)
        last_feat = outputs.hidden_states[-1]
        if self.args.seq_sim == 'query':
            query_img_ques = self.query_aggregator(last_feat[:, :end_iq])
        elif self.args.seq_sim == 'mean' or self.args.seq_sim == 'dual':
            query_img_ques = self.mean_aggregator(last_feat[:, :end_iq].mean(1))
        retrieved_embs, retrieved_mask, retrieved_seq = self.rag_retrieval(query_img_ques, db_qi_aggr, db_a_hidden, device, ans_pre_len)

        if self.args.retrive_fuse == 'concat':
            new_input_emb = torch.cat([input_emb[:, :end_iq], retrieved_embs, input_emb[:, end_iq:]], dim=1)
            new_mask = torch.cat([masks, retrieved_mask], dim=1)
            new_sequences = torch.cat([sequences[:, :end_iq], retrieved_seq, sequences[:, end_iq:]], dim=1)
            if self.args.num_beam > 1:
                output_ids = self.text_encoder.generate(inputs_embeds=new_input_emb,
                                                        attention_mask=new_mask,
                                                        max_new_tokens=20,
                                                        pad_token_id=self.args.tokenizer.pad_token_id,
                                                        eos_token_id=self.args.tokenizer.eos_token_id,
                                                        num_beams=self.args.num_beam,
                                                        num_return_sequences=self.args.num_beam)
                beam_pred_ans = self.args.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            else:
                output_ids = self.text_encoder.generate(inputs_embeds=new_input_emb,
                                                        attention_mask=new_mask, max_new_tokens=20,
                                                        pad_token_id=self.args.tokenizer.pad_token_id, eos_token_id=self.args.tokenizer.eos_token_id)
                predicted_ans = self.args.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        elif self.args.retrive_fuse == 'cross_attn':
            qs = batch['question_start']
            ql = batch['question_len']
            Qemb = input_emb[:, qs:qs + ql]
            Iemb = input_emb[:, end_iq - img_patches:end_iq]
            QIemb = torch.cat([Qemb, Iemb], dim=1)
            qi_new = cross_attention_no_param(QIemb, retrieved_embs, alpha=self.args.alpha)
            qi_new1 = torch.cat([input_emb[:, :qs], qi_new[:, :ql], input_emb[:, qs + ql:end_iq - img_patches], qi_new[:, ql:], input_emb[:, end_iq:]], dim=1)

            if self.args.num_beam > 1:
                output_ids = self.text_encoder.generate(inputs_embeds=qi_new1,
                                                        attention_mask=masks,
                                                        max_new_tokens=20,
                                                        pad_token_id=self.args.tokenizer.pad_token_id,
                                                        eos_token_id=self.args.tokenizer.eos_token_id,
                                                        num_beams=self.args.num_beam,
                                                        num_return_sequences=self.args.num_beam)
                beam_pred_ans = self.args.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            else:
                output_ids = self.text_encoder.generate(inputs_embeds=qi_new1, attention_mask=masks, max_new_tokens=20,
                                                        pad_token_id=self.args.tokenizer.pad_token_id, eos_token_id=self.args.tokenizer.eos_token_id)
                predicted_ans = self.args.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if self.args.num_beam > 1:
            beam_pred = [beam_pred_ans[k].strip() for k in range(self.args.num_beam)]
            if 'llama' in self.model_type or 'mistral' in self.model_type:
                beam_pred = [beam_predi.split('<')[0].replace("� ", "") for beam_predi in beam_pred]
            return beam_pred
        else:
            if 'llama' in self.model_type or 'mistral' in self.model_type:
                predicted_ans = predicted_ans.split('<')[0].replace("� ", "")
            return predicted_ans


    def rag_retrieval(self, query_img_ques, db_qi_aggr, db_a_hidden, device, ans_pre_len):
        db_qi_aggr = db_qi_aggr.to(device)
        query_norm = F.normalize(query_img_ques, dim=-1)
        db_norm = db_qi_aggr
        scores = torch.matmul(query_norm, db_norm.T)
        topk_scores, topk_idx = torch.topk(scores, k=self.args.top_k, dim=-1)

        retrieved_embs = []
        for idx in topk_idx[0].tolist():
            retrieved_embs.append(db_a_hidden[idx][ans_pre_len:].to(device))

        retrieved_embs = torch.cat(retrieved_embs, dim=0).unsqueeze(0).to(device)
        retrieved_mask = torch.ones(1, retrieved_embs.size(1), device=device)
        retrieved_seq = torch.full((1, retrieved_embs.size(1)), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long)

        return retrieved_embs, retrieved_mask, retrieved_seq


    def prepare_multimodal_inputs_train(self, batch, imgemb, device, token_embder):
        bs = len(imgemb)
        if self.args.mapping_type == 'MLP':
            imgemb = self.img_project(imgemb).view(-1, self.img_tokens, self.llm_emb_size)
        elif self.mapping_type == 'All':
            imgemb = self.img_project(imgemb)

        Input_Emb = []
        Masks = []
        Seq = []
        Ans_starts = []
        Len_ans = []
        end_iq = []
        end_ans = []
        question_start = batch['question_start'][0]
        prefix_a_len = batch['prefix_a_len'][0]

        for ii in range(bs):
            question_len = batch['question_len'][ii]
            answer_len = batch['answer_len'][ii]
            Len_ans.append(answer_len)
            input_ids = batch['input_ids'][ii]
            mask0 = input_ids.ne(self.args.IGNORE_INDEX)
            input_ids = input_ids[mask0]

            img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
            input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id+1:])).long()
            embedding = token_embder(input_ids_temp)

            if self.mapping_type == 'MLP' or self.args.mapping_type == 'All':
                image_emb = imgemb[ii]
            elif self.mapping_type == 'Attention':
                question_emb = embedding[question_start: question_start + question_len]
                image_emb = self.img_project(question_emb, imgemb[ii])

            sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long), input_ids[img_place_id+1:])).type(torch.long)
            Seq.append(sequence_all)
            ans_start_id = img_place_id + image_emb.size(0) + prefix_a_len
            Ans_starts.append(ans_start_id)
            end_iq.append(img_place_id + image_emb.size(0))
            end_ans.append(sequence_all.size(0))

            input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
            Input_Emb.append(input_embedding)
            mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool)
            Masks.append(mask)

        end_iq = torch.tensor(end_iq)
        end_ans = torch.tensor(end_ans)
        max_len = max(x.size(0) for x in Input_Emb)
        emb_padded = []
        mask_padded = []
        seq_padded = []
        noans_max_len = max(Ans_starts)
        noans_emb_padded = []
        noans_mask_padded = []
        noans_seq_padded = []
        for kk in range(bs):
            cur_embed = Input_Emb[kk]
            cur_len = cur_embed.size(0)
            pademb = torch.cat((cur_embed, torch.zeros((max_len - cur_len, cur_embed.size(1)), dtype=cur_embed.dtype, device=device)), dim=0)
            emb_padded.append(pademb)
            noans_pademb = torch.cat((cur_embed[:Ans_starts[kk],:], torch.zeros((noans_max_len - Ans_starts[kk], cur_embed.size(1)), dtype=cur_embed.dtype, device=device)), dim=0)
            noans_emb_padded.append(noans_pademb)

            cur_mask = Masks[kk]
            padmask = torch.cat((cur_mask, torch.zeros((max_len - cur_len), dtype=cur_mask.dtype, device=device)))
            mask_padded.append(padmask)
            noans_padmask = torch.cat((cur_mask[:Ans_starts[kk]], torch.zeros((noans_max_len - Ans_starts[kk]), dtype=cur_mask.dtype, device=device)))
            noans_mask_padded.append(noans_padmask)

            cur_seq = Seq[kk]
            padseq = torch.cat((cur_seq, torch.zeros((max_len - cur_len), dtype=cur_seq.dtype, device=device)))
            seq_padded.append(padseq)
            noans_padseq = torch.cat((cur_seq[:Ans_starts[kk]], torch.zeros((noans_max_len - Ans_starts[kk]), dtype=cur_seq.dtype, device=device)))
            noans_seq_padded.append(noans_padseq)

        emb_padded = torch.stack(emb_padded, dim=0)
        mask_padded = torch.stack(mask_padded, dim=0)
        seq_padded = torch.stack(seq_padded, dim=0)
        noans_emb_padded = torch.stack(noans_emb_padded, dim=0)
        noans_mask_padded = torch.stack(noans_mask_padded, dim=0)
        noans_seq_padded = torch.stack(noans_seq_padded, dim=0)
        Ans_starts = torch.tensor(Ans_starts, device=device)
        Len_ans = torch.tensor(Len_ans, device=device)
        return emb_padded, mask_padded, seq_padded, Ans_starts, Len_ans, end_iq, end_ans, noans_emb_padded, noans_mask_padded, noans_seq_padded


    def prepare_multimodal_inputs_train_(self, batch, imgemb, device, token_embder):
        bs = len(imgemb)
        if self.args.mapping_type == 'MLP':
            imgemb = self.img_project(imgemb).view(-1, self.img_tokens, self.llm_emb_size)
        elif self.mapping_type == 'All':
            imgemb = self.img_project(imgemb)

        Input_Emb = []
        Masks = []
        Seq = []
        Ans_starts = []
        Len_ans = []
        end_iq = []
        end_ans = []
        question_start = batch['question_start'][0]
        prefix_a_len = batch['prefix_a_len'][0]

        batch_input_id = batch['input_ids'].to(device)
        for ii in range(bs):
            question_len = batch['question_len'][ii]
            answer_len = batch['answer_len'][ii]
            Len_ans.append(answer_len)
            input_ids = batch_input_id[ii]
            mask0 = input_ids.ne(self.args.IGNORE_INDEX)
            input_ids = input_ids[mask0]

            img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
            input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id+1:])).long()
            embedding = token_embder(input_ids_temp)

            if self.mapping_type == 'MLP' or self.args.mapping_type == 'All':
                image_emb = imgemb[ii]
            elif self.mapping_type == 'Attention':
                question_emb = embedding[question_start: question_start + question_len]
                image_emb = self.img_project(question_emb, imgemb[ii])
            # print('image_emb.size(0): ', image_emb.size(0))

            sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long),input_ids[img_place_id+1:])).type(torch.long)
            Seq.append(sequence_all)
            ans_start_id = img_place_id + image_emb.size(0) + prefix_a_len
            Ans_starts.append(ans_start_id)
            end_iq.append(img_place_id + image_emb.size(0))
            end_ans.append(sequence_all.size(0))

            input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
            Input_Emb.append(input_embedding)
            mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool)
            Masks.append(mask)

        end_iq = torch.tensor(end_iq)
        end_ans = torch.tensor(end_ans)
        max_len = max(x.size(0) for x in Input_Emb)
        emb_padded = []
        mask_padded = []
        seq_padded = []
        noans_max_len = max(Ans_starts)
        noans_emb_padded = []
        noans_mask_padded = []
        noans_seq_padded = []
        for kk in range(bs):
            cur_embed = Input_Emb[kk]
            cur_len = cur_embed.size(0)
            pademb = torch.cat((cur_embed, torch.zeros((max_len - cur_len, cur_embed.size(1)), dtype=cur_embed.dtype, device=device)), dim=0)
            emb_padded.append(pademb)
            noans_pademb = torch.cat((cur_embed[:Ans_starts[kk],:], torch.zeros((noans_max_len - Ans_starts[kk], cur_embed.size(1)), dtype=cur_embed.dtype, device=device)), dim=0)
            noans_emb_padded.append(noans_pademb)

            cur_mask = Masks[kk]
            padmask = torch.cat((cur_mask, torch.zeros((max_len - cur_len), dtype=cur_mask.dtype, device=device)))
            mask_padded.append(padmask)
            noans_padmask = torch.cat((cur_mask[:Ans_starts[kk]], torch.zeros((noans_max_len - Ans_starts[kk]), dtype=cur_mask.dtype, device=device)))
            noans_mask_padded.append(noans_padmask)

            cur_seq = Seq[kk]
            padseq = torch.cat((cur_seq, torch.zeros((max_len - cur_len), dtype=cur_seq.dtype, device=device)))
            seq_padded.append(padseq)
            noans_padseq = torch.cat((cur_seq[:Ans_starts[kk]], torch.zeros((noans_max_len - Ans_starts[kk]), dtype=cur_seq.dtype, device=device)))
            noans_seq_padded.append(noans_padseq)

        emb_padded = torch.stack(emb_padded, dim=0)
        mask_padded = torch.stack(mask_padded, dim=0)
        seq_padded = torch.stack(seq_padded, dim=0)
        noans_emb_padded = torch.stack(noans_emb_padded, dim=0)
        noans_mask_padded = torch.stack(noans_mask_padded, dim=0)
        noans_seq_padded = torch.stack(noans_seq_padded, dim=0)
        Ans_starts = torch.tensor(Ans_starts, device=device)
        Len_ans = torch.tensor(Len_ans, device=device)
        return emb_padded, mask_padded, seq_padded, Ans_starts, Len_ans, end_iq, end_ans, noans_emb_padded, noans_mask_padded, noans_seq_padded


    def prepare_multimodal_inputs_test_rag(self, batch, imgemb, device, token_embder):
        if self.args.mapping_type == 'MLP':
            imgemb = self.img_project(imgemb).view(-1, self.img_tokens, self.llm_emb_size)
        elif self.args.mapping_type == 'All':
            imgemb = self.img_project(imgemb)

        question_start = batch['question_start']
        question_len = batch['question_len']
        input_ids = batch['input_id'].to(device)

        mask0 = input_ids.ne(self.args.IGNORE_INDEX)
        input_ids = input_ids[mask0]

        img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
        input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id + 1:]))
        embedding = token_embder(input_ids_temp)

        if self.args.mapping_type == 'MLP' or self.args.mapping_type == 'All':
            image_emb = imgemb[0]
        elif self.args.mapping_type == 'Attention':
            question_emb = embedding[question_start: question_start + question_len]
            image_emb = self.img_project(question_emb, imgemb)[0]

        # print('image_emb.size(0): ', image_emb.size(0))
        end_iq = img_place_id + image_emb.size(0)
        sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long), input_ids[img_place_id+1:]))
        input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
        mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool)
        input_embeddings = input_embedding.unsqueeze(0)
        masks = mask.unsqueeze(0)
        sequences = sequence_all.unsqueeze(0)
        img_patches = image_emb.size(0)
        return input_embeddings, masks, sequences, end_iq, img_patches


    def prepare_multimodal_inputs_test(self, batch, imgemb, device, token_embder):
        if self.args.mapping_type == 'MLP':
            imgemb = self.img_project(imgemb).view(-1, self.img_tokens, self.llm_emb_size)
        elif self.args.mapping_type == 'All':
            imgemb = self.img_project(imgemb)

        question_start = batch['question_start']
        question_len = batch['question_len']
        input_ids = batch['input_id'].to(device)
        mask0 = input_ids.ne(self.args.IGNORE_INDEX)
        input_ids = input_ids[mask0]

        img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
        input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id + 1:]))
        embedding = token_embder(input_ids_temp)

        if self.args.mapping_type == 'MLP' or self.args.mapping_type == 'All':
            image_emb = imgemb[0]
        elif self.args.mapping_type == 'Attention':
            question_emb = embedding[question_start: question_start + question_len]
            image_emb = self.img_project(question_emb, imgemb)[0]

        end_iq = img_place_id + image_emb.size(0)
        sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long), input_ids[img_place_id+1:]))
        input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
        mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool)
        input_embeddings = input_embedding.unsqueeze(0)
        masks = mask.unsqueeze(0)
        sequences = sequence_all.unsqueeze(0)
        return input_embeddings, masks, sequences, end_iq


    def prepare_multimodal_inputs_test_single(self, id, batch, imgemb, device, token_embder):
        if self.args.mapping_type == 'MLP':
            imgemb = self.img_project(imgemb).view(-1, self.img_tokens, self.llm_emb_size)
        elif self.args.mapping_type == 'All':
            imgemb = self.img_project(imgemb)

        question_start = batch['question_start'][id]
        question_len = batch['question_len'][id]
        input_ids = batch['input_ids'][id].to(device)
        mask0 = input_ids.ne(self.args.IGNORE_INDEX)
        input_ids = input_ids[mask0]

        img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
        input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id + 1:]))
        embedding = token_embder(input_ids_temp)

        if self.args.mapping_type == 'MLP' or self.args.mapping_type == 'All':
            image_emb = imgemb[0]
        elif self.args.mapping_type == 'Attention':
            question_emb = embedding[question_start: question_start + question_len]
            image_emb = self.img_project(question_emb, imgemb)[0]

        sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long), input_ids[img_place_id+1:]))
        input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
        mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool)
        input_embeddings = input_embedding.unsqueeze(0)
        masks = mask.unsqueeze(0)
        sequences = sequence_all.unsqueeze(0)
        return input_embeddings, masks, sequences


def aggregate_emb_contrastive(qia_emb, query_aggregator, end_iq, end_ans, temperature=0.1):
    bs = len(qia_emb)
    Query_IQ = []
    Query_A = []
    for i in range(bs):
        query_img_ques = query_aggregator(qia_emb[i, :end_iq[i]].unsqueeze(0))
        Query_IQ.append(query_img_ques)
        query_ans = query_aggregator(qia_emb[i, end_iq[i]: end_ans[i]].unsqueeze(0))
        Query_A.append(query_ans)

    Query_IQ = F.normalize(torch.cat(Query_IQ), dim=-1)
    Query_A = F.normalize(torch.cat(Query_A), dim=-1)
    sim_IQ = Query_IQ @ Query_IQ.T
    sim_A = Query_A @ Query_A.T

    logits = sim_IQ / temperature
    targets = sim_A / temperature
    mask = ~torch.eye(bs, dtype=torch.bool, device=sim_IQ.device)
    logits_masked = logits.masked_select(mask).view(bs, bs - 1)
    targets_masked = targets.masked_select(mask).view(bs, bs - 1)

    log_probs1 = F.log_softmax(logits_masked, dim=1)
    probs_target1 = F.softmax(targets_masked, dim=1)
    loss1 = F.kl_div(log_probs1, probs_target1, reduction='batchmean')
    log_probs2 = F.log_softmax(targets_masked, dim=1)
    probs_target2 = F.softmax(logits_masked, dim=1)
    loss2 = F.kl_div(log_probs2, probs_target2, reduction='batchmean')

    loss = (loss1 + loss2)/2
    return loss


def mean_emb_contrastive(qia_emb, mean_aggregator, end_iq, end_ans, temperature=0.1):
    bs = len(qia_emb)
    Query_IQ = []
    Query_A = []
    for i in range(bs):
        query_img_ques = mean_aggregator(qia_emb[i, :end_iq[i]].mean(0, keepdim=True))
        Query_IQ.append(query_img_ques)
        query_ans = mean_aggregator(qia_emb[i, end_iq[i]: end_ans[i]].mean(0, keepdim=True))
        Query_A.append(query_ans)

    Query_IQ = torch.cat(Query_IQ)
    Query_A = torch.cat(Query_A)
    Query_IQ = F.normalize(Query_IQ, dim=-1)
    Query_A = F.normalize(Query_A, dim=-1)

    sim_IQ = Query_IQ @ Query_IQ.T
    sim_A = Query_A @ Query_A.T

    logits = sim_IQ / temperature
    targets = sim_A / temperature

    mask = ~torch.eye(bs, dtype=torch.bool, device=sim_IQ.device)
    logits_masked = logits.masked_select(mask).view(bs, bs - 1)
    targets_masked = targets.masked_select(mask).view(bs, bs - 1)

    log_probs1 = F.log_softmax(logits_masked, dim=1)
    probs_target1 = F.softmax(targets_masked, dim=1)
    loss1 = F.kl_div(log_probs1, probs_target1, reduction='batchmean')
    log_probs2 = F.log_softmax(targets_masked, dim=1)
    probs_target2 = F.softmax(logits_masked, dim=1)
    loss2 = F.kl_div(log_probs2, probs_target2, reduction='batchmean')

    loss = (loss1 + loss2)/2
    return loss


def aggregate_emb_contrastive_mem(qia_emb, end_iq, a_emb, a_mask, memory_bank, query_aggregator, temperature=0.1):
    bs = len(qia_emb)
    Query_IQ = []
    Query_A = []
    for i in range(bs):
        query_img_ques = query_aggregator(qia_emb[i, :end_iq[i]].unsqueeze(0))
        Query_IQ.append(query_img_ques)
        query_ans = query_aggregator(a_emb[i, a_mask[i]].unsqueeze(0))
        Query_A.append(query_ans)

    Query_IQ = F.normalize(torch.cat(Query_IQ), dim=-1)
    Query_A = F.normalize(torch.cat(Query_A), dim=-1)
    memory_bank.update(Query_IQ.detach(), Query_A.detach())
    mem_qi, mem_a = memory_bank.get()

    # ------- current batch vs memory+batch -------
    mem_qi = mem_qi.to(Query_IQ.dtype)
    mem_a = mem_a.to(Query_A.dtype)
    sim_IQ = Query_IQ @ mem_qi.T / temperature
    sim_A = Query_A @ mem_a.T / temperature

    # ------- KL loss -------
    log_probs1 = F.log_softmax(sim_IQ, dim=1)
    probs_target1 = F.softmax(sim_A, dim=1)
    loss1 = F.kl_div(log_probs1, probs_target1, reduction='batchmean')
    log_probs2 = F.log_softmax(sim_A, dim=1)
    probs_target2 = F.softmax(sim_IQ, dim=1)
    loss2 = F.kl_div(log_probs2, probs_target2, reduction='batchmean')

    loss = (loss1 + loss2)/2
    return loss


def aggregate_emb_contrastive_mem0(qia_emb, query_aggregator, end_iq, end_ans, memory_bank, temperature=0.1):
    bs = len(qia_emb)
    Query_IQ = []
    Query_A = []
    for i in range(bs):
        query_img_ques = query_aggregator(qia_emb[i, :end_iq[i]].unsqueeze(0))
        Query_IQ.append(query_img_ques)
        query_ans = query_aggregator(qia_emb[i, end_iq[i]:end_ans[i]].unsqueeze(0))
        Query_A.append(query_ans)
    Query_IQ = F.normalize(torch.cat(Query_IQ), dim=-1)
    Query_A = F.normalize(torch.cat(Query_A), dim=-1)
    memory_bank.update(Query_IQ.detach(), Query_A.detach())
    mem_qi, mem_a = memory_bank.get()

    # ------- current batch vs memory+batch -------
    mem_qi = mem_qi.to(Query_IQ.dtype)
    mem_a = mem_a.to(Query_A.dtype)
    sim_IQ = Query_IQ @ mem_qi.T / temperature
    sim_A = Query_A @ mem_a.T / temperature

    # ------- KL loss -------
    log_probs1 = F.log_softmax(sim_IQ, dim=1)
    probs_target1 = F.softmax(sim_A, dim=1)
    loss1 = F.kl_div(log_probs1, probs_target1, reduction='batchmean')
    log_probs2 = F.log_softmax(sim_A, dim=1)
    probs_target2 = F.softmax(sim_IQ, dim=1)
    loss2 = F.kl_div(log_probs2, probs_target2, reduction='batchmean')

    loss = (loss1 + loss2)/2
    return loss


def mean_emb_contrastive_mem0(qia_emb, mean_aggregator, end_iq, end_ans, memory_bank, temperature=0.1):
    bs = len(qia_emb)
    Query_IQ = []
    Query_A = []
    for i in range(bs):
        query_img_ques = mean_aggregator(qia_emb[i, :end_iq[i]].mean(0, keepdim=True))
        Query_IQ.append(query_img_ques)
        query_ans = mean_aggregator(qia_emb[i, end_iq[i]:end_ans[i]].mean(0, keepdim=True))
        Query_A.append(query_ans)
    Query_IQ = F.normalize(torch.cat(Query_IQ), dim=-1)
    Query_A = F.normalize(torch.cat(Query_A), dim=-1)
    memory_bank.update(Query_IQ.detach(), Query_A.detach())
    mem_qi, mem_a = memory_bank.get()

    # ------- current batch vs memory+batch -------
    mem_qi = mem_qi.to(Query_IQ.dtype)
    mem_a = mem_a.to(Query_A.dtype)
    sim_IQ = Query_IQ @ mem_qi.T / temperature
    sim_A = Query_A @ mem_a.T / temperature

    # ------- KL loss -------
    log_probs1 = F.log_softmax(sim_IQ, dim=1)
    probs_target1 = F.softmax(sim_A, dim=1)
    loss1 = F.kl_div(log_probs1, probs_target1, reduction='batchmean')
    log_probs2 = F.log_softmax(sim_A, dim=1)
    probs_target2 = F.softmax(sim_IQ, dim=1)
    loss2 = F.kl_div(log_probs2, probs_target2, reduction='batchmean')

    loss = (loss1 + loss2)/2
    return loss


def mean_emb_contrastive_mem(qia_emb, end_iq, a_emb, a_mask, memory_bank, mean_aggregator, temperature=0.1):
    bs = len(qia_emb)
    Query_IQ = []
    Query_A = []
    for i in range(bs):
        query_img_ques = mean_aggregator(qia_emb[i, :end_iq[i]].mean(0, keepdim=True))
        Query_IQ.append(query_img_ques)
        query_ans = mean_aggregator(a_emb[i, a_mask[i]].mean(0, keepdim=True))
        Query_A.append(query_ans)
    Query_IQ = F.normalize(torch.cat(Query_IQ), dim=-1)
    Query_A = F.normalize(torch.cat(Query_A), dim=-1)
    memory_bank.update(Query_IQ.detach(), Query_A.detach())
    mem_qi, mem_a = memory_bank.get()

    # ------- current batch vs memory+batch -------
    mem_qi = mem_qi.to(Query_IQ.dtype)
    mem_a = mem_a.to(Query_A.dtype)
    sim_IQ = Query_IQ @ mem_qi.T / temperature
    sim_A = Query_A @ mem_a.T / temperature

    # ------- KL loss -------
    log_probs1 = F.log_softmax(sim_IQ, dim=1)
    probs_target1 = F.softmax(sim_A, dim=1)
    loss1 = F.kl_div(log_probs1, probs_target1, reduction='batchmean')
    log_probs2 = F.log_softmax(sim_A, dim=1)
    probs_target2 = F.softmax(sim_IQ, dim=1)
    loss2 = F.kl_div(log_probs2, probs_target2, reduction='batchmean')
    loss = (loss1 + loss2)/2

    return loss


def dual_emb_contrastive_mem0(qia_emb, mean_aggregator, end_iq, end_ans, memory_bank, label_ans, label_oc, label_ht, global2local_close, global2local_open, level='proto', temperature=0.1):
    device = qia_emb.device
    bs = len(qia_emb)
    Query_IQ, Query_A = [], []
    for i in range(bs):
        qi = mean_aggregator(qia_emb[i, :end_iq[i]].mean(0, keepdim=True))
        Query_IQ.append(qi)
        ans = mean_aggregator(qia_emb[i, end_iq[i]:end_ans[i]].mean(0, keepdim=True))
        Query_A.append(ans)

    Query_IQ = F.normalize(torch.cat(Query_IQ), dim=-1)
    Query_A = F.normalize(torch.cat(Query_A), dim=-1)

    # ====== update sample-level memory & prototype ======
    for is_close in [0, 1]:
        mask = (label_oc == is_close)
        if mask.any():
            memory_bank.update_samples(Query_IQ[mask].detach(), Query_A[mask].detach(), label_ans[mask], is_close=is_close)
            memory_bank.update_prototypes(Query_IQ[mask].detach(), Query_A[mask].detach(), label_ans[mask], is_close=is_close)

    close_mask = (label_oc == 1)
    loss_close = torch.tensor(0.0, device=device)
    if close_mask.any():
        q_qi = Query_IQ[close_mask]
        q_a = Query_A[close_mask]
        q_label = label_ans[close_mask]
        if level == 'sample':
            # sample-level contrastive
            mem_qi, mem_a, mem_label = memory_bank.get_samples(is_close=1)
            mem_qi = mem_qi.to(q_qi.dtype)
            mem_a = mem_a.to(q_a.dtype)
            mem_label = mem_label.to(q_label.dtype)
            # hard contrastive
            sim = q_qi @ mem_qi.T / temperature
            log_prob = F.log_softmax(sim, dim=1)
            mask = (q_label[:, None] == mem_label[None, :]).float()
            loss = -(log_prob * mask).sum(1) / mask.sum(1)
            loss_close_sample = loss.mean()

            # sim_qi = q_qi @ mem_qi.T / T
            # sim_a = q_a @ mem_a.T / T
            # loss_close_sample = F.kl_div(F.log_softmax(sim_qi, -1), F.softmax(sim_a, -1), reduction="batchmean")
            loss_close = loss_close_sample

        elif level == 'proto':
            # prototype-level contrastive
            protos_qi, protos_a = memory_bank.get_prototypes(is_close=1)
            # hard prototype contrastive
            protos_qi = protos_qi.to(q_qi.dtype)
            protos_a = protos_a.to(q_a.dtype)

            sim_proto = q_qi @ protos_qi.T / temperature
            q_label_local = torch.tensor([global2local_close[gid] for gid in q_label.tolist()], device=q_label.device)
            loss_close_proto = F.cross_entropy(sim_proto, q_label_local)
            # sim_qi_proto = q_qi @ protos_qi.T / T
            # sim_a_proto = q_a @ protos_a.T / T
            # loss_close_proto = F.kl_div(F.log_softmax(sim_qi_proto, -1), F.softmax(sim_a_proto, -1), reduction="batchmean")

            loss_close = loss_close_proto
        elif level == 'sample_proto':
            # sample-level contrastive
            mem_qi, mem_a, mem_label = memory_bank.get_samples(is_close=1)
            # hard contrastive
            mem_qi = mem_qi.to(q_qi.dtype)
            mem_a = mem_a.to(q_a.dtype)
            mem_label = mem_label.to(q_label.dtype)

            # targets = (q_label[:, None] == mem_label[None, :]).float()
            # loss_close_sample = F.cross_entropy(sim, targets.argmax(dim=1))
            sim = q_qi @ mem_qi.T / temperature
            log_prob = F.log_softmax(sim, dim=1)
            mask = (q_label[:, None] == mem_label[None, :]).float()
            loss = -(log_prob * mask).sum(1) / mask.sum(1)
            loss_close_sample = loss.mean()
            # sim_qi = q_qi @ mem_qi.T / T
            # sim_a = q_a @ mem_a.T / T
            # loss_close_sample = F.kl_div(F.log_softmax(sim_qi, -1), F.softmax(sim_a, -1), reduction="batchmean")

            # prototype-level contrastive
            protos_qi, protos_a = memory_bank.get_prototypes(is_close=1)
            # hard prototype contrastive
            protos_qi = protos_qi.to(q_qi.dtype)
            protos_a = protos_a.to(q_a.dtype)
            sim_proto = q_qi @ protos_qi.T / temperature
            q_label_local = torch.tensor([global2local_close[gid] for gid in q_label.tolist()], device=q_label.device)
            loss_close_proto = F.cross_entropy(sim_proto, q_label_local)

            # sim_qi_proto = q_qi @ protos_qi.T / T
            # sim_a_proto = q_a @ protos_a.T / T
            # loss_close_proto = F.kl_div(F.log_softmax(sim_qi_proto, -1), F.softmax(sim_a_proto, -1), reduction="batchmean")
            loss_close = loss_close_sample + loss_close_proto

    open_mask = (label_oc == 0)
    loss_open = torch.tensor(0.0, device=device)
    loss_open_head = torch.tensor(0.0, device=device)
    loss_open_tail = torch.tensor(0.0, device=device)
    if open_mask.any():
        q_qi = Query_IQ[open_mask]
        q_a = Query_A[open_mask]
        q_label = label_ans[open_mask]
        q_ht = label_ht[open_mask]
        head_mask = (q_ht == 1)

        if head_mask.any():
            q_head = q_qi[head_mask]
            l_head = q_label[head_mask]
            # prototype-level
            protos_qi, protos_a = memory_bank.get_prototypes(is_close=0)
            protos_qi = protos_qi.to(q_qi.dtype)
            protos_a = protos_a.to(q_a.dtype)

            sim_proto = q_head @ protos_qi.T / temperature
            l_head_local = torch.tensor([global2local_open[gid] for gid in l_head.tolist()], device=l_head.device)
            loss_open_head = F.cross_entropy(sim_proto, l_head_local)

        tail_mask = (q_ht == 0)
        if tail_mask.any():
            q_tail_qi = q_qi[tail_mask]
            q_tail_a = q_a[tail_mask]

            sim_a = q_tail_a @ protos_a.T
            prob_a = F.softmax(sim_a / temperature, dim=-1)
            sim_qi = q_tail_qi @ protos_qi.T
            log_prob_qi = F.log_softmax(sim_qi / temperature, dim=-1)

            loss_open_tail = F.kl_div(log_prob_qi, prob_a, reduction='batchmean')

        loss_open = loss_open_head + loss_open_tail

    return loss_close, loss_open, loss_open_head, loss_open_tail


def cross_attention_no_param(qi_emb, retrieved_embs, alpha):
    d = qi_emb.size(-1)
    attn_scores = (qi_emb @ retrieved_embs.transpose(-1, -2)) / (d ** 0.5)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_out = attn_weights @ retrieved_embs
    qi_new = qi_emb + alpha * attn_out
    return qi_new