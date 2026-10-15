import os


def run_exp(phase, multigpu=False, topk=1, cuda=1, datasetname='SLAKE', llm='gpt2-xl', llmf='all', sim='mean', abl='baseline', ld=0.5, bs=32, mbk='yes', rag='no', ret_aug='concat'):
    if phase=='train':
        if multigpu:
            the_command = ('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=' + str('2,3,4,5') \
                          + ' accelerate launch' \
                          + ' --config_file ' + './dualcon/default_config.yaml' \
                          + ' --main_process_port 29525' \
                          + ' --num_processes=' + str(4) \
                          + ' --num_machines=' + str(1) \
                          + ' --mixed_precision=' + 'fp16' \
                          + ' --deepspeed_config_file=' + './dualcon/zero2.json' \
                          + ' main.py' \
                          + ' --phase=' + phase \
                          + ' --dataset=' + datasetname \
                          + ' --wrag=' + rag \
                          + ' --retrive_fuse=' + ret_aug \
                          + ' --top_k=' + str(topk) \
                          + ' --temperature=' + str(0.5) \
                          + ' --lamda=' + str(ld) \
                          + ' --batch_size=' + str(bs) \
                          + ' --mem_bank=' + mbk \
                          + ' --seq_sim=' + sim \
                          + ' --level=' + 'proto' \
                          + ' --mapping_type=' + 'All' \
                          + ' --model_type=' + llm \
                          + ' --train_all=' + 'no' \
                          + ' --llmlora=' + llmf \
                          + ' --vislora=' + 'lora' \
                          + ' --lr=' + str(2e-5) \
                          + ' --abl=' + abl)
        else:
            the_command = ('CUDA_VISIBLE_DEVICES=' + str(cuda) \
                           + ' python main.py' \
                           + ' --phase=' + phase \
                           + ' --device=cuda:'+str(cuda) \
                           + ' --dataset=' + datasetname \
                           + ' --wrag=' + rag \
                           + ' --retrive_fuse=' + ret_aug \
                           + ' --top_k=' + str(topk) \
                           + ' --temperature=' + str(0.5) \
                           + ' --lamda=' + str(ld) \
                           + ' --batch_size=' + str(bs) \
                           + ' --mem_bank=' + mbk \
                           + ' --seq_sim=' + sim \
                           + ' --level=' + 'proto' \
                           + ' --mapping_type=' + 'All' \
                           + ' --model_type=' + llm \
                           + ' --train_all=' + 'no' \
                           + ' --llmlora=' + llmf \
                           + ' --vislora=' + 'lora' \
                           + ' --lr=' + str(2e-5) \
                           + ' --abl=' + abl)
    elif phase=='test':
        the_command = ('python main.py'
                       + ' --phase=' + phase \
                       + ' --dataset=' + datasetname \
                       + ' --device=cuda:' + str(cuda) \
                       + ' --wrag=' + rag \
                       + ' --retrive_fuse=' + ret_aug \
                       + ' --top_k=' + str(topk) \
                       + ' --temperature=' + str(0.5) \
                       + ' --lamda=' + str(ld) \
                       + ' --batch_size=' + str(bs) \
                       + ' --mem_bank=' + mbk \
                       + ' --seq_sim=' + sim \
                       + ' --level=' + 'proto' \
                       + ' --mapping_type=' + 'All' \
                       + ' --model_type=' + llm \
                       + ' --train_all=' + 'no' \
                       + ' --llmlora=' + llmf \
                       + ' --vislora=' + 'lora' \
                       + ' --lr=' + str(2e-5) \
                       + ' --abl=' + abl
                       )

    os.system(the_command)
## Path_VQA , VQA_RAD, SLAKE

run_exp('train', multigpu=True, datasetname='SLAKE', llm='pmc_llama_7b_10', llmf='lora', sim='dual', abl='bs_mg', bs=32, ld=0.5, mbk='yes', rag='no')
run_exp('test', cuda=0, datasetname='SLAKE', llm='pmc_llama_7b_10', llmf='lora', sim='dual', abl='bs_mg', bs=32, ld=0.5, mbk='yes', rag='no')