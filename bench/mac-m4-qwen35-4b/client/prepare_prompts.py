import json
from pathlib import Path
from transformers import AutoTokenizer
root = Path(__file__).resolve().parent
model = '/Users/rentamac/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
records = [json.loads(line) for line in (root/'original_prompts.jsonl').read_text().splitlines()]
output = []
for target in (128,512,1024):
    for index in range(4):
        for nominal in (128,512,1024,2048):
            candidates=[r for r in records if r['metadata']['target_prompt_tokens']==nominal]
            r=candidates[index]
            ids=tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
            if len(ids)>=target:break
        assert len(ids)>=target
        cut=min(256,target//2)
        ids=ids[:cut]+ids[cut+len(ids)-target:]
        assert len(ids)==target
        output.append(dict(model=model,prompt=ids,ignore_eos=True,seed=0,
                           metadata=dict(id=f'exact{target}_slice{index:02}',target_prompt_tokens=target,
                                         actual_prompt_tokens=target,source_doc='war_and_peace.txt',slice_index=index)))
(root/'exact_prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output))
print('prepared',len(output),'exact prompts')
