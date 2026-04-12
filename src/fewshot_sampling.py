import argparse
import random
import re
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
# from infer_llm_base import StopOnsString, StoppingCriteriaList
import re, json, os
from dateutil import parser as date_parser

import json, glob

import torch

def set_seed(seed: int) -> None: 
    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize_dates_in_text(text):
    def repl(m):
        s = m.group(0)

        # normalize "year month/monty day"
        s_norm = re.sub(
            r'(\d{4})\s*year\s*(\d{1,2})\s*(?:month|monty)\s*(\d{1,2})\s*day',
            r'\1-\2-\3',
            s,
            flags=re.I
        )

        try:
            dt = date_parser.parse(s_norm, dayfirst=False)
            return dt.strftime("%Y year %-m month %-d day")
        except Exception:
            return s  # fallback

    pattern = r"""
        '\b\d{4}\s*year\s*\d{1,2}\s*(?:month|monty)\s*\d{1,2}\s*day\b' |
        '\b\d{1,2}\s+[A-Za-z]+,\s*\d{4}\b' |
        '\b[A-Za-z]+\s+\d{1,2},\s*\d{4}\b' |
        '\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'|
		'\b\d{1,2}/[A-Za-z]+/\d{4}\b'
    """

    return re.sub(pattern, repl, text, flags=re.X | re.I)
 

def select_fewshot_examples(
    data,
    emb_xs,              # embedding of storage 
    emb_query,           # embedding of new query
    topk=4,               # number of demonstrations
    top_n_sim_sample_boundary=20,          # candidate pool size
    lambda_sim_div=0.5,    # trade-off: similarity vs diversity
):
    """
    Returns k (x, y) pairs that are:
    - similar to query_x
    - diverse among themselves (MMR)
    """

    # ---- Similarity to query ----
    sim_to_query = cosine_similarity(emb_xs, emb_query).squeeze()

    # ---- Stage 1: Top-N similar ----
    candidate_indices = np.argsort(sim_to_query)[::-1][:top_n_sim_sample_boundary]

    selected = []
    selected_indices = []

    # ---- Stage 2: MMR ----
    for _ in range(topk):
        mmr_scores = []

        for idx in candidate_indices:
            if idx in selected_indices:
                continue

            relevance = sim_to_query[idx]

            if not selected_indices:
                diversity_penalty = 0
            else:
                sim_to_selected = cosine_similarity(
                    emb_xs[idx].reshape(1, -1),
                    emb_xs[selected_indices]
                ).max()
                diversity_penalty = sim_to_selected

            mmr = lambda_sim_div * relevance - (1 - lambda_sim_div) * diversity_penalty
            mmr_scores.append((mmr, idx))

        _, best_idx = max(mmr_scores, key=lambda x: x[0])
        selected_indices.append(best_idx)
        selected.append(data[best_idx])

    return selected

def prompting_construct (query, fewshots):
    prompting = "### You are an expert in Semantic parsing task, which mapping from legal case to logical formulas (Note: following exact function name defined in the fewshot samples).\n"
    for shot_content in fewshots:
        prompting = prompting+ f"\n### Input: {shot_content[0]}\n### Logical Formulas Template:\n{shot_content[2]}\n### Output:\n{shot_content[1]}\n"
    prompting = prompting+ f"\n### Input: {query}\n### Logical Formulas Template:\n"
    return prompting

def gen_data_exp_bert_based_model(_data_path):


    # import data_lease_contract_generation
    # from data_lease_contract_generation import generate_bio_tags 
    # from importlib import reload  

    # data_lease_contract_generation=reload(data_lease_contract_generation)

    # BIO tagging function
    def generate_bio_tags(sentence, entities):
        bio_tags = ["O"] * len(sentence)
        for entity, (label, start_word) in entities.items():
            indices = [i for i, x in enumerate(sentence) if x == start_word]
            for start_idx in indices:
                if not (entity == " ".join(sentence[start_idx: start_idx + len(entity.split())])):
                    continue
                bio_tags[start_idx] = f"B-{label}"
                for i in range(start_idx + 1, start_idx + len(entity.split())):
                    bio_tags[i] = f"I-{label}"
        return bio_tags

    def bio_to_entities(bio_tagged):
        entities = []
        entity = None

        for idx,(word, tag) in enumerate(bio_tagged):
            if tag.startswith("B-"):
                if entity:
                    entities.append(entity)
                entity = [word, tag[2:],idx,idx+1]  # Start new entity
            elif tag.startswith("I-") and entity and tag[2:] == entity[1]:
                entity[0] += " " + word  # Continue current entity
                entity[3]=idx+1
            else:
                if entity:
                    entities.append(entity)
                    entity = None
        
        if entity:  # Add the last entity
            entities.append(entity)

        return [(entity[0], entity[1],entity[2], entity[3]) for entity in entities]
    
    def combine_entities_template(entities, template):

        generated_entities = entities
        generated_entities = dict([(k, v) for k, v in generated_entities.items()]) # ???? v.replace(" ", "_")
        
        template = re.sub(r'([\'\.\,?])', r' \1 ', template)
        template = template.replace("}", "} ").replace("{", " {")
        template = re.sub(r' {2,}', r' ', template.strip())

        # Fill template
        try:
            sentence = template.format(**generated_entities)
        except Exception as e:
            print(f'- [Exception] with template, entities:\n{template}\n{generated_entities}')
            return None

        _entities = dict([(v, (k, v.split(" ")[0])) for k, v in generated_entities.items()])
        
        # Generate BIO tags for this sentence
        bio_tags = generate_bio_tags(sentence.split(), _entities)
        
        # Check the reverse entity back from BIO is correct ? if it is not correct =>  skip and warning 
        checking_reverse_entities = bio_to_entities(list(zip(sentence.split(), bio_tags)))
        checking_reverse_entities = dict([(e[1],e[0]) for e in checking_reverse_entities])
        total_err = sum([1 if checking_reverse_entities.get(ent_label[0])!=ent_value else 0  for ent_value, ent_label in _entities.items()])
        if total_err > 0:
            print("[W] Can not recover entities from bio tags")
            print(list(zip(sentence.split(), bio_tags)), [(ent_label[0],ent_value, checking_reverse_entities.get(ent_label[0])) for ent_value, ent_label in _entities.items()])
            return None
        
        return {
                "sentence": sentence.split(),
                "bio_tags": bio_tags,
                "entities": generated_entities
            }
        

    def process_ner_data(_data_path):
        _data = json.load(open(_data_path))

        # Generate samples
        samples = []
        for e in _data:
            pairs_entities = e['entities']
            template = e['template']

            sample = combine_entities_template(pairs_entities, template)
            # Store the sample and BIO schema
            samples.append(sample)

        # Save to a JSON file
        with open(_data_path.replace(".json", "_nerdata.json"), "w") as f:
            json.dump(samples, f, indent=1,ensure_ascii=False)

        print(f"Generated {len(samples)} samples with BIO tagging saved to 'augmented_samples.json'.")

    process_ner_data(_data_path)

g_infor = {}
g_infor_respect_to_contract_id = {}

def diverse_sim_alg(query, args, sim_lm_model=None, storage_data=None, emb_x_storages=None, emb_templates=None, contract_id=None):
    global g_infor 
    global g_infor_respect_to_contract_id 

    def retrieve_few_shot(_g_infor, 
                          sim_lm_model=sim_lm_model, 
                          storage_data=storage_data, 
                          emb_x_storages=emb_x_storages, 
                          emb_templates=emb_templates,
                          contract_id=contract_id):
        if _g_infor is None:
            _g_infor = {}

        if storage_data is None:
            if contract_id is not None:
                data_for_fewshot = json.load(open(f'{PROJECT_FOLDER}/data/new_contracts.json')).get(contract_id)['fewshot_content']
                if data_for_fewshot is None:
                    print(f"ERR: can not find contract_id=`{contract_id}` for few-shot selection")
                    raise Exception(f"ERR: can not find contract_id=`{contract_id}` for few-shot selection")
                
                storage_data = _g_infor.get('storage_data') if 'storage_data' in _g_infor else data_for_fewshot 
                
            else:
                storage_data = _g_infor.get('storage_data') if 'storage_data' in _g_infor else \
                    json.load(open(f'{PROJECT_FOLDER}/data/all_samples_train-c3-t2_lam0.6_pool-rate-0.6_seed10.json')) 
            
        data = [[e['sentence'], e['logic_formulas'], e.get('logic_formulas_prototype'), e.get('template')] for e in storage_data]

        if sim_lm_model is None:
            sim_lm_model = _g_infor.get('sim_lm_model') if 'sim_lm_model' in _g_infor else \
                SentenceTransformer("Qwen/Qwen3-Embedding-8B") # Qwen/Qwen3-Embedding-0.6B
            _g_infor['sim_lm_model'] = sim_lm_model

        if emb_x_storages is None:
            # storage_data = _g_infor.get('storage_data') or json.load(open(f'{PROJECT_FOLDER}/data/all_samples_train-c3-t2_lam0.6_pool-rate-0.6_seed10.json'))
            # _g_infor['storage_data'] = storage_data
            emb_x_storages = _g_infor.get('emb_x_storages') if 'emb_x_storages' in _g_infor else \
                sim_lm_model.encode([e['sentence'] for e in storage_data], normalize_embeddings=True, batch_size=8, show_progress_bar=True) # encode the storage 
            _g_infor['emb_x_storages'] = emb_x_storages

        if emb_templates is None:
            template_data = [e.get('template') for e in storage_data]
            if None not in template_data:
                emb_templates = _g_infor.get('emb_templates') if 'emb_templates' in _g_infor else \
                    sim_lm_model.encode(template_data, normalize_embeddings=True, batch_size=8, show_progress_bar=True) # encode the storage 
                _g_infor['emb_templates'] = emb_templates

        emb_query = sim_lm_model.encode([query], normalize_embeddings=True, batch_size=8, show_progress_bar=True) # encode the query 
        fewshot = []

        if args.topk_sim_content > 0:
            fewshot_content = select_fewshot_examples(
                data,
                emb_x_storages,
                emb_query,
                topk=args.topk_sim_content,
                top_n_sim_sample_boundary=args.top_n_sim_sample_boundary,
                lambda_sim_div=args.lambda_sim_div
            )
            fewshot = fewshot + fewshot_content[:args.topk_sim_content]

        if args.topk_sim_templ > 0 and emb_templates is not None:
            fewshot_template = select_fewshot_examples(
                data,
                emb_templates,
                emb_query,
                topk=args.topk_sim_templ,
                top_n_sim_sample_boundary=args.top_n_sim_sample_boundary,
                lambda_sim_div=args.lambda_sim_div
            )
            fewshot = fewshot + fewshot_template[:args.topk_sim_templ]
        return fewshot 
    if contract_id is None:
        # if there is no specific requirement for contract 
        return retrieve_few_shot(g_infor)
    else:
        # if require a specific contract 
        if contract_id not in g_infor_respect_to_contract_id:
            g_infor_respect_to_contract_id[contract_id] = {}
        return retrieve_few_shot(g_infor_respect_to_contract_id.get(contract_id))

def gpt_decode(prompt, args, instruct_content = None, KEY=None, temperature=0.7):
    
    fs_samples = prompt.split("\n\n")
    template_parts = set([e.split("### Output:")[0].split("### Logical Formulas Template:")[-1].strip() for e in fs_samples[1:]])
    template_parts = [e for e in template_parts if len(e) > 0]
    if len(template_parts) == 1:
        template_found = template_parts[0]
        prompt = prompt.strip() + "\n"+ template_found.strip() +"\n"

    instruct_content = fs_samples[0].strip() + "(dont provide any explanation)." if instruct_content is None else instruct_content
    prompt = prompt.replace(fs_samples[0].strip(), instruct_content)
     
    client = OpenAI(api_key=KEY)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=temperature
    )
    raw_out = completion.choices[0].message.content.split("### Output:")[-1].strip()
    return raw_out

def llm_decode(prompt, args, tokenizer_=None, llm_model=None):

    fs_samples = prompt.split("\n\n")
    template_parts = set([e.split("### Output:")[0].split("### Logical Formulas Template:")[-1].strip() for e in fs_samples[1:]])
    template_parts = [e for e in template_parts if len(e) > 0]
    if len(template_parts) == 1:
        template_found = template_parts[0]
        prompt = prompt.strip() + "\n"+ template_found.strip() +"\n"

    if tokenizer_ is None:
        model_name = 'Qwen/Qwen3-8B'
        tokenizer_ = g_infor.get('tokenizer_') if 'tokenizer_' in g_infor else \
            AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
        tokenizer_.padding_side = 'left'
        tokenizer_.pad_token = tokenizer_.eos_token
        g_infor['tokenizer_'] = tokenizer_
           
    if llm_model is None:
        llm_model = g_infor.get('llm_model') if 'llm_model' in g_infor else \
            AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        g_infor['llm_model'] = llm_model

    # RUN inference 
    gen_kwargs = {'max_new_tokens': 500, 
                    'do_sample': False, 
                    'eos_token_id': None, #eos_tok_id, 
                    'pad_token_id': tokenizer_.pad_token_id,
                    'temperature': None, 
                    'top_p': None, 
                    'top_k': None, 
                    # "stopping_criteria": StoppingCriteriaList([StopOnsString(tokenizer_, "\n\n")])
                }

    inputs = tokenizer_([prompt], return_tensors="pt", padding_side='left', padding='longest').to(llm_model.device)
    input_length = inputs['input_ids'].shape[1]
    outputs = llm_model.generate(**inputs, **gen_kwargs)
    raw_out = tokenizer_.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    return raw_out[0].split("### Output:")[-1].strip()
    
PROJECT_FOLDER = "./"
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--project_folder', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=f".env")

    parser.add_argument('--seed',  type=int, default=10,  help='seed' )
    parser.add_argument('--topk_sim_content',  type=int, default=3,  help='topk sim ( content(q), content(s) )' )
    parser.add_argument('--topk_sim_templ',  type=int, default=2,  help='topk sim ( content(q), template(s) )' )
    parser.add_argument('--top_n_sim_sample_boundary',  type=int, default=10,  help='top_n_sim_sample_boundary' )
    parser.add_argument('--lambda_sim_div',  type=float, default=0.6,  help='lambda_sim_div* similarity + (1-lambda_sim_div) * diversity' )
    parser.add_argument('--similarity_model_name', type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument('--num_instance_per_template', type=int, default=1)
    parser.add_argument('--training_rate',  type=float, default=0.8,  help='training (pool) data rate over overall' )
    parser.add_argument('--note',  type=str, default='')
    
    args, unknown = parser.parse_known_args()

    set_seed(args.seed)

    PROJECT_FOLDER = args.project_folder if args.project_folder is not None else PROJECT_FOLDER

    if False:
        query = 'When paul inspected the cottage, which sarah had inherited, they found tina occupying it and having built garden H. paul requested tina to leave the cottage and take down garden H. Yet, tina asserts they have a rental agreement with sarah and that garden H is theirs. Can paul retrieve the cottage?'
        fewshot = diverse_sim_alg(query, args)
        prompting = prompting_construct(query, fewshot)
        output_raw = llm_decode(prompting)
        print(output_raw)
        exit()



    all_samples = []
    data_out = f"{PROJECT_FOLDER}/data/all_samples.json"
    k_case = f'c{args.topk_sim_content}-t{args.topk_sim_templ}'

    test_data_path = data_out.replace("data/", "data/prompting/").replace(".json", f"{args.note}_test-{k_case}_lam{args.lambda_sim_div}_pool-rate-{args.training_rate}_seed{args.seed}.json")
    train_data_path = data_out.replace("data/", "data/prompting/").replace(".json", f"{args.note}_train-{k_case}_lam{args.lambda_sim_div}_pool-rate-{args.training_rate}_seed{args.seed}.json")
    if os.path.exists(test_data_path) and os.path.exists(train_data_path):
        # exit if already generated 
        exit()

    org_data = list(glob.glob(f"{PROJECT_FOLDER}/data/*_augmented.json"))
    student_data = list(glob.glob(f"{PROJECT_FOLDER}/data/aug_by_js/*.json"))
    for file_name in org_data + student_data:

        # Generate samples
        print(file_name)
        dict_infor = json.load(open(file_name))
        # for template_type, dict_infor in all_kind_templates.items():
        templates = dict_infor['templates']
        pairs_entities = dict_infor['entities']
        num_instance = args.num_instance_per_template*len(templates)
        i_sample = 0
            
        max_err_time = 100
        while i_sample < num_instance and max_err_time > 0: 
            template = random.choice(templates)
            generated_entities = random.choice(pairs_entities)
            generated_entities = dict([(k, v.replace("_", " ").replace("'", "’")) for k, v in generated_entities.items()]) # ???? v.replace(" ", "_")

            # Store the sample and BIO schema
            full_formulas = "\n".join(dict_infor['logical_formulas'])
            if '}\'' not in full_formulas and '}\"' not in full_formulas:
                full_formulas = full_formulas.replace("}", "}'")
                full_formulas = full_formulas.replace("{", "'{")

            # Fill template
            try:
                sentence = template.format(**generated_entities).replace("'", "’")
                logic_formulas = full_formulas.format(**generated_entities)
            except Exception as e:
                # print(f'- [Exception] with template, entities:\n{template}\n{generated_entities}')
                max_err_time  += -1
                continue
  
            all_samples.append({
                "idx": i_sample, 
                "sentence": sentence, 
                "entities": generated_entities,
                "template" : template,
                "logic_formulas_prototype" : full_formulas,
                "logic_formulas": logic_formulas
            })
            i_sample += 1

    random.shuffle(all_samples)

    # Save to a JSON file
    print(f"Generated {len(all_samples)} samples saved to '{data_out}'.")

    storage_data = all_samples[:int(args.training_rate*len(all_samples))]
    data = [[e['sentence'], e['logic_formulas'], e['logic_formulas_prototype'], e['template']] for e in storage_data]
    
    test_data = all_samples[len(storage_data): ]

    # init similarity model 
    model = SentenceTransformer(args.similarity_model_name)

    xs = [x[0] for x in data]

    # ---- Encode ----
    emb_xs = model.encode([x[0] for x in data], normalize_embeddings=True, batch_size=8, show_progress_bar=True) # encode the storage 
    emb_templates = model.encode([x[3] for x in data], normalize_embeddings=True, batch_size=8, show_progress_bar=True) # encode the storage for template 
    emb_query = model.encode([x['sentence'] for x in test_data], normalize_embeddings=True, batch_size=8, show_progress_bar=True) # encode the query 

    for idx, e in enumerate(test_data):
        fewshot = []

        if args.topk_sim_content > 0:
            fewshot_content = select_fewshot_examples(
                data,
                emb_xs,
                emb_query[idx:idx+1],
                topk=args.topk_sim_content,
                top_n_sim_sample_boundary=args.top_n_sim_sample_boundary,
                lambda_sim_div=args.lambda_sim_div
            )
            fewshot = fewshot + fewshot_content[:args.topk_sim_content]

        if args.topk_sim_templ > 0:
            fewshot_template = select_fewshot_examples(
                data,
                emb_templates,
                emb_query[idx:idx+1],
                topk=args.topk_sim_templ,
                top_n_sim_sample_boundary=args.top_n_sim_sample_boundary,
                lambda_sim_div=args.lambda_sim_div
            )
            fewshot = fewshot + fewshot_template[:args.topk_sim_templ]

        if len(fewshot) == 0:
            print("ERRRR - can not find fewshot")
            exit(0)

        e['prompting'] = prompting_construct(e['sentence'], fewshot)
        print(e['prompting'])
        print()

    with open(test_data_path, "w") as f:
        json.dump(test_data, f, indent=1, ensure_ascii=False)

    with open(train_data_path, "w") as f:
        json.dump(storage_data, f, indent=1, ensure_ascii=False)

    config_path = test_data_path.replace(".json", f".config.json")
    with open(config_path, "w") as f:
        json.dump( dict([(k, v)for k, v in args.__dict__.items()] + [("test_data_path", test_data_path)]) , f, indent=1, ensure_ascii=False)

    gen_data_exp_bert_based_model(test_data_path)
    gen_data_exp_bert_based_model(train_data_path)