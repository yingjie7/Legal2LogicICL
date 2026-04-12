import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F
import json 
from tqdm import tqdm 
 
import re
import glob 
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
all_pred =glob.glob('./data/all_samples_test-*result.json')

def protect_parentheses(expr):
	"""
	Replace parentheses inside quoted strings with [] 
	so that the parser will not break.
	"""

	def replacer(match):
		s = match.group(0)
		s = s.replace("(", "[")
		s = s.replace(")", "]")
		return s
	s2 = re.sub(r"'[^']*'", replacer, expr)
	s1 = re.sub(r'"[^"]*"', replacer, s2)
	return s1

def restore_parentheses(s):
	return s.replace("[", "(").replace("]", ")")

def parse_logic(expr):

	expr = protect_parentheses(expr)

	# extract arguments
	args = re.findall(r'\(([^()]*)\)', expr)

	flat_args = []
	for a in args:
		flat_args.extend([x.strip() for x in a.split(",")])

	# restore
	flat_args = [restore_parentheses(a.strip('"')) for a in flat_args]

	# structure
	structure = re.sub(r'\([^()]*\)', '()', expr)

	for idx in range(len(flat_args)):
		e = flat_args[idx].strip()
		if e.startswith("\"") or e.startswith("\'"):
			flat_args[idx] = e[1:]
		if e.endswith("\"") or e.endswith("\'"):
			flat_args[idx] = e[:-1]
	return structure, flat_args

def preprocess(strin):
	if "deterioration_fact'" in strin:
		strin = strin.replace("deterioration_fact'", "deterioration_fact('")
		if strin.endswith("'"):
			strin = strin.replace("'", "')")
	return strin 

def semantic_logic_score(pred_lg, gold_lg):
	pred_facts = pred_lg.strip().split("\n")
	gold_facts = gold_lg.strip().split("\n")
	if len(pred_facts)!= len(gold_facts):
		return 0.0
	lg_args1 = []
	lg_args2 = []
	for pred, gold in zip(pred_facts, gold_facts):
		struct1, args1_fact = parse_logic(preprocess(pred))
		struct2, args2_fact = parse_logic(preprocess(gold))

		# structure mismatch
		if struct1 != struct2:
			print(f"Diff structure pred={pred}, gold={gold}")
			return 0.0
		if len(args1_fact) != len(args2_fact):
			print(f"Difference number of arguments: pred='{pred}', gold='{gold}'")
			return 0.0
		
		lg_args1.extend(args1_fact)
		lg_args2.extend(args2_fact)

	if len(lg_args1) == 0:
		return 1
	emb = model.encode(lg_args1+lg_args2, convert_to_tensor=True)
	cos = F.cosine_similarity(emb[:len(lg_args1)], emb[len(lg_args1):], dim=1)
	return cos.mean()


for file_path in all_pred[:]:
	print(file_path)
	pred_data = json.load(open(file_path))
	all_sample_scores = []
	for sample_idx, s_content in enumerate(tqdm(pred_data['detail_pred'])):
		gold_facts = s_content['logic_formulas'] 
		predicted_facts = s_content['output_pred']   
		s_score = semantic_logic_score(gold_facts, predicted_facts) 
		all_sample_scores.append(s_score)

	relax_acc = sum(all_sample_scores)/len(all_sample_scores)
	pred_data['result']['relaxed_acc'] = relax_acc.item()
	print(pred_data['result']['acc'], relax_acc.item())
	json.dump(pred_data, open(file_path.replace(".json", "_relax.json"), 'wt'), indent=1)
	


