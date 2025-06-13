import argparse
import os
import json
from tqdm import tqdm
import traceback
import pandas as pd
import random

import evaluate

os.environ["http_proxy"] = "http://zhaoziheng:aJTjJb3qJJIhkd9uui0tnFQLUCsChFwTRsDluzv5ldgo8dQWu834Ac4rQbba@10.1.20.50:23128/"
os.environ["https_proxy"] = "http://zhaoziheng:aJTjJb3qJJIhkd9uui0tnFQLUCsChFwTRsDluzv5ldgo8dQWu834Ac4rQbba@10.1.20.50:23128/"

def evaluate_json(data_json, 
                  model, 
                  detailed_output_json, 
                  merged_output_json, 
                  calculate_bleu=True, 
                  calculate_ratescore=True,
                  calculate_radgraph=True,
                  calculate_meteor=True,
                  calculate_rouge=True,
                  calculate_bertscore=True,
                  train_statistic_json=None,
                  test_statistic_json=None):
    
    if calculate_ratescore:
        from RaTEScore import RaTEScore
        ratescore = RaTEScore()
    if calculate_bleu:
        # Set proxy environment variables for network access
        bleu = evaluate.load("bleu", experiment_id=f"{random.randint(0, 1000)}")
    if calculate_radgraph:
        from radgraph import F1RadGraph
        f1radgraph = F1RadGraph(reward_level="all")
    if calculate_rouge:
        rouge = evaluate.load('rouge', experiment_id=f"{random.randint(0, 1000)}")
    if calculate_meteor:
        meteor_model = evaluate.load('meteor', experiment_id=f"{random.randint(0, 1000)}")
    if calculate_bertscore:
        bertscore_model = evaluate.load('bertscore', experiment_id=f"{random.randint(0, 1000)}")
    
    with open(data_json, 'r') as f:
        data = json.load(f)
        
    results_by_class = {'max':{}}
    
    for key, datum in tqdm(data.items()):

        for abnormality in datum['abnormality']:
            if 'description' not in abnormality or abnormality['description'] == '':
                if 'label' not in abnormality or abnormality['label'] == '':
                    continue
                else:
                    gt = abnormality['label']
            else:
                gt = abnormality['description']
                
            if len(gt) == 0:
                continue
            
            max_bertscore = 0
            max_meteor = 0
            max_rouge1 = 0
            max_rougeL = 0
            max_radgraph = 0
            max_bleu1 = max_bleu2 = max_bleu3 = max_bleu4 = 0
            max_ratescore = 0
            for prompt_type in ['cropped', 'bbox', 'contour', 'ellipse']:
                if f'{model}_{prompt_type}_answer' not in abnormality:
                    continue
                
                answer = abnormality[f'{model}_{prompt_type}_answer']
                if len(answer) == 0:
                    continue
                
                if calculate_bertscore:
                    bertscore = bertscore_model.compute(predictions=[answer], references=[gt], lang='en')['f1']
                    bertscore = sum(bertscore)/len(bertscore) * 100
                    abnormality[f'{model}_{prompt_type}_bertscore'] = bertscore
                else:
                    bertscore = 0
                if bertscore > max_bertscore:
                    max_bertscore = bertscore
                
                if calculate_meteor:
                    meteor = meteor_model.compute(predictions=[answer], references=[gt])['meteor'] * 100
                    abnormality[f'{model}_{prompt_type}_meteor'] = meteor
                else:
                    meteor = 0
                if meteor > max_meteor:
                    max_meteor = meteor

                # rouge
                if calculate_radgraph:
                    results = rouge.compute(predictions=[answer], references=[gt])
                    rouge1 = results['rouge1'] * 100
                    rougeL = results['rougeL'] * 100
                    abnormality[f'{model}_{prompt_type}_rouge1'] = rouge1
                    abnormality[f'{model}_{prompt_type}_rougeL'] = rougeL
                else:
                    rouge1 = rougeL = 0
                if rouge1 > max_rouge1:
                    max_rouge1 = rouge1
                if rougeL > max_rougeL:
                    max_rougeL = rougeL
                
                # radgraph
                if calculate_radgraph:
                    mean_reward, _, _, _ = f1radgraph(hyps=[answer], refs=[gt])
                    radgraph_score = sum(mean_reward)/len(mean_reward) * 100
                    abnormality[f'{model}_{prompt_type}_radgraph'] = radgraph_score
                else:
                    radgraph_score = 0
                if radgraph_score > max_radgraph:
                    max_radgraph = radgraph_score
                
                # bleu
                if calculate_bleu:
                    bleu1_score = bleu.compute(predictions=[answer], references=[gt], max_order=1)['bleu'] * 100
                    bleu2_score = bleu.compute(predictions=[answer], references=[gt], max_order=2)['bleu'] * 100
                    bleu3_score = bleu.compute(predictions=[answer], references=[gt], max_order=3)['bleu'] * 100
                    bleu4_score = bleu.compute(predictions=[answer], references=[gt], max_order=4)['bleu'] * 100
                    abnormality[f'{model}_{prompt_type}_blue1'] = bleu1_score
                    abnormality[f'{model}_{prompt_type}_blue2'] = bleu2_score
                    abnormality[f'{model}_{prompt_type}_blue3'] = bleu3_score
                    abnormality[f'{model}_{prompt_type}_blue4'] = bleu4_score
                else:
                    bleu1_score = bleu2_score = bleu3_score = bleu4_score = 0
                if bleu1_score > max_bleu1:
                    max_bleu1 = bleu1_score
                if bleu2_score > max_bleu2:
                    max_bleu2 = bleu2_score
                if bleu3_score > max_bleu3:
                    max_bleu3 = bleu3_score
                if bleu4_score > max_bleu4:
                    max_bleu4 = bleu4_score
                    
                # ratescore
                if calculate_ratescore:
                    try:
                        rate_score = ratescore.compute_score([answer], [gt])[0] * 100
                        abnormality[f'{model}_{prompt_type}_ratescore'] = rate_score
                    except:
                        print(f'** Answer ** {answer}')
                        print(f'** GT ** {gt}')
                        traceback.print_exc()
                else:
                    rate_score = 0
                if rate_score == 50:
                    rate_score = 0
                if rate_score > max_ratescore:
                    max_ratescore = rate_score
                
                if prompt_type not in results_by_class:
                    results_by_class[prompt_type] = {}
                
                if 'category' not in abnormality:
                    if '其他' not in results_by_class[prompt_type]:
                        results_by_class[prompt_type]['其他'] = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': [], 'ratescore': [], 'bertscore': [], 'meteor': [], 'rouge1': [], 'rougeL': [], 'radgraph': []}
                    results_by_class[prompt_type]['其他']['bleu1'].append(bleu1_score)
                    results_by_class[prompt_type]['其他']['bleu2'].append(bleu2_score)
                    results_by_class[prompt_type]['其他']['bleu3'].append(bleu3_score)
                    results_by_class[prompt_type]['其他']['bleu4'].append(bleu4_score)
                    results_by_class[prompt_type]['其他']['ratescore'].append(rate_score)
                    results_by_class[prompt_type]['其他']['bertscore'].append(bertscore)
                    results_by_class[prompt_type]['其他']['meteor'].append(meteor)
                    results_by_class[prompt_type]['其他']['rouge1'].append(rouge1)
                    results_by_class[prompt_type]['其他']['rougeL'].append(rougeL)
                    results_by_class[prompt_type]['其他']['radgraph'].append(radgraph_score)

                else:
                    for category in abnormality["category"]:
                        if category not in results_by_class[prompt_type]:
                            results_by_class[prompt_type][category] = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': [], 'ratescore': [], 'bertscore': [], 'meteor': [], 'rouge1': [], 'rougeL': [], 'radgraph': []}
                        results_by_class[prompt_type][category]['bleu1'].append(bleu1_score)
                        results_by_class[prompt_type][category]['bleu2'].append(bleu2_score)
                        results_by_class[prompt_type][category]['bleu3'].append(bleu3_score)
                        results_by_class[prompt_type][category]['bleu4'].append(bleu4_score)
                        results_by_class[prompt_type][category]['ratescore'].append(rate_score)
                        results_by_class[prompt_type][category]['bertscore'].append(bertscore)
                        results_by_class[prompt_type][category]['meteor'].append(meteor)
                        results_by_class[prompt_type][category]['rouge1'].append(rouge1)
                        results_by_class[prompt_type][category]['rougeL'].append(rougeL)
                        results_by_class[prompt_type][category]['radgraph'].append(radgraph_score)

            if 'category' not in abnormality:
                if '其他' not in results_by_class['max']:
                    results_by_class['max']['其他'] = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': [], 'ratescore': [], 'bertscore': [], 'meteor': [], 'rouge1': [], 'rougeL': [], 'radgraph': []}
                results_by_class['max']['其他']['bleu1'].append(max_bleu1)
                results_by_class['max']['其他']['bleu2'].append(max_bleu2)
                results_by_class['max']['其他']['bleu3'].append(max_bleu3)
                results_by_class['max']['其他']['bleu4'].append(max_bleu4)
                results_by_class['max']['其他']['ratescore'].append(max_ratescore)
                results_by_class['max']['其他']['bertscore'].append(max_bertscore)
                results_by_class['max']['其他']['meteor'].append(max_meteor)
                results_by_class['max']['其他']['rouge1'].append(max_rouge1)
                results_by_class['max']['其他']['rougeL'].append(max_rougeL)
                results_by_class['max']['其他']['radgraph'].append(max_radgraph)
            else:    
                for category in abnormality["category"]:
                    if category not in results_by_class['max']:
                        results_by_class['max'][category] = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': [], 'ratescore': [], 'bertscore': [], 'meteor': [], 'rouge1': [], 'rougeL': [], 'radgraph': []}
                    results_by_class['max'][category]['bleu1'].append(max_bleu1)
                    results_by_class['max'][category]['bleu2'].append(max_bleu2)
                    results_by_class['max'][category]['bleu3'].append(max_bleu3)
                    results_by_class['max'][category]['bleu4'].append(max_bleu4)
                    results_by_class['max'][category]['ratescore'].append(max_ratescore)
                    results_by_class['max'][category]['bertscore'].append(max_bertscore)
                    results_by_class['max'][category]['meteor'].append(max_meteor)
                    results_by_class['max'][category]['rouge1'].append(max_rouge1)
                    results_by_class['max'][category]['rougeL'].append(max_rougeL)
                    results_by_class['max'][category]['radgraph'].append(max_radgraph)
                    
    for prompt_type in results_by_class:
        overall_bleu1 = []
        overall_bleu2 = []
        overall_bleu3 = []
        overall_bleu4 = []
        overall_ratescore = []
        overall_bertscore = []
        overall_meteor = []
        overall_rouge1 = []
        overall_rougeL = []
        overall_radgraph = []
        
        for category in results_by_class[prompt_type]:
            # Calculate averages for all metrics
            for metric_name, metric_score_ls in zip(['bleu1', 'bleu2', 'bleu3', 'bleu4', 'ratescore', 'bertscore', 'meteor', 'rouge1', 'rougeL', 'radgraph'], [overall_bleu1, overall_bleu2, overall_bleu3, overall_bleu4, overall_ratescore, overall_bertscore, overall_meteor, overall_rouge1, overall_rougeL, overall_radgraph]):
                if len(results_by_class[prompt_type][category][metric_name]) >= 0:
                    tmp = sum(results_by_class[prompt_type][category][metric_name]) / (len(results_by_class[prompt_type][category][metric_name])+1e-20)
                    if tmp >= 0:
                        results_by_class[prompt_type][category][metric_name] = tmp
                        metric_score_ls.append(results_by_class[prompt_type][category][metric_name])    # 每个category内部平均得到一个值
        
        # Calculate overall scores
        results_by_class[prompt_type]['overall'] = {}
        for metric_name, metric_score_ls in zip(['bleu1', 'bleu2', 'bleu3', 'bleu4', 'ratescore', 'bertscore', 'meteor', 'rouge1', 'rougeL', 'radgraph'], [overall_bleu1, overall_bleu2, overall_bleu3, overall_bleu4, overall_ratescore, overall_bertscore, overall_meteor, overall_rouge1, overall_rougeL, overall_radgraph]):
            avg = sum(metric_score_ls) / (len(metric_score_ls)+1e-20)
            if avg >= 0:
                results_by_class[prompt_type]['overall'][metric_name] = avg # 所有category平均得到一个值
    
    with open(merged_output_json, 'w', encoding='utf-8') as f:
        json.dump(results_by_class, f, ensure_ascii=False, indent=4)
    print('Summary results saved to: ', merged_output_json)
        
    with open(detailed_output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print('Detailed results saved to: ', detailed_output_json)
    
    def save_results_to_excel(results_by_class, output_excel, train_statistic_json=None, test_statistic_json=None):
        """
        Save results_by_class to an Excel file with each prompt_type as a separate sheet.
        
        Args:
            results_by_class (dict): Results dictionary organized by prompt type and category
            output_excel (str): Path to the output Excel file
            train_statistic_json (str): Path to the train category statistic json file
            test_statistic_json (str): Path to the test category statistic json file
        """
        
        if train_statistic_json is not None:
            with open(train_statistic_json, 'r', encoding='utf-8') as f:
                train_category_counts = json.load(f)
        
        if test_statistic_json is not None:
            with open(test_statistic_json, 'r', encoding='utf-8') as f:
                test_category_counts = json.load(f)
        
        # Create Excel writer
        with pd.ExcelWriter(output_excel) as writer:
            for prompt_type in results_by_class:
                # Create data for this sheet
                categories = []
                bleu1_scores = []
                bleu2_scores = []
                bleu3_scores = []
                bleu4_scores = []
                rate_scores = []
                bertscore_scores = []
                meteor_scores = []
                rouge1_scores = []
                rougeL_scores = []
                radgraph_scores = []
                se_scores = []
                sp_scores = []
                
                if test_statistic_json is not None:
                    test_num = []
                if train_statistic_json is not None:
                    train_num = []
                
                # Extract data for each category
                for category, scores in results_by_class[prompt_type].items():
                    if category != 'overall':  # Add all categories except 'overall'
                        categories.append(category)
                        bleu1_scores.append(scores.get('bleu1', 0))
                        bleu2_scores.append(scores.get('bleu2', 0))
                        bleu3_scores.append(scores.get('bleu3', 0))
                        bleu4_scores.append(scores.get('bleu4', 0))
                        rate_scores.append(scores.get('ratescore', 0))
                        bertscore_scores.append(scores.get('bertscore', 0))
                        meteor_scores.append(scores.get('meteor', 0))
                        rouge1_scores.append(scores.get('rouge1', 0))
                        rougeL_scores.append(scores.get('rougeL', 0))
                        radgraph_scores.append(scores.get('radgraph', 0))
                        
                        if train_statistic_json is not None:
                            if category in train_category_counts:
                                train_num.append(train_category_counts[category]['num_detected'])
                            else:
                                train_num.append(0)
                        if test_statistic_json is not None:
                            if category in test_category_counts:
                                test_num.append(test_category_counts[category]['num_detected'])
                            else:
                                test_num.append(0)
                
                # Add overall score at the end
                if 'overall' in results_by_class[prompt_type]:
                    categories.append('overall')
                    overall = results_by_class[prompt_type]['overall']
                    bleu1_scores.append(overall.get('bleu1', 0))
                    bleu2_scores.append(overall.get('bleu2', 0))
                    bleu3_scores.append(overall.get('bleu3', 0))
                    bleu4_scores.append(overall.get('bleu4', 0))
                    rate_scores.append(overall.get('ratescore', 0))
                    bertscore_scores.append(overall.get('bertscore', 0))
                    meteor_scores.append(overall.get('meteor', 0))
                    rouge1_scores.append(overall.get('rouge1', 0))
                    rougeL_scores.append(overall.get('rougeL', 0))
                    radgraph_scores.append(overall.get('radgraph', 0))
                    
                    if train_statistic_json is not None:
                        train_num.append(sum(train_category_counts[category]['num_detected'] for category in train_category_counts))
                    if test_statistic_json is not None:
                        test_num.append(sum(test_category_counts[category]['num_detected'] for category in test_category_counts))
                
                # Create DataFrame and save to sheet
                df = {
                    'class': categories,
                    'bleu1': bleu1_scores,
                    'bleu2': bleu2_scores,
                    'bleu3': bleu3_scores,
                    'bleu4': bleu4_scores,
                    'ratescore': rate_scores,
                    'bertscore': bertscore_scores,
                    'meteor': meteor_scores,
                    'rouge1': rouge1_scores,
                    'rougeL': rougeL_scores,
                    'radgraph': radgraph_scores,
                }
                if test_statistic_json is not None:
                    df['test_num'] = test_num
                if train_statistic_json is not None:
                    df['train_num'] = train_num
                df = pd.DataFrame(df)
                
                df.to_excel(writer, sheet_name=prompt_type, index=False)
        
        print('Summary xlsx saved to: ', output_excel)

    save_results_to_excel(results_by_class, merged_output_json.replace('json', 'xlsx'), train_statistic_json, test_statistic_json)

if __name__ == "__main__":
    def str2bool(v):
        return v.lower() in ('true', 't')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str)
    parser.add_argument("--model", type=str, help='MedDr, LLaVA-Med, BiomedGPT, MedFlamingo, GPT4o, QWen25VL7B')
    parser.add_argument("--merged_output_json", type=str)
    parser.add_argument("--detailed_output_json", type=str, default=None)
    parser.add_argument("--bleu", type=str2bool, default='True', help='must turn proxy on!')
    parser.add_argument("--ratescore", type=str2bool, default='True')
    parser.add_argument("--radgraph", type=str2bool, default='True')
    parser.add_argument("--rouge", type=str2bool, default='True')
    parser.add_argument("--meteor", type=str2bool, default='True')
    parser.add_argument("--bertscore", type=str2bool, default='True')
    parser.add_argument("--train_statistic_json", type=str, default=None)
    parser.add_argument("--test_statistic_json", type=str, default=None)
    args = parser.parse_args()
    
    if args.detailed_output_json is None:
        args.detailed_output_json = args.data_json

    evaluate_json(
        args.data_json, 
        model=args.model,
        merged_output_json=args.merged_output_json,
        detailed_output_json=args.detailed_output_json,
        calculate_bleu=args.bleu,
        calculate_ratescore=args.ratescore,
        calculate_radgraph=args.radgraph,
        calculate_meteor=args.meteor,
        calculate_rouge=args.rouge,
        calculate_bertscore=args.bertscore,
        train_statistic_json=args.train_statistic_json,
        test_statistic_json=args.test_statistic_json
    )