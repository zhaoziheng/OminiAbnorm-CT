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
        
    results_by_class = {}
    
    TP = FP = TN = FN = 0
    
    prompt_type = 'bbox'    # so far, we only use bbox as prompt
    
    for key, datum in tqdm(data.items()):
        
        for abnormality_group in datum['abnormality_group'].values():
            abnormality = abnormality_group[0]
            
            # Evalute the generation results of positive results
            
            if 'description' not in abnormality or abnormality['description'] == '':
                if 'label' not in abnormality or abnormality['label'] == '':
                    continue
                else:
                    gt = abnormality['label']
            else:
                gt = abnormality['description']
                
            if len(gt) == 0:
                continue

            if 'pos_refer_generation_results' not in abnormality:
                answer = ''
            else:
                answer = abnormality['pos_refer_generation_results']
                
            if "I don't see any relevant abnormalities on the image." in answer or answer == '':
                # 漏检，直接判0
                FN += 1
                bertscore = meteor = rouge1 = rougeL = radgraph_score = bleu1_score = bleu2_score = bleu3_score = bleu4_score = rate_score = 0
            else:
                TP += 1
                try:
                    
                    bertscore = meteor = rouge1 = rougeL = radgraph_score = bleu1_score = bleu2_score = bleu3_score = bleu4_score = rate_score = 0
                    if calculate_bertscore:
                        bertscore = bertscore_model.compute(predictions=[answer], references=[gt], lang='en')['f1']
                        bertscore = sum(bertscore)/len(bertscore) * 100
                        abnormality['pos_refer_generation_bertscore'] = bertscore
                    else:
                        bertscore = 0
                    
                    if calculate_meteor:
                        meteor = meteor_model.compute(predictions=[answer], references=[gt])['meteor'] * 100
                        abnormality['pos_refer_generation_meteor'] = meteor
                    else:
                        meteor = 0

                    # rouge
                    if calculate_rouge:
                        results = rouge.compute(predictions=[answer], references=[gt])
                        rouge1 = results['rouge1'] * 100
                        rougeL = results['rougeL'] * 100
                        abnormality['pos_refer_generation_rouge1'] = rouge1
                        abnormality['pos_refer_generation_rougeL'] = rougeL
                    else:
                        rouge1 = rougeL = 0
                    
                    # radgraph
                    if calculate_radgraph:
                        mean_reward, _, _, _ = f1radgraph(hyps=[answer], refs=[gt])
                        radgraph_score = sum(mean_reward)/len(mean_reward) * 100
                        abnormality['pos_refer_generation_radgraph'] = radgraph_score
                    else:
                        radgraph_score = 0
                    
                    # bleu
                    if calculate_bleu:
                        bleu1_score = bleu.compute(predictions=[answer], references=[gt], max_order=1)['bleu'] * 100
                        bleu2_score = bleu.compute(predictions=[answer], references=[gt], max_order=2)['bleu'] * 100
                        bleu3_score = bleu.compute(predictions=[answer], references=[gt], max_order=3)['bleu'] * 100
                        bleu4_score = bleu.compute(predictions=[answer], references=[gt], max_order=4)['bleu'] * 100
                        abnormality['pos_refer_generation_blue1'] = bleu1_score
                        abnormality['pos_refer_generation_blue2'] = bleu2_score
                        abnormality['pos_refer_generation_blue3'] = bleu3_score
                        abnormality['pos_refer_generation_blue4'] = bleu4_score
                    else:
                        bleu1_score = bleu2_score = bleu3_score = bleu4_score = 0
                        
                    # ratescore
                    if calculate_ratescore:
                        rate_score = ratescore.compute_score([answer], [gt])[0] * 100
                        abnormality['pos_refer_generation_ratescore'] = rate_score
                    else:
                        rate_score = 0
                    if rate_score == 50:
                        rate_score = 0
                        
                except:
                    print(f'** Answer ** {answer}')
                    print(f'** GT ** {gt}')
                    traceback.print_exc()
            
            if prompt_type not in results_by_class:
                results_by_class[prompt_type] = {}
            
            if 'category' not in abnormality:
                # 第一种统计方式
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
                    # 第一种统计方式
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
            
            # Evalute the generation results of negative results
            
        if 'neg_refer_generation_results' in datum:
            
            answer = datum['neg_refer_generation_results']
            
            if "I don't see any relevant abnormalities on the image." in answer:
                TN += 1
                bleu1_score = bleu2_score = bleu3_score = bleu4_score = meteor = rouge1 = rougeL = radgraph_score = bertscore = rate_score = 100
            else:
                FP += 1
                bertscore = meteor = rouge1 = rougeL = radgraph_score = bleu1_score = bleu2_score = bleu3_score = bleu4_score = rate_score = 0
                
            if prompt_type not in results_by_class:
                results_by_class[prompt_type] = {}
            
            if 'negative' not in results_by_class[prompt_type]:
                results_by_class[prompt_type]['negative'] = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': [], 'ratescore': [], 'bertscore': [], 'meteor': [], 'rouge1': [], 'rougeL': [], 'radgraph': []}
            results_by_class[prompt_type]['negative']['bleu1'].append(bleu1_score)
            results_by_class[prompt_type]['negative']['bleu2'].append(bleu2_score)
            results_by_class[prompt_type]['negative']['bleu3'].append(bleu3_score)
            results_by_class[prompt_type]['negative']['bleu4'].append(bleu4_score)
            results_by_class[prompt_type]['negative']['ratescore'].append(rate_score)
            results_by_class[prompt_type]['negative']['bertscore'].append(bertscore)
            results_by_class[prompt_type]['negative']['meteor'].append(meteor)
            results_by_class[prompt_type]['negative']['rouge1'].append(rouge1)
            results_by_class[prompt_type]['negative']['rougeL'].append(rougeL)
            results_by_class[prompt_type]['negative']['radgraph'].append(radgraph_score)
                    
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
                if len(results_by_class[prompt_type][category][metric_name]) > 0:
                    tmp = sum(results_by_class[prompt_type][category][metric_name]) / (len(results_by_class[prompt_type][category][metric_name])+1e-20)
                    results_by_class[prompt_type][category][metric_name] = tmp
                    metric_score_ls.append(results_by_class[prompt_type][category][metric_name])
        
        # Calculate overall scores
        results_by_class[prompt_type]['overall'] = {}
        for metric_name, metric_score_ls in zip(['bleu1', 'bleu2', 'bleu3', 'bleu4', 'ratescore', 'bertscore', 'meteor', 'rouge1', 'rougeL', 'radgraph'], [overall_bleu1, overall_bleu2, overall_bleu3, overall_bleu4, overall_ratescore, overall_bertscore, overall_meteor, overall_rouge1, overall_rougeL, overall_radgraph]):
            avg = sum(metric_score_ls) / (len(metric_score_ls)+1e-20)
            results_by_class[prompt_type]['overall'][metric_name] = avg
     
    results_by_class['bbox']['overall']['sensitivity'] = TP / (TP + FN + 1e-20)
    results_by_class['bbox']['overall']['specificity'] = TN / (TN + FP + 1e-20)
        
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
                        se_scores.append(results_by_class['bbox']['overall']['sensitivity'])
                        sp_scores.append(results_by_class['bbox']['overall']['specificity'])
                        
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
                    se_scores.append(results_by_class['bbox']['overall']['sensitivity'])
                    sp_scores.append(results_by_class['bbox']['overall']['specificity'])
                    
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
                    'sensitivity': se_scores,
                    'specificity': sp_scores
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