from datasets import load_dataset
import numpy as np
from transformers import pipeline

# def evaluate_huggingface(dataset, template=None, model='base'):

#     if model == 'base':
#         classifier = pipeline("zero-shot-classification", device=0)
#     else:
#         classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", device=0)
    
#     correct = 0
#     predictions, gold_labels = [], []
#     for text, gold_label_idx in tqdm(zip(dataset["test_texts"], dataset["test_labels"]), total=len(dataset["test_texts"])):

#         if template is not None:
#             result = classifier(text, dataset["class_names"], multi_label=False, template=template)
#         else:
#             result = classifier(text, dataset["class_names"], multi_label=False)
#         predicted_label = result['labels'][0]
        
#         gold_label = dataset["class_names"][gold_label_idx]
        
#         predictions.append(predicted_label)
#         gold_labels.append(gold_label)
        
#         if predicted_label == gold_label:
#             correct += 1
            
#     accuracy = correct/len(predictions)
#     return accuracy

# huggingface_acc_roberta = evaluate_huggingface(dataset, model='roberta')
labels = ['far right','center right','center', 'center left','far left']

classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", device=0)
result = classifier(['Immigration causes local'], ['hyperpartisan','neutral'], multi_label=False, template=None)
print(result)