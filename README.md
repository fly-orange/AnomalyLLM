# Large Language Model-based Knowledge Distillation For Time Series Anomaly Detection


## Preparation
1. Installation 
<pre>
git clone https://github.com/salesforce/Merlion.git‘
cd Merlion
pip install salesforce-merlion==1.1.1
pip install -r requirements.txt
</pre>

2. Load GPT2  
Download GPT2 from https://huggingface.co/docs/transformers/model_doc/gpt2  
Put it in "models/HOC/hoc_network/gpt2"
## 


## Operation 
<pre>
python train.py --selected_dataset dataset_name --method method_name
</pre>

## Reference
1. https://github.com/ruiking04/COCA
2. https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All