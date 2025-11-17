# Improving cancer driver gene prediction using biological knowledge-guided prompts for LLM
Bioprompt is a novel cancer driver gene identification method that leverages biological knowledge-guided prompts for large language models (LLMs) and integrates multiomics data via contrastive learning.

This repo is for the source code of "Improving cancer driver gene prediction by integrating large language models and multiomics data with biological knowledge-guided prompts". \
Paper Link: 

Setup
------------------------
The setup process for Bioprompt requires the following steps:
### Download
Download Bioprompt.  The following command clones the current Bioprompt repository from GitHub:

    git clone https://github.com/weiba/Bioprompt.git

### Environment Settings
> python=3.9.19
>
> torch==2.0.1+cu118
>
> numpy==1.26.4
>
> pandas==2.2.1
>
> ollama==0.3.3
>
> scipy==1.13.0
>
> scikit-learn==1.4.2

Ollama list

> NAME            ID              SIZE
>
> gemma2:latest   ff02c3702f32    5.4 GB
>
> llama3.1:latest 42182419e950    4.7 GB
>
> 

GPU: GeForce RTX 3090 24G	CUDA Version: 12.0

CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz

### Usage
#### Step 1: Process GO Data

Run the GO data processing notebook to generate foundational GO data sentences:

> jupyter notebook LLM/go_data_process.ipynb

#### Step 2: Optimize Prompts

Fine-tune and optimize the biological prompts using:

> jupyter notebook LLM/prompt_finetune.ipynb

#### Step 3: Generate LLM Outputs

Use the optimized prompts with the large language model to process gene-related vocabulary:

> python LLM/LLMoutput.py

#### Step 4: Generate BioBERT Embeddings

Process the LLM-generated vocabulary using BioBERT to obtain embedding vectors:

> python bert_embedding.py

### Step 5: Run Bioprompt Framework

Execute the optimized cancer driver gene prediction model:

> **For MTGCN+Prompt model:**
>
> python bioprompt/bioprompt/mtgcn+prompt/MTGCN_LLM.py
>
> **For MNGCL+Prompt model:**
>
> python bioprompt/bioprompt/mngcl+prompt/train_MNGCL_cv_LLM.py

## Results

The framework has been validated on pan-cancer and 15 individual cancer datasets, demonstrating improved performance over existing models. Ablation studies confirm the critical role of GO-guided prompts in generating valuable gene semantic information.

## Project Structure

> bioprompt/
> ├── baseline/                    # Baseline models for comparison
> │   ├── DISFusion/              # DISFusion model implementation
> │   ├── MNGCL-master/           # MNGCL model implementation
> │   └── MTGCN-master/           # MTGCN model implementation
> ├── bioprompt/                   # Main Bioprompt framework
> │   ├── mngcl+prompt/           # MNGCL integrated with LLM prompts
> │   └── mtgcn+prompt/           # MTGCN integrated with LLM prompts
> ├── data/                        # Data directories
> │   ├── bert_out/               # BioBERT embedding outputs
> │   ├── DISfusion_Data/         # Data for DISFusion baseline
> │   ├── MNGCL_data/             # Data for MNGCL baseline
> │   └── MTGCN_data/             # Data for MTGCN baseline
> └── LLM/                         # Large Language Model components
>     ├── bert_model/             # Pre-trained BioBERT model
>     ├── csv/                    # CSV outputs from different LLMs
>     │   ├── gemma2/
>     │   ├── gpt3/
>     │   └── llama3.1/
>     ├── txt/                    # Text data and prompts
>     │   ├── data/              # Raw data files
>     │   └── prompt/            # Prompt templates
>     ├── go_data_process.ipynb  # GO data processing notebook
>     ├── prompt_finetune.ipynb  # Prompt optimization notebook
>     ├── LLMoutput.py           # LLM inference script
>     └── bert_embedding.py      # BioBERT embedding generation

### Cite
```

```



