# Large Language Models Resources

## Foundational Papers

### Architecture & Attention
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017) - **The transformer paper**
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Rush (2018) - Line-by-line implementation guide

### GPT Family (Causal LM)
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - Radford et al. (2018) - GPT-1
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al. (2019) - GPT-2
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - Brown et al. (2020) - GPT-3
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - Ouyang et al. (2022) - InstructGPT/ChatGPT

### BERT Family (Masked LM)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Devlin et al. (2019) - **The BERT paper**
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) - Liu et al. (2019)
- [ALBERT: A Lite BERT](https://arxiv.org/abs/1909.11942) - Lan et al. (2020) - Parameter-efficient BERT
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) - He et al. (2021)

### T5 & Encoder-Decoder
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683) - Raffel et al. (2020) - **Text-to-Text Transfer Transformer**
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) - Lewis et al. (2020)

### Other Important Architectures
- [XLNet: Generalized Autoregressive Pretraining](https://arxiv.org/abs/1906.08237) - Yang et al. (2019) - Permutation LM
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) - Dai et al. (2019)

## Tokenization

### Papers
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - Sennrich et al. (2016) - **BPE**
- [SentencePiece: A simple and language independent approach](https://arxiv.org/abs/1808.06226) - Kudo & Richardson (2018)
- [Japanese and Korean Voice Search](https://research.google/pubs/pub37842/) - Schuster & Nakajima (2012) - WordPiece

### Resources
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) - Interactive visualization

## Scaling & Training

### Scaling Laws
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Kaplan et al. (2020) - OpenAI scaling laws
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Hoffmann et al. (2022) - **Chinchilla paper**, optimal tokens per parameter
- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) - Wei et al. (2022)

### Large-Scale Models
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Touvron et al. (2023)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) - Touvron et al. (2023)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) - Chowdhery et al. (2022)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) - OpenAI (2023)

## Fine-Tuning & Adaptation

### Parameter-Efficient Fine-Tuning
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al. (2021) - **LoRA paper**
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - Dettmers et al. (2023)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) - Houlsby et al. (2019) - Adapter layers
- [Prefix-Tuning: Optimizing Continuous Prompts](https://arxiv.org/abs/2101.00190) - Li & Liang (2021)
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) - Lester et al. (2021)

### Instruction Tuning & RLHF
- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) - Wei et al. (2022)
- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) - Ouyang et al. (2022)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - Bai et al. (2022) - Anthropic's approach
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) - Rafailov et al. (2023) - Simpler alternative to RLHF

## Prompting & In-Context Learning

### Papers
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Wei et al. (2022)
- [Self-Consistency Improves Chain of Thought](https://arxiv.org/abs/2203.11171) - Wang et al. (2022)
- [ReAct: Synergizing Reasoning and Acting in LLMs](https://arxiv.org/abs/2210.03629) - Yao et al. (2022)
- [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804) - Liu et al. (2021)

### Guides
- [Prompt Engineering Guide](https://www.promptingguide.ai/) - Comprehensive resource
- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) - Official guide
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering) - Claude-specific best practices

## Evaluation

### Papers
- [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) - Hendrycks et al. (2021)
- [Beyond the Imitation Game (BIG-bench)](https://arxiv.org/abs/2206.04615) - Srivastava et al. (2022)
- [HumanEval: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) - Chen et al. (2021)
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) - Lin et al. (2022)

### Leaderboards
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - Hugging Face
- [HELM: Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/) - Stanford CRFM

## Books

- [Speech and Language Processing (3rd ed.)](https://web.stanford.edu/~jurafsky/slp3/) - Jurafsky & Martin - **Free online**, comprehensive NLP textbook
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) - Tunstall, von Werra, Wolf (2022) - Practical Hugging Face guide

## Courses

- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) - Free lectures and notes
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) - Free, hands-on transformer training
- [Fast.ai: From Deep Learning Foundations to Stable Diffusion](https://course.fast.ai/) - Practical deep learning including LLMs
- [DeepLearning.AI: ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) - Free short course

## Implementation Resources

### Libraries
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - **Primary library** for pretrained models
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [JAX/Flax](https://github.com/google/flax) - Google's framework for LLM training
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Fine-tuning toolkit
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) - Easy fine-tuning for LLaMA and others

### Code Repositories
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy's minimal GPT implementation (pedagogical)
- [minGPT](https://github.com/karpathy/minGPT) - Even more minimal GPT
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - Line-by-line implementation

## Blogs & Articles

- [Jay Alammar's Blog](https://jalammar.github.io/) - Visual explanations (Illustrated Transformer, GPT-2, BERT)
- [Lil'Log](https://lilianweng.github.io/) - Lilian Weng's technical blog (OpenAI)
- [The Gradient](https://thegradient.pub/) - ML research magazine
- [Weights & Biases Blog](https://wandb.ai/site/articles) - Practical ML engineering
- [Sebastian Ruder's Blog](https://ruder.io/) - NLP research updates

## Tools

- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualization
- [Gradio](https://www.gradio.app/) - Quick model demos and UIs
- [vLLM](https://github.com/vllm-project/vllm) - Fast inference serving
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - Hugging Face's inference server

## Community

- [Hugging Face Forums](https://discuss.huggingface.co/) - Active community for transformers
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) - Research discussions
- [EleutherAI Discord](https://www.eleuther.ai/get-involved) - Open-source LLM development
