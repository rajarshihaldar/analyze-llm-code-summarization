# llm_analysis

Code for the paper "Analyzing the Performance of Large Language Models on Code Summarization" accepted in LREC-COLING 2024.

## Getting scores from Llama 2:

First run inference.

```
cd llama/
python inference.py
```

Then compute BLEU score.
```
python postprocess.py
python eval.py
```

## Getting scores from PaLM 2:

```
cd palm/
python inference.py
python postprocess.py
python eval.py
```

## Running CodeT5

```
cd codet5/
python train.py
python eval.py --task translate
python eval.py --task score
```

## Getting BERTScores

For PaLM and Llama 2:
```
cd bertscore/
python eval_bertscore.py
```

For CodeT5:
```
cd bertscore/
python codet5_bertscore.py
```