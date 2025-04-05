# HRGR

Code for the paper: "It Takes a Graph to Know a Graph: Rewiring for Homophily with a Reference Graph".
More details can be found in our paper [TODO](https://arxiv.org/******)

*first author (TODO)*, *second author (TODO)*

We also provide implementations of [SDRF](https://arxiv.org/pdf/2111.14522), [FoSR](https://arxiv.org/pdf/2210.11790), and [BORF](https://proceedings.mlr.press/v202/nguyen23c/nguyen23c.pdf) with our clustering approach, enabling these rewiring algorithms to scale to large graphs.

## Installing
You can install all the required packages by executing the following command:

```bash
pip install -r requirements.txt
```

## Experiments
Run node classification experiments using our HRGR method, along with the baseline algorithms [SDRF](https://arxiv.org/pdf/2111.14522), [FoSR](https://arxiv.org/pdf/2210.11790), and [BORF](https://proceedings.mlr.press/v202/nguyen23c/nguyen23c.pdf) all implemented with our clustering strategy to support large-scale graphs.
All run scripts are located in the `algorithms_run_scripts/` folder.

### HRGR (ours)
```bash
python HRGR_node_classification.py --dataset <str> --model <str> --lr <float> --weight_decay <float> \
                                   --scheme <'D', 'PDP'> --data_eps <float> --sample_rate <float> \
                                   --add_or_delete <'add', 'delete'> --cluster_size <int> 
```
### SDRF
```bash
python sdrf_node_classification.py --dataset <str> --model <str> --lr <float> --weight_decay <float> \
                                   --sdrf_max_iter_ratio <float> --sdrf_tau <int> --sdrf_removal_bound <float> \
                                   --cluster_size <int> 
```
### FoSR
```bash
python fosr_node_classification.py --dataset <str> --model <str> --lr <float> --weight_decay <float> \
                                   --fosr_num_iterations <int> --cluster_size <int> 
```
### BORF
```bash
python borf_node_classification.py --dataset <str> --model <str> --lr <float> --weight_decay <float> \
                                   --borf_num_iterations <int> --borf_batch_add <int> --borf_batch_remove <int> \
                                   --cluster_size <int>
```
### No Rewiring (Standard Training)
```bash
python None_node_classification.py --dataset <str> --model <str> --lr <float> --weight_decay <float>
```
