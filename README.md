# Robotic GraphNets

Adapted from https://github.com/Microsoft/gated-graph-neural-network-samples

Example of commands:
```bash
python generate_data.py --output_file=data/train9_11.json --n_graphs=100000 --min_nodes=9 --max_nodes=11
python generate_data.py --output_file=data/eval9_11.json --n_graphs=10000 --min_nodes=9 --max_nodes=11
python graphnet_dense.py --train_file=train9_11.json --valid_file=eval9_11.json --num_epochs=100
python plot_losses.py --log_file=logs/2018-05-06-21-33-41_90597_log.json

# Full evaluation. Make sure the batch size is a multiple of the number of samples.
python graphnet_dense.py --restore=logs/2018-05-06-21-33-41_90597_model_best.pickle --valid_file=eval12.json --num_epochs=0 --batch_size=100

# Quick evaluation.
python graphnet_dense.py --restore=logs/2018-05-06-21-33-41_90597_model_best.pickle --valid_file=eval9_11.json --evaluation_file=logs/eval.json --batch_size=12 --restrict_data=12
python plot_evaluation.py --evaluation_file=logs/eval.json
```
