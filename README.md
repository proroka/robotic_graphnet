# Robotic GraphNets

Adapted from https://github.com/Microsoft/gated-graph-neural-network-samples

Example of commands:
```bash
python generate_data.py --output_file=data/eval8_10.json --n_graphs=100000 --min_nodes=8 --max_nodes=10
python graphnet_dense.py --train_file=data/train8_10.json --valid_file=data/eval8_10.json --num_epochs=5
python plot_losses.py --log_file=logs/2018-05-06-18-26-57_86993_log.json

# Full evaluation.
python graphnet_dense.py --restore=logs/2018-05-06-18-26-57_86993_model_best.pickle --valid_file=eval11.json --num_epochs=0

# Quick evaluation.
python graphnet_dense.py --restore=logs/2018-05-06-18-26-57_86993_model_best.pickle --valid_file=eval8_10.json --evaluation_file=logs/eval.json --batch_size=12 --restrict_data=12
python plot_evaluation.py --evaluation_file=logs/eval.json
```
