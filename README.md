[![](logo.svg)](https://axiommath.ai/)

## Requirements

Requirements are contained in environment.yml, you can set up a micromamba env with
```
micromamba env create -f environment.yml
```
and then run
```
micromamba activate env_axplorer
```

## How to train a model from scratch?

Suppose you want to train a model for the Turan problem using data generated on the fly.

In this case it's pretty easy. From cli, you can just run

```bash
python train.py \
    --env_name square \
    --exp_name square_exp \
    --N 30 \
    --encoding_tokens single_integer \
    --max_len 100 \
    --temperature '0.6' \
    --inc_temp '0.1'
```

And wait until the model has been trained. In the command above, `exp_name`is simply the name of the experiment and you can give any name you like, while `env_name` is the name of the problem to be considered (`square` is the name given internally to the Turan problem, a list of `env_name` corresponding to problems already implemented in this repo can be found below at the end of this Readme).

## What if I need to restart a model?

In this case, you need to specify the same configuration and explicitly define exp_id to match the exp_id of the stopped experiment.

```bash
python train.py \
    --env_name square \
    --exp_name square_exp \
    --exp_id 2026_01_22_18_51_53 \
    --N 30 \
    --encoding_tokens single_integer \
    --max_len 100 \
    --temperature '0.6' \
    --inc_temp '0.1'
```

Under the hood, we store the model and optimizer checkpoints in that folder, so it's easy to resume model training.

## What if I want to use higher quality data?

In this case, you can proceed in two steps: first generate your data and then train the model. This is particularly helpful when you need either a large amount of data or when data generation is costly. For example the command

```bash
python train.py \
    --env_name square \
    --exp_name square_exp \
    --N 30 \
    --gensize 10000000 \
    --pop_size 10000000 \
    --data_generation_only true
```

generates and saves 10 million examples. Later you can copy the train and test data to the new experiment.

If you want to generate 10 million examples but store only the best 100,000, you can run

```bash
python train.py \
    --env_name square \
    --exp_name square_exp \
    --N 30 \
    --gensize 10000000 \
    --pop_size 100000 \
    --data_generation_only true
```

Generating data may be expensive and long. After creating data, please make a copy to the training folder. You'll likely need to use this golden data over and over. If you don't make a copy, the training job will actually **OVERWRITES** the data.

## I want to know more. What's happening under the hood?

Of course, so you can adapt to your use case (see below to adapt to implement your own math problem).

The training pipeline iteratively improves solutions by combining neural network learning with a classical search algorithm. Here's how it works:

### Overview

```
                      ┌─────────────────────────────────────────────────────┐
                      │                   During each epoch                 │
                      │                                                     │
┌──────────┐          │   ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│ Generate │          │   │  Train   │    │  Sample  │    │  Search  │      │
│  Initial │─────────────►│  Model   │───►│  Model   │───►│ & Score  │      │
│   Data   │          │   └──────────┘    └──────────┘    └──────────┘      │
└──────────┘          │         ▲                               │           │
                      │         │         ┌──────────┐          │           │
                      │         └─────────│  Select  │◄─────────┘           │
                      │                   │   Best   │                      │
                      │                   └──────────┘                      │
                      └─────────────────────────────────────────────────────┘
```

### Step-by-step breakdown

1. **Initial Data Generation** (first epoch only)
   - Generate `gensize` random valid examples using environment-specific greedy construction.
   - Each example is scored based on the optimization objective.
   - Keep the top `pop_size` examples as the initial training set.

2. **Training Phase** (each epoch)
   - Tokenize the training data into sequences the model can process.
   - Train a decoder-only transformer for a number of steps (i.e. gradient updates). This number is given by `max_steps`.
   - During this training, the model learns to predict the next token given previous tokens.
   - This captures patterns present in high-scoring examples.

3. **Sampling Phase** (each epoch)
   - Sample `num_samples_from_model` new sequences from the trained model.
   - Use temperature-controlled sampling for diversity.
   - Decode sequences back into problem-specific objects.

4. **Local Search Phase** (each epoch)
   - Apply local search to sampled objects to fix constraint violations.
   - Optionally improve valid samples with additional greedy steps.
   - Score all processed samples.

5. **Selection Phase** (each epoch)
   - Combine new samples with existing training data.
   - Remove duplicates if `keep_only_unique=True`.
   - Select the top `pop_size` examples by score.
   - These become the training data for the next epoch.

6. **Temperature Adjustment**
   - If too many duplicates are generated, increase temperature by `inc_temp`. This encourages more exploration when the model converges.

## How can I use it on my own math problem ?

To do so, you only need to create an environment corresponding to your math problem. A step-by-step guide with an example is provided in `new_envs.ipynb`.


## Explanation of different flags

The full list of flags for each specific environment and more details about what they represent can be found in the `register_args` method of each different environment.

The full list of flags for the model architecture and training parameters can be found in the method `get_parser` of `train.py`. Here, you can find references of the most important flags you should tune for your specific problem.

- Training parameters:
  - `gensize` is the number of initial data points.
  - `max_epochs` is the maximum number of epochs.
  - `max_steps` is the number of training steps per epoch.
  - `num_samples_from_model` is the number of samples from the trained model after each epoch.
  - `pop_size` is the number of examples kept after each epoch (when selecting the best examples).
  - `env_name` is the math problem considered as defined in `envs/__init__.py`.

- Model parameters:
  - `n_layer` is the number of layers in the decoder-only model.
  - `n_embd` is the vector dimension where we project each token.
  - If the problem is permutation invariant, you can set `no_positional=True`.
  - `max_len` is the maximum length supported by the model.

- Optimization parameters:
  - `batch_size` is the model batch size used for training.
  - `learning_rate` and `weight_decay` are two parameters used in AdamW.

- Sampling/generation parameters:
  - `temperature` is the starting sampling temperature.
  - `temp_span` is the temperature span across sampling.
  - `inc_temp` is the temperature increment whenever the model generates too many duplicates.
  - If you want to avoid duplicate objects, you should set `keep_only_unique=True`.

- Local search parameters:
  - During local search, it's possible to remove invalid examples completely (with `redeem_only=False` and `always_search=False`), just fix the invalid examples (with `redeem_only=True` and `always_search=False`), or fix and try to improve the model samples (with `always_search=True`).

- Environment-specific parameters. When you set up a new problem, there are a few parameters to configure:

  - `k` is the problem-specific dimension: for the Turan problem you work on pairs of nodes, so `k=2`. For the no-5-points-on-a-sphere problem, each point lives in [N]^3, so `k=3`.
  - `are_coordinates_symmetric` determines whether coordinates can be shuffled. For the Turan problem, (i, j) and (j, i) represent the same edge on the graph, so `are_coordinates_symmetric=True`. In the other problems listed in the repository, `are_coordinates_symmetric=False`.
  - `encoding_tokens` specifies how to encode the data for the model. If `encoding_tokens=single_integer`, you encode the coordinates as a single number. For instance, for the Turan problem, you encode each edge as Ni+j. For the no-5-points-on-a-sphere problem, you encode each point as N²i+Nj+k. If `encoding_tokens=sequence_k_tokens`, you encode each coordinate as a separate token. For example, if the edge is (i, j), the model will see i and j as two separate tokens. If `encoding_tokens=adjacency`, you encode the full dense adjacency matrix of a graph.
  - `make_object_canonical` is set to True when you want to deduplicate different objects into their canonical form. This is needed only when `keep_only_unique=True`.
  - `augment_data_representation` is set to True when, after making the object canonical, you want to feed the model with one of the many possible different object representations. For example, for no-5-points-on-a-sphere, there are up to 48 different representations of the same cube. During training, the model sees different data representations to ensure model robustness.


## Available environments

| Environment | Description | `env_name` |
|-------------|-------------|------------|
| Square-free graphs | Maximize edges in a graph with no 4-cycles | `square` |
| Isosceles-free point sets | Maximize points in a grid [N]^2 with no isosceles triangles | `isosceles` |
| Sphere point sets | Maximize points in a grid [N]^3 with no 5 points on a sphere | `sphere` |

## License

This repository uses the Apache-2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgements

The original code of PatternBoost was written by François Charton, Jordan S. Ellenberg, Adam Zsolt Wagner, and Geordie Williamson, and can be found [here](https://github.com/zawagner22/transformers_math_experiments).
