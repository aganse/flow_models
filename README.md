# flow_models

Normalizing-flow models are invertible neural networks (INNs) — generative
models that provide exact likelihood computation (unlike GANs and VAEs) by
ensuring all transformations are invertible with efficiently computable
Jacobians.  This enables generative image modeling, anomaly detection,
classification, parameter estimation, and Bayesian inverse problems — all
from the same architecture, just with different input/output partitioning.
INNs allow for such a range of applications - a real Swiss-army-knife of
the modeling world that I'm recently fascinated with.  <IMG SRC="doc/sak.jpg" ALT="" WIDTH=25>

<IMG SRC="doc/INNfig_3sec.gif" ALT="Seven applications of flow-model in different forms" WIDTH=700>

These normalizing-flow models transform complex data distributions into more
tractable ones (usually Gaussian) in which it's feasible to do probabilistic
calculations.  The diagram above shows seven such applications; each frame's
details reference an example and a key paper (cited at the bottom of this
README).  The implementation here uses TensorFlow Probability, which provides
clean building blocks for these models.  I've been jotting up some writeups on
each case on my website as I experiment with it:
- ["Flow_models: Overview / Introduction"](http://research.ganse.org/datasci/flow_models)
- ["Flow_models 1: Distribution mapping"](http://research.ganse.org/datasci/flow_models/flow_models_1.html)
- ["Flow_models 2: Generative image modeling"](http://research.ganse.org/datasci/flow_models/flow_models_2.html)


## A. Prep/Setup

### Platform Options
To train the models in this repo you have a number of options for platforms:
1. directly in Python (local in python virtural environment, no Docker)
2. locally in Docker (on home machine or on a cloud instance, CPU or GPU)
3. cloud training via SageMaker Training Jobs (for full single runs)
4. cloud training via AWS Batch (for queued/parallel job sweeps)

Local runs (options 1-3) could either be on an average home machine (eg for
options 1 or 2 for the lightweight application 1 or tests), or on a GPU-enabled
machine or EC2 instance (for options 1 or 3 for application 2 and up).  For the
latter (options 1 or 3 on GPU) in AWS, you could follow [these
instructions](https://github.com/aganse/py_tf2_gpu_dock_mlflow/blob/main/doc/aws_ec2_install.md)
to configure a GPU-enabled EC2 instance.  For submitted cloud runs (Options
4–5), no local GPU or EC2 setup is needed — training runs entirely on separate
AWS services.

For local runs without Docker (option 1), you'll need to create a python virtual
environment and install the dependencies:
```
make create-env             # creates .venvN and pip-installs requirements.txt
source .venvN/bin/activate  # enter your desired venv (the N increments with each create-env call)
make install-dev            # if you wish to do dev/tests/linting (installs requirements-dev.txt)
```
This is not needed for Docker or cloud runs — dependencies are baked into the
Docker image.

### Training Data
For training images (for image-based applications like `train_flowmodels2.py`),
of course you can use whatever images you want.  For my example experimentation
I used the nicely curated Kaggle dataset
[animal-faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces) which
contains ~5000 cats, ~5000 dogs, and ~5000 misc wild animals (fox, leopard,
lion, tiger, wolf, etc).

- for local runs (options 1–3): place images in some directory (e.g. `data/`) and
  set `IMAGES_PATH` to that path.
- for cloud runs (options 4–5): upload to S3 and set `IMAGES_PATH=s3://mybucket/prefix`.

Use the following directory structure in `data/` (but note subdirectories are
merged by the data generator, so `cat` and `beachball` images mix together in
the below).  This is not a supervised learning model so labels are not used for
training, but for validation this directory structure provides a convenient
labeling of what images contain what in the dataset.
```
data/                <-- or s3://mybucket/prefix/
    train/
        cat/
    val/
        beachball/   <-- these show up as outliers in latent space
        cat/         <-- these don't
```


## B. Running the Training

Follow the directions for the respective run option (same numbers as the list
above in Section A1. Platform Options):

### 1. directly in Python (local, in python virtual environment, no Docker):
1. Enter the python environment: `source .venvN/bin/activate`
2. Optionally set `export TF_CPP_MIN_LOG_LEVEL=2` to reduce TensorFlow log noise.
3. Edit the parameter dicts near the top of `train_flowmodelsN.py` to set
   desired hyperparameters. Refer to [`doc/config.md`](doc/config.md) for a
   description of every parameter. (Note: `params/paramsN.json` is only read
   in Docker/SageMaker runs, not in this direct-Python mode.)
4. Run `python train_flowmodelsN.py` (where `N` is 1, 2, etc.).

### 2. locally in Docker (CPU or GPU):
(CPU for application 1 or smoke-testing the container; GPU for application 2+
on a GPU-equipped machine.)

Edit `params/paramsN.json`, then:
```bash
make local-build DEVICE=cpu         # or DEVICE=gpu
make run-local SCRIPT=N DEVICE=cpu  # or DEVICE=gpu
```

### 3. cloud training via SageMaker Training Jobs (recommended for full runs):
Edit `params/paramsN.json`, then:
```bash
make run-build BRANCH=myfeature DEVICE=gpu  # default BRANCH=main, default DEVICE=gpu
                                            # (also note build-status and build-logs targets)
make sm-run SCRIPT=N                        # uses :latest (by default points to main-gpu image)
make sm-run SCRIPT=N TAG=mybranch-gpu       # use a specific image tag (of form gitbranch-device)
                                            # (also note sm-list-jobs, sm-status, sm-logs targets)
```
See [`sagemaker-support/README.md`](sagemaker-support/README.md) for setup
and monitoring details.

### 4. cloud training via AWS Batch (for queued/parallel job sweeps):
Edit the parameter dicts in `train_flowmodelsN.py`, rebuild and push the image
(`make run-build [BRANCH=main] [DEVICE=gpu]`), then run `make run-batchjob`.

(Params-file override support for Batch, equivalent to Options 2–4, will be
added soon.)

See [`awsbatch-support/README.md`](awsbatch-support/README.md) for full
setup and submission instructions.


### Reinstantiating model from saved weights
If both `model/model_arch.json` and `model/model_weights.weights.h5` are being
saved (note the latter may not be depending on the value of
training_params["save_model_weights"] because that weights file can be huge),
the model object could be reinstantiated via:
```
from tensorflow.keras.models import model_from_json
with open("model/model_arch.json") as f:
    model = model_from_json(f.read(), custom_objects={"FlowModel": FlowModel})
model.load_weights("model/model_weights.weights.h5")
```


## C. References

### Papers
* Distribution mapping and generative image modeling with INNs
  - [RealNVP paper](https://arxiv.org/pdf/1605.08803)
  - [NICE paper](https://arxiv.org/pdf/1410.8516)
  - [Glow paper](https://arxiv.org/pdf/1807.03039)
* Generative classification and ill-conditioned parameter estimation with INNs
  - [Ardizzone 2019 INNs paper](https://arxiv.org/pdf/1808.04730)
* Bayesian inverse problems with INNs
  - [Zhang & Curtis 2021 JGR paper](https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2021JB022320)
* TensorFlow Probability components
  - [tfp.bijectors.RealNVP API](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP)

### Other notes/etc.
* [A RealNVP tutorial found in Github](https://github.com/MokkeMeguru/glow-realnvp-tutorial/blob/master/tips/RealNVP_mnist_en.ipynb)
* [Kang ISSP 2020 paper on NICE INNs](https://jaekookang.me/issp2020)
* [Eric Jang Normalizing Flows Tutorial](https://blog.evjang.com/2018/01/nf2.html)
* [Lilian Weng Flow-based Deep Generative Models tutorial](http://lilianweng.github.io/posts/2018-10-13-flow-models)
* [Jaekoo Kang's flow_based_models NICE & RealNVP repo](https://github.com/jaekookang/flow_based_models)
* [Jaekoo Kang's INNs repo (Ardizzone implementation)](https://github.com/jaekookang/invertible_neural_networks)
* [Chanseok Kang's RealNVP notebook](https://colab.research.google.com/github/goodboychan/goodboychan.github.io/blob/main/_notebooks/2021-09-08-01-AutoRegressive-flows-and-RealNVP.ipynb#scrollTo=NNun_3RT3A56)
* [RealNVP implementation example in Stackoverflow](https://stackoverflow.com/questions/57261612/better-way-of-building-realnvp-layer-in-tensorflow-2-0)
* [Brian Keng's Normalizing Flows with Real NVP article, more mathematical](https://bjlkeng.io/posts/normalizing-flows-with-real-nvp/#modified-batch-normalization)
* Helpful rundown of bits-per-dimension in Papamakarios et al 2018 paper
  "Masked Autoregressive Flow for Density Estimation": https://arxiv.org/pdf/1705.07057
  section E.2; note they call it "bits per pixel".  They express in
  average log likelihoods too (note that's actually what the NLL value
  is at very bottom of this script here).
* Note in NICE paper regarding flow_steps: "Examining the Jacobian, we
  observe that at least three coupling layers are necessary to allow all
  dimensions to influence one another. We generally use four."  And they
  used 1000-5000 nodes in their hidden layers, with 4-5 hidden layers per
  coupling layer.
