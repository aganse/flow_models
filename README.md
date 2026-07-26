# flow_models

Normalizing-flow models are invertible neural networks (INNs) — generative
models that support exact likelihood computation (unlike GANs and VAEs) by
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
imagery references an example and a key paper (cited at the bottom of this
README).  This implementation uses TensorFlow Probability, which provides clean
building blocks for these models.  See also:
- ["Flow_models: Overview / Introduction"](http://research.ganse.org/datasci/flow_models)
- ["Flow_models 1: Distribution mapping"](http://research.ganse.org/datasci/flow_models/flow_models_1.html)
- ["Flow_models 2: Generative image modeling"](http://research.ganse.org/datasci/flow_models/flow_models_2.html)

There are different ways to train/run this repo's models, as referred to in sections A and B below:
- Option 1: directly in Python (local in python virtural environment, no Docker)
- Option 2: locally in Docker (CPU, for smoke-testing the container)
- Option 3: locally in Docker (GPU, on a GPU-equipped machine or EC2 instance)
- Option 4: cloud training via SageMaker Training Jobs (for single full runs)
- Option 5: cloud training via AWS Batch (for queued/parallel job sweeps)


### A. To install/prepare

1. **(Option 3 only; local/Docker/GPU) Set up a GPU-equipped machine.**  For
   local GPU Docker runs, follow
   [these instructions](https://github.com/aganse/py_tf2_gpu_dock_mlflow/blob/main/doc/aws_ec2_install.md)
   to configure a GPU-enabled EC2 instance.  For cloud runs (Options 4–5) no
   local GPU or EC2 setup is needed — training runs entirely on AWS.

2. **(Option 1 only; local/pyenv/no-Docker) Create the Python environment and
   install dependencies:**
    ```
    make create-env           # creates .venvN and pip-installs requirements.txt
    source .venvN/bin/activate
    make install-dev          # installs requirements-dev.txt for tests/linting
    ```
   Not needed for Docker or cloud runs — dependencies are baked into the image.

3. **Get training images** (for image-based applications like `train_flowmodels2.py`):

    Of course you can use whatever images you want.  For my experimentation I
    used the nicely curated Kaggle dataset
    [animal-faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces)
    which contains ~5000 cats, ~5000 dogs, and ~5000 misc wild animals (fox,
    leopard, lion, tiger, wolf, etc).

    - for run options 1–3 (local): place images in some directory (`data/`) and
      set `IMAGES_PATH` to that path.
    - for run options 4–5 (cloud): upload to S3 and set `IMAGES_PATH=s3://mybucket/prefix`.

    Use the following directory structure (but note subdirectories are merged by
    the data generator, so `cat` and `beachball` images mix together in the below).
    This is not a supervised learning model so labels are not used for training,
    but for validation this directory structure provides a convenient labeling of
    what image contents are in the dataset.

    ```
    data/                <-- or s3://mybucket/prefix/
        train/
            cat/
        val/
            beachball/   <-- these show up as outliers in latent space
            cat/         <-- these don't
    ```

### B. To run the training

**Option 1 — directly in Python (local, in python virtual environment, no Docker):**

1. Enter the python environment: `source .venvN/bin/activate`
2. Optionally set `export TF_CPP_MIN_LOG_LEVEL=2` to reduce TensorFlow log noise.
3. Edit the parameter dicts near the top of `train_flowmodelsN.py` to set
   desired hyperparameters. Refer to [`doc/config.md`](doc/config.md) for a
   description of every parameter. (Note: `params/paramsN.json` is only read
   in Docker/SageMaker runs, not in this direct-Python mode.)
4. Run `python train_flowmodelsN.py` (where `N` is 1, 2, etc.).

**Option 2 — locally in Docker (CPU, for smoke-testing the container):**

Edit `params/paramsN.json`, then:
```bash
make build-cpu
make run-local SCRIPT=N DEVICE=cpu
```

**Option 3 — locally in Docker (GPU, on a GPU-equipped machine):**

Edit `params/paramsN.json`, then:
```bash
make build-gpu
make run-local SCRIPT=N DEVICE=gpu
```

**Option 4 — cloud training via SageMaker Training Jobs (recommended for full runs):**

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

**Option 5 — cloud training via AWS Batch (for queued/parallel job sweeps):**

Edit the parameter dicts in `train_flowmodelsN.py`, rebuild and push the image
(`make run-build [BRANCH=main] [DEVICE=gpu]`), then run `make run-batchjob`.

(Params-file override support for Batch, equivalent to Options 2–4, will be
added soon.)

See [`awsbatch-support/README.md`](awsbatch-support/README.md) for full
setup and submission instructions.

###

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


### C. Key references

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


### D. Misc other notes and refs that I perused

* A RealNVP tutorial found in Github:  https://github.com/MokkeMeguru/glow-realnvp-tutorial/blob/master/tips/RealNVP_mnist_en.ipynb

* Kang ISSP 2020 paper on NICE INNs:  https://jaekookang.me/issp2020/

* Eric Jang Normalizing Flows Tutorial:  https://blog.evjang.com/2018/01/nf2.html

* Lilian Weng Flow-based Deep Generative Models tutorial:  http://lilianweng.github.io/posts/2018-10-13-flow-models

* Jaekoo Kang's flow_based_models NICE & RealNVP repo:  https://github.com/jaekookang/flow_based_models

* Jaekoo Kang's INNs repo (Ardizzone implementation):  https://github.com/jaekookang/invertible_neural_networks

* Chanseok Kang's RealNVP notebook:
    https://colab.research.google.com/github/goodboychan/goodboychan.github.io/blob/main/_notebooks/2021-09-08-01-AutoRegressive-flows-and-RealNVP.ipynb#scrollTo=NNun_3RT3A56

* RealNVP implementation example in Stackoverflow:
    https://stackoverflow.com/questions/57261612/better-way-of-building-realnvp-layer-in-tensorflow-2-0

* Brian Keng's Normalizing Flows with Real NVP article, more mathematical:
    https://bjlkeng.io/posts/normalizing-flows-with-real-nvp/#modified-batch-normalization

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
