# triton-translate
A Triton Inference Server model repository for machine translation

Machine translation models translate text from one language to another, a long-standing
application of neural networks. However, this history presents some challenges.
Traditionally, most models were trained on sentence-level language pairs due to memory
limitations in early approaches. Consequently, even modern architectures, capable of
handling much larger contexts, often prematurely end translations after only a sentence
or two. This necessitates the development of text segmentation techniques to divide a
client's input into manageable chunks suitable for the translation model.

The [translate](docs/translate.md) deployment is the main interface that should be
used by most clients. Currently supported models utilized by translate:

* [fastText Language Detection](docs/fasttext_language_identification.md)
  Language identification model. Currently the only available, but future versions may
  include Lingua.
* [Sentencex](docs/sentencex.md)
  Lightweight sentence segmentation. Seems to work well for most languages, with Thai
  and Khmer being noticeable exceptions given their lack of punctutation. Additional
  options like PySBD may be added in the future.
* [SeamlessM4Tv2Large](docs/seamlessm4t_text2text.md)
  Machine translation model that utilizes just the Text-to-Text portion of the
  SeamlessM4T model.
* [NLLB](docs/nllb_200_distilled_600M.md)
  Machine translation model suitable for faster, though slightly worse, translations.

## Running Tasks
Running tasks is orchestrated by using [Taskfile.dev](https://taskfile.dev/)

# Taskfile Instructions

This document provides instructions on how to run tasks defined in the `Taskfile.yml`.  

Create a task.env at the root of project to define enviroment overrides. 

## Tasks Overview

The `Taskfile.yml` includes the following tasks:

- `triton-start`
- `triton-stop`
- `model-import`
- `build-execution-env-all`
- `build-*-env` (with options: `fasttext_language_identification`, `sentencex`, `seamlessm4t_text2text`, `nllb_200_distilled_600M`)

## Task Descriptions

### `triton-start`

Starts the Triton server.

```sh
task triton-start
```

### `triton-stop`

Stops the Triton server.

```sh
task triton-stop
```

### `model-import`

Import model files from huggingface

```sh
task model-import
```

### `task build-execution-env-all`

Builds all the conda pack environments used by Triton

```sh
task build-execution-env-all
```

### `task build-*-env`

Builds specific conda pack environments used by Triton

```sh
#Example 
task build-sentencex-env
```

### `Complete Order`

Example of running multiple tasks to stage items needed to run Triton Server

```sh
task build-execution-env-all
task model-import
task triton-start
# Tail logs of running containr
docker logs -f $(docker ps -q --filter "name=triton-inference-server")
```
