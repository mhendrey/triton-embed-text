# Embed Text
This is a BLS deployment that lets a client send text and get back a vector
embedding. Currently this supports both the [Multilingual E5](multilingual_e5_large.md)
and the [SigLIP Text](siglip_text.md) model. Multilingual E5 is the default
`embed_model`. In order to use the SigLIP text embedding model, you must specify that
as a request parameter (see examples below)

Because dynamic batching has been enabled for these Triton Inference Server
deployments, clients simply send each request separately. This simplifies the code for
the client, see examples below, yet they reap the benefits of batched processing. In
addition, this allows for controlling the GPU RAM consumed by the server.

Optional Request Parameters:
* `embed_model`: str, optional, default="multilingual_e5_large"
  Specify which embedding model to use. Choices are `multilingual_e5_large` or
  `siglip_text`.

## Multilingual E5 Text Embeddings
For optimal performance, all text sent must have either "query: " or "passage: "
prepended to your text. Notice that there is a space after the colon. See the **NOTE**
in [Multilingual E5](multilingual_e5_large.md) about when to use one vs the other.

### Send Single Request
```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"

text = (
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape."
)
inference_json = {
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
text_embed_response = requests.post(
    url=f"{base_url}/embed_text/infer",
    json=inference_json,
)

response_json = text_embed_response.json()
"""
{
    "model_name": "embed_text",
    "model_version": "1",
    "outputs": [
        {
            "name": "EMBEDDING",
            "shape": [1, 1024],
            "datatype": "FP32",
            "data": [
                0.00972023,
                -0.00810159,
                -0.00420804,
                ...,
            ]
        }
    ]
}
"""
text_embedding = np.array(response_json["outputs"][0]["data"]).astype(np.float32)
```

### Sending Many Requests
When embedding multiple text strings, use multithreading to send requests in parallel,
which maximizes throughput and efficiency.

NOTE: You will encounter an "OSError: Too many open files" if you send a lot of
requests. Typically the default ulimit is 1024 on most system. Either increase this
using `ulimit -n {n_files}`, or don't create too many futures before you processing
some of them.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"

texts = [
    "query: Did Palpatine die in The Return of the Jedi?",
    "query: What causes ocean tides?",
    "query: How does photosynthesis work in plants?",
    "query: Explain the concept of supply and demand in economics.",
    "query: What is the difference between weather and climate?",
    "query: How does the human immune system defend against pathogens?",
    "query: How are artificial intelligence models created?",
]

futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, text in enumerate(texts):
        infer_request = {
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [text],
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/embed_text/infer",
            json=infer_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response_json = future.result().json()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        if "error" not in response_json:
            embedding = response_json["outputs"][0]["data"]
            embedding = np.array(embedding).astype(np.float32)
            embeddings[futures[future]] = embedding
        else:
            print(f"{futures[future]} threw {response_json['error']}")
print(embeddings)
```

### Performance Analysis
There is some data in [data/embed_text](../data/embed_text/multilingual_text.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container to measure the throughput of the Multilingual E5 Text model.

```
sdk-container:/workspace perf_analyzer \
    -m embed_text \
    -v \
    --input-data data/embed_text/multilingual_text.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000 \
    --bls-composing=multilingual_e5_large
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 245.8 infer/sec. Avg latency: 242271 usec (std 53572 usec). 
  * Pass [2] throughput: 248.161 infer/sec. Avg latency: 241291 usec (std 52339 usec). 
  * Pass [3] throughput: 248.395 infer/sec. Avg latency: 242541 usec (std 53301 usec). 
  * Client: 
    * Request count: 17821
    * Throughput: 247.452 infer/sec
    * Avg client overhead: 0.01%
    * Avg latency: 242034 usec (standard deviation 27317 usec)
    * p50 latency: 258591 usec
    * p90 latency: 264519 usec
    * p95 latency: 266524 usec
    * p99 latency: 280632 usec
    * Avg HTTP time: 242014 usec (send 42 usec + response wait 241972 usec + receive 0 usec)
  * Server: 
    * Inference count: 17821
    * Execution count: 1906
    * Successful request count: 17821
    * Avg request latency: 243266 usec (overhead 135125 usec + queue 21340 usec + compute 86801 usec)

  * Composing models: 
  * multilingual_e5_large, version: 1
      * Inference count: 17838
      * Execution count: 17838
      * Successful request count: 17837
      * Avg request latency: 107801 usec (overhead 0 usec + queue 21340 usec + compute 86801 usec)

    * Composing models: 
    * multilingual_e5_large_model, version: 1
        * Inference count: 17851
        * Execution count: 1683
        * Successful request count: 17851
        * Avg request latency: 102068 usec (overhead 10 usec + queue 20406 usec + compute input 123 usec + compute infer 81527 usec + compute output 1 usec)

    * multilingual_e5_large_tokenize, version: 1
        * Inference count: 17853
        * Execution count: 3427
        * Successful request count: 17853
        * Avg request latency: 6105 usec (overhead 22 usec + queue 934 usec + compute input 92 usec + compute infer 5055 usec + compute output 1 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 247.452 infer/sec, latency 242034 usec

### Validation
See the [Multilingual E5](./multilingual_e5_large.md) validation section for details.

## SigLIP Text Embedding Model
Since this is not the default `embed_model`, you must pass it explicitly as a request
parameter. Otherwise everything works the same way. This model is useful in conjunction
with the SigLIP Vision embedding model to enable image search via natural language
or performing zero-shot image classification.

### Sending Single Request
Here's an example of sending a string of text.

```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"

text = (
    "A photo of a person in an astronaut suit riding a "
    + "unicorn on the surface of the moon."
)
inference_json = {
    "parameters": {"embed_model": "siglip_text"},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
text_embed_response = requests.post(
    url=f"{base_url}/embed_text/infer",
    json=inference_json,
)

response_json = text_embed_response.json()
"""
{
    "model_name": "embed_text",
    "model_version": "1",
    "outputs": [
        {
            "name": "EMBEDDING",
            "shape": [1, 1152],
            "datatype": "FP32",
            "data": [
                1.19238,
                0.90869,
                0.44360,
                ...,
            ]
        }
    ]
}
"""
text_embedding = np.array(response_json["outputs"][0]["data"]).astype(np.float32)

```

### Sending Many Requests
When embedding multiple text strings, use multithreading to send requests in parallel,
which maximizes throughput and efficiency.

NOTE: You will encounter an "OSError: Too many open files" if you send a lot of
requests. Typically the default ulimit is 1024 on most system. Either increase this
using `ulimit -n {n_files}`, or don't create too many futures before you processing
some of them.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"

alt_texts = [
    "Close-up of a student sitting next to her practice piano.",
    "Toddler excitedly eating a slice of cheese pizza larger than her head!",
    "Wide angle from above showing the entire garden, guests, and wedding party as the bride walks down the aisle.",
    "A smiling, elderly mother places her hand on her daughter’s cheek as she prepares for the wedding ceremony.",
    "An embracing couple in formal attire smile as they look at a Rainbow lorikeet perched on the man’s finger.",
    "High school senior in blue jeans and a t-shirt sits on a rock for his senior portraits.",
]

futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, alt_text in enumerate(alt_texts):
        infer_request = {
            "parameters": {"embed_model": "siglip_text"},
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [alt_text],
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/embed_text/infer",
            json=infer_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response_json = future.result().json()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        if "error" not in response_json:
            embedding = response_json["outputs"][0]["data"]
            embedding = np.array(embedding).astype(np.float32)
            embeddings[futures[future]] = embedding
        else:
            print(f"{futures[future]} threw {response_json['error']}")
print(embeddings)
```
### Performance Analysis
There is some data in [data/embed_text](../data/embed_text/imagenet_categories.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container. Because we need to send the request parameter, we must specify using the
gRPC protocol which supports this option in `perf_analyzer`.

```
sdk-container:/workspace perf_analyzer \
    -m embed_text \
    -v \
    -i grpc \
    --input-data data/embed_text/imagenet_categories.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000 \
    --bls-composing=siglip_text \
    --request-parameter=embed_model:siglip_text:string
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 1257.01 infer/sec. Avg latency: 47656 usec (std 16454 usec). 
  * Pass [2] throughput: 1282.46 infer/sec. Avg latency: 46829 usec (std 16034 usec). 
  * Pass [3] throughput: 1269.33 infer/sec. Avg latency: 47268 usec (std 15954 usec). 
  * Client: 
    * Request count: 91526
    * Throughput: 1269.61 infer/sec
    * Avg client overhead: 0.08%
    * Avg latency: 47248 usec (standard deviation 7701 usec)
    * p50 latency: 53086 usec
    * p90 latency: 64418 usec
    * p95 latency: 67270 usec
    * p99 latency: 76386 usec
    * Avg gRPC time: 47236 usec (marshal 3 usec + response wait 47233 usec + unmarshal 0 usec)
  * Server: 
    * Inference count: 91526
    * Execution count: 7231
    * Successful request count: 91526
    * Avg request latency: 48743 usec (overhead 24050 usec + queue 7233 usec + compute 17460 usec)

  * Composing models: 
  * siglip_text, version: 1
      * Inference count: 91526
      * Execution count: 91526
      * Successful request count: 91526
      * Avg request latency: 24475 usec (overhead 0 usec + queue 7233 usec + compute 17460 usec)

  * Composing models: 
  * siglip_text_model, version: 1
      * Inference count: 91526
      * Execution count: 6702
      * Successful request count: 91526
      * Avg request latency: 19452 usec (overhead 11 usec + queue 6274 usec + compute input 93 usec + compute infer 13073 usec + compute output 1 usec)

  * siglip_text_tokenize, version: 1
      * Inference count: 91527
      * Execution count: 13838
      * Successful request count: 91527
      * Avg request latency: 5282 usec (overhead 30 usec + queue 959 usec + compute input 94 usec + compute infer 4197 usec + compute output 1 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 1269.61 infer/sec, latency 47248 usec


### Validation
See the [SigLIP Text](./siglip_text.md) validation section for details.


## E5 Large v2 Text Embeddings
For optimal performance, all text sent must have either "query: " or "passage: "
prepended to your text. Notice that there is a space after the colon. See the **NOTE**
in [E5 Large v2](e5_large_v2.md) about when to use one vs the other.

### Send Single Request
```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"

text = (
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape."
)
inference_json = {
    "parameters": {"embed_model": "e5_large_v2"},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
text_embed_response = requests.post(
    url=f"{base_url}/embed_text/infer",
    json=inference_json,
)

response_json = text_embed_response.json()
"""
{
    "model_name": "embed_text",
    "model_version": "1",
    "outputs": [
        {
            "name": "EMBEDDING",
            "shape": [1, 1024],
            "datatype": "FP32",
            "data": [
                0.033483,
               -0.082776,
                0.056425,
                ...,
            ]
        }
    ]
}
"""
text_embedding = np.array(response_json["outputs"][0]["data"]).astype(np.float32)
```

### Sending Many Requests
When embedding multiple text strings, use multithreading to send requests in parallel,
which maximizes throughput and efficiency.

NOTE: You will encounter an "OSError: Too many open files" if you send a lot of
requests. Typically the default ulimit is 1024 on most system. Either increase this
using `ulimit -n {n_files}`, or don't create too many futures before you processing
some of them.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"

texts = [
    "query: Did Palpatine die in The Return of the Jedi?",
    "query: What causes ocean tides?",
    "query: How does photosynthesis work in plants?",
    "query: Explain the concept of supply and demand in economics.",
    "query: What is the difference between weather and climate?",
    "query: How does the human immune system defend against pathogens?",
    "query: How are artificial intelligence models created?",
]

futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, text in enumerate(texts):
        infer_request = {
            "parameters": {"embed_model": "e5_large_v2"},
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [text],
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/embed_text/infer",
            json=infer_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response_json = future.result().json()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        if "error" not in response_json:
            embedding = response_json["outputs"][0]["data"]
            embedding = np.array(embedding).astype(np.float32)
            embeddings[futures[future]] = embedding
        else:
            print(f"{futures[future]} threw {response_json['error']}")
print(embeddings)
```

### Performance Analysis
There is some data in [data/embed_text](../data/embed_text/e5_large_v2_text.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container to measure the throughput of the E5 Large v2 Text model.

```
sdk-container:/workspace perf_analyzer \
    -m embed_text \
    -v \
    -i grpc \
    --input-data data/embed_text/e5_large_v2_text.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000 \
    --bls-composing=e5_large_v2 \
    --request-parameter=embed_model:e5_large_v2:string
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 254.506 infer/sec. Avg latency: 233597 usec (std 52831 usec). 
  * Pass [2] throughput: 259.104 infer/sec. Avg latency: 233241 usec (std 51816 usec). 
  * Pass [3] throughput: 256.531 infer/sec. Avg latency: 234004 usec (std 51862 usec). 
  * Client: 
    * Request count: 18490
    * Throughput: 256.714 infer/sec
    * Avg client overhead: 0.02%
    * Avg latency: 233613 usec (standard deviation 26944 usec)
    * p50 latency: 248912 usec
    * p90 latency: 258105 usec
    * p95 latency: 262214 usec
    * p99 latency: 286423 usec
    * Avg gRPC time: 233602 usec (marshal 2 usec + response wait 233600 usec + unmarshal 0 usec)
  * Server: 
    * Inference count: 18490
    * Execution count: 2014
    * Successful request count: 18490
    * Avg request latency: 235605 usec (overhead 134852 usec + queue 23033 usec + compute 77720 usec)

  * Composing models: 
  * e5_large_v2, version: 1
      * Inference count: 18491
      * Execution count: 18491
      * Successful request count: 18491
      * Avg request latency: 100414 usec (overhead 0 usec + queue 23033 usec + compute 77720 usec)

  * Composing models: 
  * e5_large_v2_model, version: 1
      * Inference count: 18491
      * Execution count: 1795
      * Successful request count: 18491
      * Avg request latency: 93983 usec (overhead 11 usec + queue 22009 usec + compute input 125 usec + compute infer 71836 usec + compute output 1 usec)

  * e5_large_v2_tokenize, version: 1
      * Inference count: 18521
      * Execution count: 3519
      * Successful request count: 18521
      * Avg request latency: 6802 usec (overhead 21 usec + queue 1024 usec + compute input 92 usec + compute infer 5664 usec + compute output 1 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 256.714 infer/sec, latency 233613 usec


### Validation
See the [E5 Large v2](./e5_large_v2.md) validation section for details.
