from concurrent.futures import ThreadPoolExecutor, as_completed
import mteb
import numpy as np
import requests

class MyModel():
    def __init__(self, base_url: str = "http://localhost:8000/v2/models"):
        self.base_url = base_url

    def encode(
            self, sentences: list[str], prefix: str="query: ", **kwargs,
    ) -> np.ndarray:
        """
        Encodes the given sentences using the encoder
        
        Parameters
        ----------
        sentences: list[str]
            The sentences to encode
        prefix: str, optional
            The prefix to add to the sentences, by default "query: "

        Returns
        -------
        np.ndarray
            The encoded sentences
        """
        embeddings = np.zeros((len(sentences), 1024), dtype=np.float32)
        futures = {}
        with ThreadPoolExecutor(max_workers=60) as executor:
            for i, sentence in enumerate(sentences):
                inference_request = {
                    "inputs": [
                        {
                            "name": "INPUT_TEXT",
                            "shape": [1, 1],
                            "datatype": "BYTES",
                            "data": [prefix + sentence],
                        }
                    ]
                }
                future = executor.submit(requests.post,
                    url=f"{self.base_url}/e5_large_v2/infer",
                    json=inference_request,
                )
                futures[future] = i
    
            for future in as_completed(futures):
                try:
                    response_json = future.result().json()
                except Exception as exc:
                    print(f"{futures[future]} threw {exc}")
                else:
                    if "error" not in response_json:
                        data = response_json["outputs"][0]["data"]
                        embedding = np.array(data, dtype=np.float32)
                        embeddings[futures[future]] = embedding
                    else:
                        print(f"{futures[future]} threw {response_json['error']}")

        return embeddings

    def encode_queries(self, queries: list[str], **kwargs) -> np.ndarray:
        return self.encode(queries, prefix="query: ", **kwargs)

    def encode_corpus(self, corpus: list[str] | list[dict[str,str]], **kwargs) -> np.ndarray:
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [doc["title"] + " " + doc["text"] for doc in corpus]
        return self.encode(corpus, prefix="passage: ", **kwargs)

def main():
    task_list = [
        "STS15", "STS16", "SprintDuplicateQuestions", "TwitterSemEval2015",
        "ArxivClusteringS2S", "RedditClustering"
    ]
    tasks = mteb.get_tasks(tasks=task_list, languages=["eng"])

    evaluation = mteb.MTEB(tasks=tasks, eval_splits=["test"])
    model = MyModel()
    results = evaluation.run(model, output_folder=None, eval_splits=["test"])
    for result in results:
        print(f"{result.task_name} {result.score}")


if __name__ == "__main__":
    main()
