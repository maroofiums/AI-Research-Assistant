import mlflow
import time

mlflow.set_experiment("AI-Research-Assistant")

def start_rag_run():
    return mlflow.start_run()

def log_rag_metrics(
    question,
    answer,
    retrieved_docs,
    latency
):

    mlflow.log_params("question",question)

    mlflow.log_metric(
        "retrieved_chunks",
        len(retrieved_docs)
    )
    
    mlflow.log_metric(
        "latency_secs",
        latency
    )

    mlflow.log_text(
        answer,
        "answer.txt"
    )


