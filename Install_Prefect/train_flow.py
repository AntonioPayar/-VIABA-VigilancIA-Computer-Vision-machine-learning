from prefect import flow
import os

@flow(log_prints=True)
def train_model(data_path: str, model_name: str = "my_model"):
    print(f"Entrenando modelo {model_name} con datos de {data_path}")
    print("¡Modelo entrenado!")
    return model_name

if __name__ == "__main__":
    train_model.deploy(
        name="viaba-training-deployment",
        work_pool_name="KubTest",  
        image="reitzz/viaba-training-flow:latest",  
        push=False,  
        parameters={"data_path": "/data/training_data.csv"}, 
        job_variables={"env": {"MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI")}}
    )