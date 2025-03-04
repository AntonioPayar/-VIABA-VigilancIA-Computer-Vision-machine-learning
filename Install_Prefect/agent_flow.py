from prefect import flow

@flow(name="Kubernetes Agent Deployment")
def agent_flow():
    pass  # No necesitamos hacer nada aquí

if __name__ == "__main__":
    agent_flow()