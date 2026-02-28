from eval import evaluate_dto

if __name__ == "__main__":
    agent = evaluate_dto(n_agents=14)
    agent.save()