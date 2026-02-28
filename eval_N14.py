from eval import evaluate_agent

if __name__ == "__main__":
    agent = evaluate_agent(n_agents=14)
    agent.save()