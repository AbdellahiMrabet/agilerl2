from eval import evaluate_agent

if __name__ == "__main__":
    agent = evaluate_agent(n_fogs=5, n_agents=9)
    agent.save()