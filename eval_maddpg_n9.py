from eval import evaluate_agent

if __name__ == "__main__":
    agent = evaluate_agent(n_agents=9, algo='MADDPG')
    agent.save()