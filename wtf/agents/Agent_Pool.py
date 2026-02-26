''' 
Created without AI support
Agent pool class to store and sample agents
'''

import random

class AgentPool():
    def __init__(self, max_agents=4, static_agents=[]):
        self.max_agents = max_agents
        self.static_agents = static_agents
        self.agents = []

    def add_agent(self, agent):
        if len(self.agents) >= self.max_agents:
            self.agents.pop(0)
        self.agents.append(agent)
    
    def get_agent(self):
        return random.choice(self.static_agents + self.agents)