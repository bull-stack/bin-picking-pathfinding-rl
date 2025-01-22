from agent import Agent
import numpy as np

class Agents:
    def __init__(self, table, bins, radius, num_agents):
        self.agents = []
        self.table = table
        self.bins = bins
        self.radius = radius
        self.active_agent = None
        self.num_agents = num_agents
    def add_agent(self):
        """Adds a new agent to the list."""
        agent = Agent(radius=self.radius, table=self.table, bins=self.bins)
        agent.randomize_position()
        self.agents.append(agent)

    def reset(self):
        """Resets all agents and sets the first agent as active."""
        self.agents.clear()
        for _ in range(self.num_agents):
            self.add_agent()
        self.pick_new_agent()
    def get_agents(self):
        """Returns a list of all agents."""
        return self.agents

    def get_active_agent(self):
        """Returns the currently active agent."""
        return self.active_agent

    def update_active_agent(self, action):
        """Updates the position and state of the active agent."""
        self.active_agent.update(action)

    def pick_new_agent(self):
        """Switches control picks new agent, if available."""
        if not self.agents:  # Check if self.agents is empty
            self.active_agent = None
            return  # Exit the method early

        min_distance = float('inf')
        for agent in self.agents:       
            distance = agent.get_distance_to_bin()
            if distance < min_distance:
                min_distance = distance
                self.active_agent = agent
        self.active_agent.set_closest_bin()


    # def all_agents_done(self):
    #     """Checks if all agents have completed their tasks."""
    #     return all(agent.in_bin for agent in self.agents)

