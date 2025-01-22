import pygame 
from bin import Bin
class Renderer:
    def __init__(self, screen_width, screen_height, colors, title="Table Pathfinding"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.title = title
        self.colors = colors
        self.screen = None
        self.clock = None
        
    def generate_rect(self, x, y, width, height):
        return pygame.Rect(x, y, width, height)

    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()

    def draw_table(self, table):
        pygame.draw.rect(
            self.screen,
            self.colors["table"],
            pygame.Rect(
                table.top_left[0], table.top_left[1], table.table_width, table.table_height
            ),
        )

    def draw_bins(self, bins, target_bin, table):
        for i, bin in enumerate(bins.get_bins()):
            color = (
                self.colors["bin_reached"] if bin == target_bin 
                else self.colors["bin_in_use"] if bin.in_use 
                else self.colors["bin"]
            )
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(
                    bin.position[0] - bin.width // 2,
                    bin.position[1] - bin.height // 2,
                    bin.width,
                    bin.height
                ),
            )
            # Draw goal entry zone
            bin_entry_zone = bin.get_bin_entry_zone(table)
            pygame.draw.rect(self.screen, (100, 100, 255), bin_entry_zone, 2)

            arrow_start = bin.position + (bin.entry_offset * bin.entry_vector)
            arrow_end = arrow_start + (50 * bin.entry_vector)
            pygame.draw.line(self.screen, self.colors["arrow"], arrow_start.astype(int), arrow_end.astype(int), 2)
            
    def draw_object(self, agents):
        for agent in agents.get_agents():
            color = (
                self.colors["agent"] if agent == agents.get_active_agent() else self.colors["arrow"]
            )
            pygame.draw.circle(
                self.screen, color, agent.position.astype(int), agent.radius
            )

    def render(self, table, bins, agents):
        if self.screen is None:
            self.initialize()

        self.screen.fill(self.colors["background"])

        # Draw the table, bins, and object
        self.draw_table(table)
        self.draw_bins(bins, agents.get_active_agent().get_target_bin(), table)
        self.draw_object(agents)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
