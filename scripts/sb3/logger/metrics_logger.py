from stable_baselines3.common.callbacks import BaseCallback

import os
# MetricsLoggerCallback for saving training metrics
class MetricsLoggerCallback(BaseCallback):
    """
    Custom callback for logging training metrics, saving the model, 
    and exporting metrics to a CSV file for further analysis.
    """
    def __init__(self, 
                 num_agents: int,
                 check_freq: int, 
                 save_path: str, 
                 n_steps: int, 
                 verbose: int = 1, 
                 model_name: str = "model"):
        super(MetricsLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.n_steps = n_steps
        self.model_name = model_name
        self.num_agents = num_agents
    def _init_callback(self) -> None:
        """
        Create necessary directories for saving models and logs.
        """
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at every environment step.
        """
        # Save the model at the specified frequency
        if self.n_calls % self.check_freq == 0:
            model_save_dir = os.path.join(self.save_path, self.model_name)
            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(model_save_dir, f"best_{self.num_agents}_{self.n_calls}.zip")
            self.model.save(model_path)
            if self.verbose:
                print(f"Model saved at step {self.n_calls} to {model_path}")
        return True
