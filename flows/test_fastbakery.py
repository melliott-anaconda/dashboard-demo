from metaflow import FlowSpec, step, pypi

class TestFastBakeryFlow(FlowSpec):
    @pypi(packages={"pandas": "2.0.3", "numpy": "1.24.3", "scikit-learn": "1.3.2"})
    @step
    def start(self):
        import pandas as pd
        import numpy as np
        print(f"pandas={pd.__version__}, numpy={np.__version__}")
        print("Fast-bakery environment works on K8s!")
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    TestFastBakeryFlow()
