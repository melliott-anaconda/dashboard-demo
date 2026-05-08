from metaflow import FlowSpec, step

class HelloFlow(FlowSpec):
    @step
    def start(self):
        print("Welcome to your first flow!")
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    HelloFlow()
