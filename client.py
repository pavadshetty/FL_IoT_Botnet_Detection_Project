import flwr as fl
from model import create_model
from dataset import load_data

X, y = load_data()
model = create_model()

class IoTClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X, y, epochs=1, batch_size=16, verbose=0)
        return model.get_weights(), len(X), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc = model.evaluate(X, y, verbose=0)
        return loss, len(X), {"accuracy": acc}


if __name__ == "__main__":
	fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=IoTClient(),
)


