if __name__ == "__main__":
    import model_builder
    import json

    with open("model.json", "r") as f:
        config = json.load(f)
    mb = model_builder.ModelBuilder()
    model = mb.build(config, (28,28,1), (10,))
    print(model.summary())