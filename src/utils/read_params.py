from json import load


def read_params() -> None:
    with open('params.json') as json_file:
        params = load(json_file)
    return params
