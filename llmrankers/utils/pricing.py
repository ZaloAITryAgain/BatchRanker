PRICING_PER_1K_TOKENS = {
    "gpt-4o-mini": {
        "input": 0.000150,
        "output": 0.000600,
    },
    "gpt-4o": {
        "input": 0.00250,
        "output": 0.01000,
    },
    "gpt-3.5-turbo": {
        "input": 0.003,
        "output": 0.006,
    },
}


def get_pricing(model_name_or_path, num_input_tokens, num_output_tokens):
    input_price = PRICING_PER_1K_TOKENS[model_name_or_path]["input"] * num_input_tokens / 1_000
    output_price = PRICING_PER_1K_TOKENS[model_name_or_path]["output"] * num_output_tokens / 1_000
    return {
        "total": input_price + output_price,
        "input": input_price,
        "output": output_price,
    }
