"""Minimal evaluator-signed source fixture for the isolated Defend worker."""


def evaluate(inputs: object, config: dict[str, object]) -> str:
    """Return sealed evaluator evidence without ambient process or file capabilities."""
    if type(inputs).__name__ != "VerifiedEvaluationInputs" or type(config) is not dict:
        raise TypeError("fixture worker inputs are not exact")
    if set(config) != {"delay_iterations", "request_base64"}:
        raise ValueError("fixture worker configuration differs")
    delay_iterations = config["delay_iterations"]
    request_base64 = config["request_base64"]
    if type(delay_iterations) is not int or type(request_base64) is not str:
        raise TypeError("fixture worker configuration is invalid")
    accumulator = 0
    for value in range(delay_iterations):
        accumulator ^= value
    if accumulator < 0:
        raise AssertionError("unreachable")
    return request_base64
