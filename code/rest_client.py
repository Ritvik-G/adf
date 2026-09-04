import re
import requests


def make_rest_call(api_base_url, operation, a, b, verbose=True):
    """Make REST API call for the operation"""
    operation_map = {
        '+': 'add',
        '-': 'subtract',
        '*': 'multiply',
        '/': 'divide'
    }

    endpoint = f"{api_base_url}/{operation_map[operation]}"

    try:
        response = requests.post(endpoint, json={"a": float(a), "b": float(b)})
        response.raise_for_status()
        result = response.json()['result']
        if verbose:
            print(f"REST Call: {operation_map[operation]}({a}, {b}) = {result}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error making REST call: {e}")
        raise


def evaluate_expression(expression, api_base_url="http://localhost:5000", verbose=True):
    """
    Evaluate expression by repeatedly finding and solving innermost operations

    Args:
        expression (str): Mathematical expression to evaluate
        api_base_url (str): Base URL of the REST API
        verbose (bool): Print step-by-step execution

    Returns:
        float: Final result of the expression
    """
    expression = expression.replace(' ', '')

    if verbose:
        print(f"Original Expression: {expression}\n")

    # Pattern to match: number operator number
    pattern = r'\(?\s*(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*\)?'

    step = 1
    while True:
        # Find the innermost operation
        match = re.search(pattern, expression)

        if not match:
            # No more operations, return final result
            result = float(expression.strip('()'))
            if verbose:
                print(f"\nFinal Result: {result}")
            return result

        # Extract the operation
        a, operator, b = match.groups()

        if verbose:
            print(f"Step {step}:")

        # Make REST call
        result = make_rest_call(api_base_url, operator, a, b, verbose=verbose)

        # Replace the operation with its result in the expression
        expression = expression[:match.start()] + str(result) + expression[match.end():]

        if verbose:
            print(f"Expression now: {expression}\n")

        step += 1
