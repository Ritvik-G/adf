import re
import requests
import xml.etree.ElementTree as ET

SOAP_NS = "http://schemas.xmlsoap.org/soap/envelope/"
TNS = "calculator.soap"

OPERATION_MAP = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
}


def make_soap_call(base_url, operation, a, b, verbose=True):
    """Make a raw SOAP call for the given arithmetic operation"""
    envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenv="{SOAP_NS}" xmlns:tns="{TNS}">
  <soapenv:Body>
    <tns:{operation}>
      <tns:a>{a}</tns:a>
      <tns:b>{b}</tns:b>
    </tns:{operation}>
  </soapenv:Body>
</soapenv:Envelope>"""

    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": f'"{TNS}#{operation}"',
    }

    response = requests.post(base_url, data=envelope.encode("utf-8"), headers=headers)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    result_el = root.find(f".//{{{TNS}}}result")
    result = float(result_el.text)
    if verbose:
        print(f"SOAP Call: {operation}({a}, {b}) = {result}")
    return result


def evaluate_expression(expression, soap_base_url="http://localhost:8000/", verbose=True):
    """
    Evaluate expression by repeatedly finding and solving innermost operations via SOAP

    Args:
        expression (str): Mathematical expression to evaluate
        soap_base_url (str): Base URL of the SOAP service
        verbose (bool): Print step-by-step execution

    Returns:
        float: Final result of the expression
    """
    expression = expression.replace(" ", "")

    if verbose:
        print(f"Original Expression: {expression}\n")

    # Pattern to match: number operator number
    pattern = r"\(?\s*(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*\)?"

    step = 1
    while True:
        match = re.search(pattern, expression)

        if not match:
            result = float(expression.strip("()"))
            if verbose:
                print(f"\nFinal Result: {result}")
            return result

        a, operator, b = match.groups()
        operation = OPERATION_MAP[operator]

        if verbose:
            print(f"Step {step}:")

        result = make_soap_call(soap_base_url, operation, a, b, verbose=verbose)

        expression = expression[: match.start()] + str(result) + expression[match.end() :]

        if verbose:
            print(f"Expression now: {expression}\n")

        step += 1
