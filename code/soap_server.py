"""
SOAP Calculator Server
Handles add, subtract, multiply, divide via raw SOAP XML over HTTP.
Runs on port 8000 to avoid conflict with REST server (port 5000).
Also exposes /metrics so the benchmark can measure server-side CPU/memory
cost separately from client-side cost.
"""

import xml.etree.ElementTree as ET
from flask import Flask, request, Response, jsonify
from monitor import SystemMonitor

app = Flask(__name__)
monitor = SystemMonitor()
monitor.start()

SOAP_NS = "http://schemas.xmlsoap.org/soap/envelope/"
TNS = "calculator.soap"

WSDL = f"""<?xml version="1.0" encoding="UTF-8"?>
<definitions name="CalculatorService"
  targetNamespace="{TNS}"
  xmlns="http://schemas.xmlsoap.org/wsdl/"
  xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/"
  xmlns:tns="{TNS}"
  xmlns:xsd="http://www.w3.org/2001/XMLSchema">

  <message name="BinaryOpRequest">
    <part name="a" type="xsd:double"/>
    <part name="b" type="xsd:double"/>
  </message>
  <message name="BinaryOpResponse">
    <part name="result" type="xsd:double"/>
  </message>

  <portType name="CalculatorPortType">
    <operation name="add">
      <input message="tns:BinaryOpRequest"/>
      <output message="tns:BinaryOpResponse"/>
    </operation>
    <operation name="subtract">
      <input message="tns:BinaryOpRequest"/>
      <output message="tns:BinaryOpResponse"/>
    </operation>
    <operation name="multiply">
      <input message="tns:BinaryOpRequest"/>
      <output message="tns:BinaryOpResponse"/>
    </operation>
    <operation name="divide">
      <input message="tns:BinaryOpRequest"/>
      <output message="tns:BinaryOpResponse"/>
    </operation>
  </portType>

  <binding name="CalculatorBinding" type="tns:CalculatorPortType">
    <soap:binding style="document" transport="http://schemas.xmlsoap.org/soap/http"/>
    <operation name="add">
      <soap:operation soapAction="{TNS}#add"/>
    </operation>
    <operation name="subtract">
      <soap:operation soapAction="{TNS}#subtract"/>
    </operation>
    <operation name="multiply">
      <soap:operation soapAction="{TNS}#multiply"/>
    </operation>
    <operation name="divide">
      <soap:operation soapAction="{TNS}#divide"/>
    </operation>
  </binding>

  <service name="CalculatorService">
    <port name="CalculatorPort" binding="tns:CalculatorBinding">
      <soap:address location="http://localhost:8000/"/>
    </port>
  </service>
</definitions>"""


def soap_response(operation, result):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenv="{SOAP_NS}" xmlns:tns="{TNS}">
  <soapenv:Body>
    <tns:{operation}Response>
      <tns:result>{result}</tns:result>
    </tns:{operation}Response>
  </soapenv:Body>
</soapenv:Envelope>"""


def soap_fault(message):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenv="{SOAP_NS}">
  <soapenv:Body>
    <soapenv:Fault>
      <faultcode>soapenv:Server</faultcode>
      <faultstring>{message}</faultstring>
    </soapenv:Fault>
  </soapenv:Body>
</soapenv:Envelope>"""


@app.route("/", methods=["GET", "POST"])
def calculator():
    # Serve WSDL on GET /?wsdl
    if request.method == "GET":
        return Response(WSDL, mimetype="text/xml")

    # Parse incoming SOAP envelope
    try:
        root = ET.fromstring(request.data)
    except ET.ParseError as e:
        return Response(soap_fault(f"Invalid XML: {e}"), mimetype="text/xml", status=400)

    body = root.find(f"{{{SOAP_NS}}}Body")
    if body is None or len(body) == 0:
        return Response(soap_fault("Missing SOAP Body"), mimetype="text/xml", status=400)

    action_el = list(body)[0]
    # Strip namespace from tag to get operation name
    tag = action_el.tag
    operation = tag.split("}")[-1] if "}" in tag else tag

    # Extract operands (try with and without namespace prefix)
    def get_value(name):
        el = action_el.find(f"{{{TNS}}}{name}")
        if el is None:
            el = action_el.find(name)
        if el is None:
            raise ValueError(f"Missing operand: {name}")
        return float(el.text)

    try:
        a = get_value("a")
        b = get_value("b")
    except (ValueError, TypeError) as e:
        return Response(soap_fault(str(e)), mimetype="text/xml", status=400)

    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return Response(soap_fault("Division by zero"), mimetype="text/xml", status=400)
        result = a / b
    else:
        return Response(soap_fault(f"Unknown operation: {operation}"), mimetype="text/xml", status=400)

    return Response(soap_response(operation, result), mimetype="text/xml")


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(monitor.stats())


@app.route("/metrics/reset", methods=["POST"])
def metrics_reset():
    monitor.stop()
    monitor.start()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    print("Starting SOAP server on http://localhost:8000")
    print("WSDL available at http://localhost:8000/?wsdl")
    # debug=False: Flask's debug reloader forks a child process, which would
    # make the /metrics numbers measure the wrong process.
    app.run(debug=False, port=8000, threaded=True)
