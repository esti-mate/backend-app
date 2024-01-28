from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/get-predictions', methods=['POST'])
def get_predictions():
    # Ensure that request is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Validate presence of organizationId and text
    if 'organizationId' not in data or 'text' not in data:
        return jsonify({"error": "Missing 'organizationId' or 'text' in request"}), 400

    organization_id = data['organizationId']
    text = data['text']

    # Here you can process the data or perform any actions needed
    # For now, it just echoes back the received data

    response_data = {
        "organizationId": organization_id,
        "text": text
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
