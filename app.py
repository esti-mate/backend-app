from flask import Flask, request, jsonify
from EstiMateModelHandler import EstimateModelHandler
import torch
app = Flask(__name__)


@app.route("/get-predictions", methods=["POST"])
def get_predictions():

    """Get predictions for a given organization ID and text."""

    # Ensure that request is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Validate presence of organizationId and text
    if "organizationId" not in data or "text" not in data:
        return jsonify({"error": "Missing 'organizationId' or 'text' in request"}), 400

    try:
        organization_id = data["organizationId"]
        text = data["text"]
    

        handler = EstimateModelHandler()
        model =  handler.get_model(organization_id)
        tokenizer = handler.get_tokenizer()

        if model is None:
            return jsonify({"error": f'Model not found for {organization_id}'}), 404

        # Get predictions
        input_ids = tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            inference_output = model(input_ids)["logits"]
        logits = inference_output.detach().cpu().numpy()
        logits.tolist()

        response_data = {"organizationId": organization_id, "prediction": str(logits[0][0])}

        return jsonify(response_data)
    except ValueError:
        return jsonify({"error": "Invalid 'organizationId'"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3003)
