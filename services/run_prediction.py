import torch
from flask import jsonify
from services.EstiMateModelHandler import EstimateModelHandler

def get_predictions(organization_id, text):
    
    try:
        handler = EstimateModelHandler()
        model = handler.get_model(organization_id)
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
